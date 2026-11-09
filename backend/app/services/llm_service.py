"""
Intelligent RAG System — LLM Generation Service
================================================
Sends retrieved context to an LLM and generates a grounded answer.

WHAT IS GROUNDING?
An LLM without RAG answers from its training data — which may be outdated
or hallucinated. "Grounding" means: we FORCE the LLM to answer ONLY based
on the passages we provide. If the passages don't contain the answer,
the LLM should say so, not invent one.

HOW WE ACHIEVE GROUNDING:
1. We build a system prompt that tells the LLM to ONLY use provided context
2. We inject the retrieved passages as numbered references
3. We ask the LLM to cite which passage its answer came from
4. We tell the LLM to say "I don't know" if context is insufficient

WHY GROQ?
- Free tier with generous limits
- Very fast inference (they use custom H100 chips)
- Compatible with OpenAI's API format
- Good for prototyping and demos

RETRIEVAL-ONLY MODE:
If no LLM key is configured (GROQ_API_KEY is empty or provider="none"),
the system returns the retrieved passages without generating an answer.
This is still useful — it shows what the system found.
"""

import time
import uuid
from typing import Optional

from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


def build_rag_prompt(question: str, passages: list[dict]) -> str:
    """
    Build the context-injected prompt for the LLM.
    
    The prompt structure is critical for grounded generation:
    1. Tell the LLM its role and rules
    2. Provide numbered source passages
    3. Ask the specific question
    4. Remind it to cite sources and admit uncertainty
    
    Args:
        question: The user's question
        passages: Retrieved chunks, each with 'text', 'filename', 'page_number'
    
    Returns:
        A formatted prompt string
    """
    # Build the context block
    context_parts = []
    for i, passage in enumerate(passages, start=1):
        filename = passage.get("filename", "Unknown document")
        page = passage.get("page_number")
        page_str = f", Page {page}" if page else ""
        context_parts.append(
            f"[Source {i}: {filename}{page_str}]\n{passage['text']}"
        )

    context_block = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are a knowledgeable assistant that answers questions using ONLY the provided source passages.

RULES:
1. Answer ONLY based on the information in the provided sources below.
2. If the sources do not contain enough information to answer the question, say clearly: "The provided documents do not contain sufficient information to answer this question."
3. When you use information from a source, mention which source (e.g., "According to Source 1..." or "As stated in Source 2...").
4. Do NOT make up information not found in the sources.
5. Be concise but complete. Use clear paragraphs.

---
SOURCES:
{context_block}
---

QUESTION: {question}

ANSWER:"""
    return prompt


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    reraise=True,
)
def call_groq_api(prompt: str, model: str, temperature: float) -> tuple[str, int]:
    """
    Call the Groq API with retry logic.
    
    Returns: (answer_text, tokens_used)
    
    WHY TENACITY?
    API calls can fail temporarily (rate limits, network issues).
    tenacity automatically retries with exponential backoff:
    Wait 2s → retry → wait 4s → retry → wait 8s → give up
    This makes the system more resilient without complex error handling code.
    """
    settings = get_settings()
    client = Groq(api_key=settings.groq_api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a precise, citation-aware assistant. Always ground your answers in the provided sources.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=temperature,
        max_tokens=1024,
    )

    answer = response.choices[0].message.content.strip()
    tokens = response.usage.total_tokens if response.usage else 0
    return answer, tokens


class GenerationResult:
    """Result of the generation step."""
    def __init__(
        self,
        answer: str,
        llm_used: bool,
        model_used: Optional[str],
        tokens_used: int,
        generation_time_ms: float,
    ):
        self.answer = answer
        self.llm_used = llm_used
        self.model_used = model_used
        self.tokens_used = tokens_used
        self.generation_time_ms = generation_time_ms


def generate_answer(
    question: str,
    retrieved_passages: list[dict],
) -> GenerationResult:
    """
    Generate an answer from retrieved passages.
    
    If LLM is configured: sends question + context to LLM.
    If not: returns a retrieval-only response with a summary message.
    
    Args:
        question: The user's original question
        retrieved_passages: Chunks from vector_store.search_similar_chunks()
    
    Returns:
        GenerationResult with the answer and metadata
    """
    settings = get_settings()
    start = time.time()

    if not retrieved_passages:
        # No relevant passages found — tell the user clearly
        return GenerationResult(
            answer=(
                "No relevant information was found in the knowledge base for this question. "
                "Please check that relevant documents have been uploaded and indexed."
            ),
            llm_used=False,
            model_used=None,
            tokens_used=0,
            generation_time_ms=round((time.time() - start) * 1000, 2),
        )

    if not settings.llm_enabled:
        # Retrieval-only mode — LLM not configured
        logger.info("LLM not configured — returning retrieval-only response")
        passages_summary = "\n\n".join(
            f"• [{p.get('filename', 'Unknown')}] {p['text'][:300]}..."
            for p in retrieved_passages[:3]
        )
        return GenerationResult(
            answer=(
                f"**Retrieval-Only Mode** — LLM not configured.\n\n"
                f"The following {len(retrieved_passages)} relevant passage(s) were found "
                f"in your knowledge base:\n\n{passages_summary}\n\n"
                f"To enable AI-generated answers, add your GROQ_API_KEY to the .env file."
            ),
            llm_used=False,
            model_used=None,
            tokens_used=0,
            generation_time_ms=round((time.time() - start) * 1000, 2),
        )

    # === LLM Generation ===
    # Truncate context to avoid exceeding token limits
    max_chars = settings.max_context_tokens
    truncated_passages = []
    total_chars = 0
    for p in retrieved_passages:
        if total_chars + len(p["text"]) > max_chars:
            break
        truncated_passages.append(p)
        total_chars += len(p["text"])

    prompt = build_rag_prompt(question, truncated_passages)

    try:
        answer, tokens_used = call_groq_api(
            prompt=prompt,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
        )
        logger.info(
            "LLM generation complete",
            model=settings.llm_model,
            tokens=tokens_used,
            passages_used=len(truncated_passages),
        )
        return GenerationResult(
            answer=answer,
            llm_used=True,
            model_used=settings.llm_model,
            tokens_used=tokens_used,
            generation_time_ms=round((time.time() - start) * 1000, 2),
        )
    except Exception as e:
        logger.error("LLM generation failed", error=str(e))
        return GenerationResult(
            answer=(
                f"The AI generation step encountered an error: {str(e)}\n\n"
                f"The retrieved passages are still available in the Sources section."
            ),
            llm_used=False,
            model_used=None,
            tokens_used=0,
            generation_time_ms=round((time.time() - start) * 1000, 2),
        )

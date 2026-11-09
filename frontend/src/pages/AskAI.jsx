import { useState, useRef } from 'react'
import { askQuestion } from '../services/api.js'

/**
 * AskAI Page — The main RAG query interface.
 * Shows: question input, generated answer, expandable source citations.
 */
export default function AskAI({ systemStatus }) {
  const [question, setQuestion] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [expandedSources, setExpandedSources] = useState({})
  const inputRef = useRef(null)

  const exampleQuestions = [
    'What is the attendance policy?',
    'What are the examination rules?',
    'How are grades calculated?',
    'What disciplinary actions exist for academic misconduct?',
  ]

  const handleSubmit = async (q) => {
    const question_text = (q || question).trim()
    if (!question_text) return
    if (question_text.length < 3) {
      setError('Please enter a more specific question (at least 3 characters).')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)
    setExpandedSources({})
    if (q) setQuestion(q)

    try {
      const data = await askQuestion(question_text)
      setResult(data)
    } catch (err) {
      setError(err.detail || err.message)
    } finally {
      setLoading(false)
    }
  }

  const toggleSource = (idx) => {
    setExpandedSources(prev => ({ ...prev, [idx]: !prev[idx] }))
  }

  const getScoreColor = (score) => {
    if (score >= 0.7) return 'var(--accent-green)'
    if (score >= 0.4) return 'var(--accent-blue)'
    return 'var(--accent-yellow)'
  }

  return (
    <>
      <div className="page-header">
        <h1>✦ Ask AI</h1>
        <p>Ask questions and get answers grounded in your knowledge base</p>
      </div>

      <div className="page-content">
        {/* LLM status banner */}
        {systemStatus && !systemStatus.llm_enabled && (
          <div className="alert alert-warning" style={{ marginBottom: 16 }}>
            ⚠️ <strong>Retrieval-only mode</strong> — No LLM configured.
            Results will show retrieved passages without AI-generated answers.
            Add <code>GROQ_API_KEY</code> to <code>.env</code> to enable AI answers.
          </div>
        )}

        {systemStatus?.total_documents === 0 && (
          <div className="alert alert-info" style={{ marginBottom: 16 }}>
            ℹ️ No documents indexed yet. Upload documents in <strong>Knowledge Base</strong> first.
          </div>
        )}

        {/* Question input */}
        <div className="card" style={{ marginBottom: 20 }}>
          <div className="card-title" style={{ marginBottom: 12 }}>Ask a Question</div>
          <div className="query-input-container">
            <textarea
              ref={inputRef}
              className="query-input"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="e.g. What is the attendance policy for students?"
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  handleSubmit()
                }
              }}
              rows={2}
              disabled={loading}
            />
            <button
              className="btn btn-primary"
              onClick={() => handleSubmit()}
              disabled={loading || !question.trim()}
              style={{ alignSelf: 'flex-end', minWidth: 100 }}
            >
              {loading ? (
                <><span className="loading-spinner" /> Searching</>
              ) : '✦ Ask'}
            </button>
          </div>

          {/* Example questions */}
          {!result && !loading && (
            <div style={{ marginTop: 12 }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', marginBottom: 8 }}>
                Try an example:
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                {exampleQuestions.map((q, i) => (
                  <button
                    key={i}
                    className="btn btn-secondary btn-sm"
                    onClick={() => handleSubmit(q)}
                    disabled={loading}
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Error state */}
        {error && (
          <div className="alert alert-error">❌ {error}</div>
        )}

        {/* Loading state */}
        {loading && (
          <div className="card">
            <div style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '12px 0' }}>
              <span className="loading-spinner" style={{ width: 22, height: 22 }} />
              <div>
                <div style={{ fontSize: '0.9rem', fontWeight: 500 }}>Searching knowledge base...</div>
                <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginTop: 2 }}>
                  Embedding query → semantic search → {systemStatus?.llm_enabled ? 'generating answer' : 'formatting results'}
                </div>
              </div>
            </div>
            <div className="loading-skeleton" style={{ height: 80, borderRadius: 'var(--radius-sm)', marginTop: 8 }} />
          </div>
        )}

        {/* Results */}
        {result && !loading && (
          <>
            {/* Answer box */}
            <div className="answer-box">
              <div className="answer-box-header">
                <span>{result.llm_used ? '🤖' : '🔍'}</span>
                <span>
                  {result.llm_used
                    ? `AI Answer · ${result.model_used}`
                    : 'Retrieved Passages (Retrieval-only mode)'}
                </span>
                {result.answer_grounded && (
                  <span className="badge badge-ready" style={{ marginLeft: 'auto' }}>
                    ✓ Grounded
                  </span>
                )}
              </div>
              <div className="answer-box-content">
                {result.answer}
              </div>

              {/* Timing metadata */}
              <div className="timing-row" style={{ padding: '0 20px 16px' }}>
                <div className="timing-item">
                  🔍 Retrieval: <span>{result.retrieval_time_ms}ms</span>
                </div>
                {result.llm_used && (
                  <div className="timing-item">
                    🤖 Generation: <span>{result.generation_time_ms}ms</span>
                  </div>
                )}
                <div className="timing-item">
                  ⏱ Total: <span>{result.total_time_ms}ms</span>
                </div>
                <div className="timing-item">
                  📦 Sources: <span>{result.sources.length}</span>
                </div>
              </div>
            </div>

            {/* Sources */}
            {result.sources.length > 0 && (
              <div className="sources-section">
                <div className="sources-title">
                  📎 Source Evidence ({result.sources.length} passages retrieved)
                </div>
                {result.sources.map((source, i) => (
                  <div key={source.chunk_id} className="source-card">
                    <div
                      className="source-card-header"
                      onClick={() => toggleSource(i)}
                    >
                      <div className="source-card-info">
                        <span style={{ fontSize: '1rem' }}>
                          {source.filename?.endsWith('.pdf') ? '📄' : '📝'}
                        </span>
                        <span className="source-filename">{source.filename}</span>
                        {source.page_number && (
                          <span className="source-page">Page {source.page_number}</span>
                        )}
                        <span style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)' }}>
                          Chunk #{source.chunk_index + 1}
                        </span>
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                        {source.relevance_score !== null && (
                          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                            <div className="relevance-bar">
                              <div
                                className="relevance-bar-fill"
                                style={{
                                  width: `${Math.min(100, (source.relevance_score || 0) * 100)}%`,
                                  background: getScoreColor(source.relevance_score),
                                }}
                              />
                            </div>
                            <span className="source-score">
                              {((source.relevance_score || 0) * 100).toFixed(0)}%
                            </span>
                          </div>
                        )}
                        <span style={{ color: 'var(--text-tertiary)', fontSize: '0.8rem' }}>
                          {expandedSources[i] ? '▲' : '▼'}
                        </span>
                      </div>
                    </div>
                    {expandedSources[i] && (
                      <div className="source-passage">
                        "{source.passage}"
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}

            {result.sources.length === 0 && (
              <div className="empty-state">
                <div className="empty-state-icon">🔍</div>
                <h3>No relevant passages found</h3>
                <p>
                  The knowledge base didn't contain information closely matching your question.
                  Try uploading more relevant documents or rephrasing your question.
                </p>
              </div>
            )}

            {/* Ask another question */}
            <div style={{ marginTop: 20, textAlign: 'center' }}>
              <button
                className="btn btn-secondary"
                onClick={() => { setResult(null); setQuestion(''); inputRef.current?.focus(); }}
              >
                ✦ Ask Another Question
              </button>
            </div>
          </>
        )}
      </div>
    </>
  )
}

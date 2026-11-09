/**
 * API Service — All backend communication lives here.
 * 
 * WHY A CENTRALIZED SERVICE?
 * Instead of writing fetch() calls in every component, we centralize them.
 * This means: one place to change the base URL, add auth headers later,
 * handle errors consistently, and mock in tests.
 */

const BASE_URL = '/api/v1';

class ApiError extends Error {
  constructor(message, status, detail) {
    super(message);
    this.status = status;
    this.detail = detail;
  }
}

async function handleResponse(response) {
  if (!response.ok) {
    let detail = '';
    try {
      const errorBody = await response.json();
      detail = errorBody.detail || errorBody.error || JSON.stringify(errorBody);
    } catch {
      detail = response.statusText;
    }
    throw new ApiError(`HTTP ${response.status}: ${detail}`, response.status, detail);
  }
  return response.json();
}

// =============================================================================
// System / Dashboard
// =============================================================================

export async function fetchSystemStatus() {
  const res = await fetch(`${BASE_URL}/system/status`);
  return handleResponse(res);
}

export async function fetchDashboardStats() {
  const res = await fetch(`${BASE_URL}/system/dashboard`);
  return handleResponse(res);
}

// =============================================================================
// Documents
// =============================================================================

export async function fetchDocuments() {
  const res = await fetch(`${BASE_URL}/documents`);
  return handleResponse(res);
}

export async function fetchDocument(id) {
  const res = await fetch(`${BASE_URL}/documents/${id}`);
  return handleResponse(res);
}

export async function uploadDocument(file, onProgress) {
  const formData = new FormData();
  formData.append('file', file);

  // Use XMLHttpRequest for upload progress tracking
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    
    xhr.upload.addEventListener('progress', (event) => {
      if (event.lengthComputable && onProgress) {
        onProgress(Math.round((event.loaded / event.total) * 100));
      }
    });

    xhr.addEventListener('load', () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(JSON.parse(xhr.responseText));
      } else {
        let detail = xhr.statusText;
        try { detail = JSON.parse(xhr.responseText).detail || detail; } catch {}
        reject(new ApiError(`Upload failed: ${detail}`, xhr.status, detail));
      }
    });

    xhr.addEventListener('error', () => {
      reject(new ApiError('Network error during upload', 0, 'Network error'));
    });

    xhr.open('POST', `${BASE_URL}/documents/upload`);
    xhr.send(formData);
  });
}

export async function deleteDocument(id) {
  const res = await fetch(`${BASE_URL}/documents/${id}`, { method: 'DELETE' });
  return handleResponse(res);
}

export async function pollDocumentStatus(id, maxAttempts = 30, intervalMs = 2000) {
  /**
   * Poll a document's status until it's ready or errored.
   * Returns the final document object.
   */
  for (let i = 0; i < maxAttempts; i++) {
    const doc = await fetchDocument(id);
    if (doc.status === 'ready' || doc.status === 'error') {
      return doc;
    }
    await new Promise(r => setTimeout(r, intervalMs));
  }
  throw new ApiError('Document processing timed out', 408, 'Timeout');
}

// =============================================================================
// Query / Ask AI
// =============================================================================

export async function askQuestion(question, topK = null) {
  const body = { question };
  if (topK !== null) body.top_k = topK;

  const res = await fetch(`${BASE_URL}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return handleResponse(res);
}

export { ApiError };

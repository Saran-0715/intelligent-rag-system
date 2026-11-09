import { useState, useEffect, useCallback, useRef } from 'react'
import { fetchDocuments, uploadDocument, deleteDocument, pollDocumentStatus } from '../services/api.js'

/**
 * KnowledgeBase Page — Document upload, listing, and management.
 * All actions connect to the real backend.
 */
export default function KnowledgeBase({ onStatusRefresh }) {
  const [documents, setDocuments] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [uploadProgress, setUploadProgress] = useState(null)   // null | 0-100
  const [uploadStatus, setUploadStatus] = useState(null)       // null | {type, message}
  const [deletingId, setDeletingId] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const fileInputRef = useRef(null)

  const loadDocuments = useCallback(async () => {
    try {
      const data = await fetchDocuments()
      setDocuments(data.documents)
      setError(null)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    loadDocuments()
  }, [loadDocuments])

  const handleFileSelect = async (file) => {
    if (!file) return

    const allowed = ['.pdf', '.txt']
    const ext = '.' + file.name.split('.').pop().toLowerCase()
    if (!allowed.includes(ext)) {
      setUploadStatus({ type: 'error', message: `Unsupported file type: ${ext}. Please upload PDF or TXT files.` })
      return
    }

    setUploadProgress(0)
    setUploadStatus(null)

    try {
      // 1. Upload the file
      const uploadResult = await uploadDocument(file, (pct) => setUploadProgress(pct))
      setUploadProgress(100)
      setUploadStatus({ type: 'info', message: `✓ "${file.name}" uploaded. Processing (extracting text, chunking, embedding)...` })

      // 2. Refresh list immediately (shows processing status)
      await loadDocuments()
      onStatusRefresh?.()

      // 3. Poll until processing is done
      try {
        const finalDoc = await pollDocumentStatus(uploadResult.id)
        if (finalDoc.status === 'ready') {
          setUploadStatus({
            type: 'success',
            message: `✓ "${file.name}" is ready. ${finalDoc.chunk_count} chunks indexed.`,
          })
        } else {
          setUploadStatus({
            type: 'error',
            message: `❌ Processing failed: ${finalDoc.error_message}`,
          })
        }
      } catch (pollErr) {
        setUploadStatus({ type: 'warning', message: 'Processing is taking longer than expected. Refresh the page to check status.' })
      }

      await loadDocuments()
      onStatusRefresh?.()
    } catch (err) {
      setUploadStatus({ type: 'error', message: `Upload failed: ${err.detail || err.message}` })
    } finally {
      setUploadProgress(null)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  const handleDelete = async (doc) => {
    if (!confirm(`Delete "${doc.filename}" and all its indexed data? This cannot be undone.`)) return

    setDeletingId(doc.id)
    try {
      await deleteDocument(doc.id)
      await loadDocuments()
      onStatusRefresh?.()
    } catch (err) {
      setError(`Failed to delete: ${err.message}`)
    } finally {
      setDeletingId(null)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) handleFileSelect(file)
  }

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`
  }

  const formatDate = (iso) => {
    try {
      return new Date(iso).toLocaleDateString('en-IN', { month: 'short', day: 'numeric', year: 'numeric' })
    } catch { return iso }
  }

  return (
    <>
      <div className="page-header">
        <h1>📚 Knowledge Base</h1>
        <p>Upload and manage documents in your knowledge base</p>
      </div>

      <div className="page-content">
        {/* Upload Zone */}
        <div className="card" style={{ marginBottom: 20 }}>
          <div className="card-header">
            <div className="card-title">Upload Document</div>
            <div className="card-subtitle">PDF and TXT files · Max {50}MB</div>
          </div>

          <div
            className={`upload-zone ${dragOver ? 'drag-over' : ''}`}
            onClick={() => fileInputRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
          >
            <div className="upload-zone-icon">
              {uploadProgress !== null ? '⏳' : '📂'}
            </div>
            <div className="upload-zone-text">
              {uploadProgress !== null
                ? `Uploading... ${uploadProgress}%`
                : 'Click to browse or drag & drop'}
            </div>
            <div className="upload-zone-hint">Supported: .pdf, .txt</div>

            {uploadProgress !== null && (
              <div className="upload-progress">
                <div className="upload-progress-bar" style={{ width: `${uploadProgress}%` }} />
              </div>
            )}
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.txt"
            style={{ display: 'none' }}
            onChange={(e) => handleFileSelect(e.target.files[0])}
          />

          {uploadStatus && (
            <div className={`alert alert-${uploadStatus.type}`} style={{ marginTop: 12, marginBottom: 0 }}>
              {uploadStatus.message}
            </div>
          )}
        </div>

        {/* Documents List */}
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Documents</div>
              <div className="card-subtitle">{documents.length} document{documents.length !== 1 ? 's' : ''} in knowledge base</div>
            </div>
            <button className="btn btn-secondary btn-sm" onClick={loadDocuments} disabled={loading}>
              {loading ? '...' : '↻ Refresh'}
            </button>
          </div>

          {error && <div className="alert alert-error">{error}</div>}

          {loading ? (
            <div style={{ padding: '20px 0' }}>
              {[0,1,2].map(i => (
                <div key={i} style={{ display: 'flex', gap: 12, padding: '12px 0', borderBottom: '1px solid var(--border-subtle)' }}>
                  <div className="loading-skeleton" style={{ width: 36, height: 36, borderRadius: 'var(--radius-sm)' }} />
                  <div style={{ flex: 1 }}>
                    <div className="loading-skeleton" style={{ height: 16, width: '40%', marginBottom: 8 }} />
                    <div className="loading-skeleton" style={{ height: 12, width: '60%' }} />
                  </div>
                </div>
              ))}
            </div>
          ) : documents.length === 0 ? (
            <div className="empty-state">
              <div className="empty-state-icon">📂</div>
              <h3>No documents yet</h3>
              <p>Upload your first PDF or TXT file above to get started.</p>
            </div>
          ) : (
            <table className="doc-table">
              <thead>
                <tr>
                  <th>Document</th>
                  <th>Type</th>
                  <th>Size</th>
                  <th>Chunks</th>
                  <th>Status</th>
                  <th>Uploaded</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {documents.map(doc => (
                  <tr key={doc.id}>
                    <td>
                      <div className="doc-filename">
                        <span>{doc.file_type === 'pdf' ? '📄' : '📝'}</span>
                        <span style={{ maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                          {doc.filename}
                        </span>
                        {doc.page_count && (
                          <span style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)' }}>
                            {doc.page_count}p
                          </span>
                        )}
                      </div>
                    </td>
                    <td>
                      <span className={`badge badge-${doc.file_type}`}>
                        {doc.file_type?.toUpperCase()}
                      </span>
                    </td>
                    <td className="text-muted text-small">{formatFileSize(doc.file_size)}</td>
                    <td className="text-muted text-small text-mono">{doc.chunk_count}</td>
                    <td>
                      <span className={`badge badge-${doc.status}`}>
                        {doc.status === 'processing' && <span className="loading-spinner" style={{ width: 10, height: 10, marginRight: 4 }} />}
                        {doc.status}
                      </span>
                    </td>
                    <td className="text-muted text-small">{formatDate(doc.created_at)}</td>
                    <td>
                      <button
                        className="btn btn-danger btn-sm"
                        onClick={() => handleDelete(doc)}
                        disabled={deletingId === doc.id}
                        title="Delete document and all indexed data"
                      >
                        {deletingId === doc.id ? '...' : '🗑'}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </>
  )
}

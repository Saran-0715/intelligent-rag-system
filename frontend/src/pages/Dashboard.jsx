import { useState, useEffect, useCallback } from 'react'
import { fetchDashboardStats } from '../services/api.js'

/**
 * Dashboard Page — Shows real statistics from the backend.
 * No hardcoded numbers — everything comes from actual API calls.
 */
export default function Dashboard({ systemStatus, onNavigate }) {
  const [stats, setStats] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  const loadStats = useCallback(async () => {
    try {
      setError(null)
      const data = await fetchDashboardStats()
      setStats(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    loadStats()
  }, [loadStats])

  const formatDate = (iso) => {
    try {
      return new Date(iso).toLocaleString('en-IN', {
        month: 'short', day: 'numeric',
        hour: '2-digit', minute: '2-digit'
      })
    } catch { return iso }
  }

  return (
    <>
      <div className="page-header">
        <h1>⬡ Dashboard</h1>
        <p>System overview and recent activity</p>
      </div>

      <div className="page-content">
        {/* System warnings */}
        {systemStatus?.warnings?.map((w, i) => (
          <div key={i} className="alert alert-warning">
            ⚠️ {w}
          </div>
        ))}

        {/* Stats cards */}
        {loading ? (
          <div className="stats-grid">
            {[0,1,2,3].map(i => (
              <div key={i} className="stat-card">
                <div className="loading-skeleton" style={{ height: 40, width: '60%', marginBottom: 8 }} />
                <div className="loading-skeleton" style={{ height: 16, width: '80%' }} />
              </div>
            ))}
          </div>
        ) : error ? (
          <div className="alert alert-error">❌ Failed to load stats: {error}</div>
        ) : (
          <div className="stats-grid">
            <div className="stat-card">
              <span className="stat-icon">📄</span>
              <div className="stat-value">{stats?.total_documents ?? 0}</div>
              <div className="stat-label">Documents</div>
            </div>
            <div className="stat-card">
              <span className="stat-icon">🧩</span>
              <div className="stat-value">{stats?.total_chunks ?? 0}</div>
              <div className="stat-label">Indexed Chunks</div>
            </div>
            <div className="stat-card">
              <span className="stat-icon">💬</span>
              <div className="stat-value">{stats?.total_queries ?? 0}</div>
              <div className="stat-label">Queries Asked</div>
            </div>
            <div className="stat-card">
              <span className="stat-icon">🤖</span>
              <div className="stat-value" style={{ fontSize: '1.1rem' }}>
                {systemStatus?.llm_enabled ? 'ON' : 'OFF'}
              </div>
              <div className="stat-label">AI Generation</div>
            </div>
          </div>
        )}

        {/* Quick actions */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 24 }}>
          <div className="card" style={{ cursor: 'pointer' }} onClick={() => onNavigate('knowledge-base')}>
            <div style={{ fontSize: '1.5rem', marginBottom: 8 }}>📚</div>
            <div className="card-title">Upload Documents</div>
            <div className="card-subtitle">Add PDFs or text files to your knowledge base</div>
            <div style={{ marginTop: 12 }}>
              <span className="btn btn-primary btn-sm">Go to Knowledge Base →</span>
            </div>
          </div>
          <div className="card" style={{ cursor: 'pointer' }} onClick={() => onNavigate('ask-ai')}>
            <div style={{ fontSize: '1.5rem', marginBottom: 8 }}>✦</div>
            <div className="card-title">Ask a Question</div>
            <div className="card-subtitle">Search your knowledge base with natural language</div>
            <div style={{ marginTop: 12 }}>
              <span className="btn btn-primary btn-sm">Go to Ask AI →</span>
            </div>
          </div>
        </div>

        {/* Recent queries */}
        <div className="card">
          <div className="card-header">
            <div>
              <div className="card-title">Recent Queries</div>
              <div className="card-subtitle">Last questions asked through the system</div>
            </div>
          </div>

          {!stats || stats.recent_queries.length === 0 ? (
            <div className="empty-state" style={{ padding: '32px 24px' }}>
              <div className="empty-state-icon">💬</div>
              <h3>No queries yet</h3>
              <p>Go to <strong>Ask AI</strong> to start querying your knowledge base.</p>
            </div>
          ) : (
            <div>
              {stats.recent_queries.map((q, i) => (
                <div
                  key={q.id}
                  style={{
                    padding: '12px 0',
                    borderBottom: i < stats.recent_queries.length - 1 ? '1px solid var(--border-subtle)' : 'none',
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: 12,
                  }}
                >
                  <span style={{ fontSize: '0.9rem', marginTop: 2 }}>{q.llm_used ? '🤖' : '🔍'}</span>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ fontSize: '0.875rem', color: 'var(--text-primary)', marginBottom: 2, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                      {q.question}
                    </div>
                    <div style={{ fontSize: '0.72rem', color: 'var(--text-tertiary)' }}>
                      {formatDate(q.created_at)} · {q.llm_used ? 'AI answer' : 'Retrieval only'}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* RAG Pipeline explanation */}
        <div className="card" style={{ marginTop: 16 }}>
          <div className="card-header">
            <div className="card-title">How the RAG Pipeline Works</div>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 12 }}>
            {[
              { step: '1', icon: '📄', title: 'Upload', desc: 'PDF/TXT uploaded and stored' },
              { step: '2', icon: '✂️', title: 'Chunk', desc: 'Text split into 512-char pieces' },
              { step: '3', icon: '🔢', title: 'Embed', desc: 'Each chunk → 384-dim vector' },
              { step: '4', icon: '💾', title: 'Index', desc: 'Vectors stored in Chroma DB' },
              { step: '5', icon: '🔍', title: 'Retrieve', desc: 'Top-5 similar chunks found' },
              { step: '6', icon: '🤖', title: 'Generate', desc: 'LLM answers from context' },
            ].map(item => (
              <div
                key={item.step}
                style={{
                  padding: '14px',
                  background: 'var(--bg-tertiary)',
                  borderRadius: 'var(--radius-sm)',
                  border: '1px solid var(--border-subtle)',
                  textAlign: 'center',
                }}
              >
                <div style={{ fontSize: '1.4rem', marginBottom: 6 }}>{item.icon}</div>
                <div style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-primary)', marginBottom: 4 }}>
                  {item.step}. {item.title}
                </div>
                <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)' }}>{item.desc}</div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </>
  )
}

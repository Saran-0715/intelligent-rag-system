/**
 * Sidebar — Navigation panel with system status indicator.
 */
export default function Sidebar({ currentPage, onNavigate, systemStatus, statusLoading }) {
  const navItems = [
    { id: 'dashboard',      label: 'Dashboard',       icon: '⬡',  available: true },
    { id: 'knowledge-base', label: 'Knowledge Base',  icon: '📚', available: true },
    { id: 'ask-ai',         label: 'Ask AI',          icon: '✦',  available: true },
    { id: 'evaluation',     label: 'Evaluation',      icon: '📊', available: false },
    { id: 'monitoring',     label: 'Monitoring',      icon: '📈', available: false },
    { id: 'settings',       label: 'Settings',        icon: '⚙', available: false },
  ]

  const getStatusDot = () => {
    if (statusLoading) return 'warning'
    if (!systemStatus) return 'error'
    return systemStatus.warnings?.length > 0 ? 'warning' : 'ok'
  }

  const getStatusText = () => {
    if (statusLoading) return 'Connecting...'
    if (!systemStatus) return 'Backend offline'
    if (!systemStatus.llm_enabled) return 'Retrieval mode'
    return 'All systems operational'
  }

  return (
    <aside className="sidebar">
      {/* Logo / Branding */}
      <div className="sidebar-logo">
        <div className="sidebar-logo-title">◈ RAG System</div>
        <div className="sidebar-logo-subtitle">Phase 1 MVP · v0.1.0</div>
      </div>

      {/* Navigation */}
      <nav className="sidebar-nav">
        <div className="nav-section-label">Navigation</div>
        {navItems.map(item => (
          <button
            key={item.id}
            className={`nav-item ${currentPage === item.id ? 'active' : ''}`}
            onClick={() => onNavigate(item.id)}
            title={!item.available ? 'Coming in a future phase' : undefined}
          >
            <span className="nav-icon">{item.icon}</span>
            {item.label}
            {!item.available && <span className="nav-item-soon">Phase 2+</span>}
          </button>
        ))}
      </nav>

      {/* System status */}
      <div className="sidebar-footer">
        <div className="system-status-badge">
          <span className={`status-dot ${getStatusDot()}`} />
          <span style={{ fontSize: '0.72rem' }}>{getStatusText()}</span>
        </div>
        {systemStatus && (
          <div style={{ marginTop: 8, padding: '0 4px' }}>
            <div style={{ fontSize: '0.68rem', color: 'var(--text-tertiary)', lineHeight: 1.8 }}>
              <div>📄 {systemStatus.total_documents} docs · {systemStatus.total_chunks} chunks</div>
              <div>🤖 {systemStatus.llm_enabled ? systemStatus.llm_model : 'No LLM'}</div>
            </div>
          </div>
        )}
      </div>
    </aside>
  )
}

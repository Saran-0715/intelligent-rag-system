import { useState, useEffect, useCallback } from 'react'
import Sidebar from './components/Sidebar.jsx'
import Dashboard from './pages/Dashboard.jsx'
import KnowledgeBase from './pages/KnowledgeBase.jsx'
import AskAI from './pages/AskAI.jsx'
import { fetchSystemStatus } from './services/api.js'

/**
 * App — Root component, manages page routing and system status.
 * 
 * We use simple state-based "routing" instead of React Router for Phase 1.
 * This avoids a dependency and is easy to understand.
 * Phase 5 will add proper routing.
 */
export default function App() {
  const [currentPage, setCurrentPage] = useState('dashboard')
  const [systemStatus, setSystemStatus] = useState(null)
  const [statusLoading, setStatusLoading] = useState(true)

  // Load system status on mount (and periodically refresh)
  const loadStatus = useCallback(async () => {
    try {
      const status = await fetchSystemStatus()
      setSystemStatus(status)
    } catch (err) {
      console.error('Failed to load system status:', err)
      setSystemStatus(null)
    } finally {
      setStatusLoading(false)
    }
  }, [])

  useEffect(() => {
    loadStatus()
    // Refresh every 30 seconds
    const interval = setInterval(loadStatus, 30000)
    return () => clearInterval(interval)
  }, [loadStatus])

  const renderPage = () => {
    switch (currentPage) {
      case 'dashboard':
        return <Dashboard systemStatus={systemStatus} onNavigate={setCurrentPage} />
      case 'knowledge-base':
        return <KnowledgeBase onStatusRefresh={loadStatus} />
      case 'ask-ai':
        return <AskAI systemStatus={systemStatus} />
      default:
        return (
          <div className="page-content">
            <div className="empty-state">
              <div className="empty-state-icon">🚧</div>
              <h3>Coming in a Future Phase</h3>
              <p>This feature is planned for a later development phase.</p>
            </div>
          </div>
        )
    }
  }

  return (
    <div className="app-layout">
      <Sidebar
        currentPage={currentPage}
        onNavigate={setCurrentPage}
        systemStatus={systemStatus}
        statusLoading={statusLoading}
      />
      <div className="main-content">
        {renderPage()}
      </div>
    </div>
  )
}

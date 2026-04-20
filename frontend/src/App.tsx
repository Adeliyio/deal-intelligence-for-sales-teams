import { useState } from 'react'
import PipelineView from './pages/PipelineView'
import DealDetail from './pages/DealDetail'
import EvalDashboard from './pages/EvalDashboard'

type View = 'pipeline' | 'deal-detail' | 'eval'

function App() {
  const [currentView, setCurrentView] = useState<View>('pipeline')
  const [selectedDealId, setSelectedDealId] = useState<string | null>(null)

  const handleDealSelect = (dealId: string) => {
    setSelectedDealId(dealId)
    setCurrentView('deal-detail')
  }

  const handleBack = () => {
    setCurrentView('pipeline')
    setSelectedDealId(null)
  }

  return (
    <div className="min-h-screen bg-slate-50">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 px-6 py-4">
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-8">
            <div>
              <h1 className="text-xl font-bold text-slate-900">
                Tommy's Deal Intelligence
              </h1>
              <p className="text-sm text-slate-500">Revenue Intelligence Copilot</p>
            </div>

            {/* Navigation */}
            <nav className="flex gap-1">
              <button
                onClick={() => { setCurrentView('pipeline'); setSelectedDealId(null) }}
                className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                  currentView === 'pipeline' || currentView === 'deal-detail'
                    ? 'bg-primary-50 text-primary-700'
                    : 'text-slate-600 hover:bg-slate-100'
                }`}
              >
                Pipeline
              </button>
              <button
                onClick={() => setCurrentView('eval')}
                className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                  currentView === 'eval'
                    ? 'bg-primary-50 text-primary-700'
                    : 'text-slate-600 hover:bg-slate-100'
                }`}
              >
                Evaluation
              </button>
            </nav>
          </div>

          <div className="flex items-center gap-4">
            <span className="text-xs bg-green-100 text-green-700 px-2 py-1 rounded-full">
              System Healthy
            </span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-6">
        {currentView === 'pipeline' && (
          <PipelineView onDealSelect={handleDealSelect} />
        )}
        {currentView === 'deal-detail' && selectedDealId && (
          <DealDetail dealId={selectedDealId} onBack={handleBack} />
        )}
        {currentView === 'eval' && <EvalDashboard />}
      </main>
    </div>
  )
}

export default App

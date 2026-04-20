import { useState } from 'react'
import PipelineView from './pages/PipelineView'
import DealDetail from './pages/DealDetail'

type View = 'pipeline' | 'deal-detail'

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
          <div>
            <h1 className="text-xl font-bold text-slate-900">
              Deal Intelligence
            </h1>
            <p className="text-sm text-slate-500">Revenue Intelligence Copilot</p>
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
      </main>
    </div>
  )
}

export default App

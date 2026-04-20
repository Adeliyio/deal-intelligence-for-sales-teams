interface AgentDebateViewProps {
  leadAssessment?: string
  riskAnalysis?: string
  strategy?: string
  communication?: string
  criticReview?: string
}

interface AgentCardProps {
  name: string
  role: string
  content: string
  color: string
}

function AgentCard({ name, role, content, color }: AgentCardProps) {
  return (
    <div className={`border-l-4 ${color} bg-white rounded-r-lg p-4 shadow-sm`}>
      <div className="flex items-center gap-2 mb-2">
        <span className="font-semibold text-slate-900">{name}</span>
        <span className="text-xs text-slate-400">{role}</span>
      </div>
      <p className="text-sm text-slate-700 whitespace-pre-wrap">{content}</p>
    </div>
  )
}

export default function AgentDebateView({
  leadAssessment,
  riskAnalysis,
  strategy,
  communication,
  criticReview,
}: AgentDebateViewProps) {
  return (
    <div className="space-y-4">
      <h3 className="font-bold text-slate-900 text-lg">Agent Debate</h3>
      <p className="text-sm text-slate-500 mb-4">
        Multi-agent analysis showing how the system reasons about this deal.
        Agents collaborate and challenge each other before surfacing recommendations.
      </p>

      <div className="space-y-3">
        {leadAssessment && (
          <AgentCard
            name="Lead Intelligence"
            role="Deal Scorer"
            content={leadAssessment}
            color="border-blue-500"
          />
        )}

        {riskAnalysis && (
          <AgentCard
            name="Risk Analysis"
            role="Failure Pattern Detection"
            content={riskAnalysis}
            color="border-red-500"
          />
        )}

        {strategy && (
          <AgentCard
            name="Strategy"
            role="Action Recommender"
            content={strategy}
            color="border-emerald-500"
          />
        )}

        {communication && (
          <AgentCard
            name="Communication"
            role="Outreach Generator"
            content={communication}
            color="border-purple-500"
          />
        )}

        {criticReview && (
          <AgentCard
            name="Critic"
            role="Quality Control"
            content={criticReview}
            color="border-amber-500"
          />
        )}

        {!leadAssessment && !riskAnalysis && !strategy && (
          <div className="text-center py-8 text-slate-400">
            <p>Agent analysis requires OpenAI API key.</p>
            <p className="text-sm mt-1">
              ML predictions and rule-based strategy are shown above.
            </p>
          </div>
        )}
      </div>
    </div>
  )
}

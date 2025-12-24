import { TrendingUp, BarChart3, Building2, Target } from 'lucide-react';

export default function IntelligenceView() {
  return (
    <div className="h-full flex flex-col bg-slate-900 text-white p-6">
      <header className="mb-6">
        <h1 className="text-2xl font-bold flex items-center gap-3">
          <TrendingUp className="w-7 h-7 text-violet-400" />
          Executive Intelligence
        </h1>
        <p className="text-slate-400 mt-1">
          Efficiency analysis, margin profiling, vendor comparison, and initiative tracking
        </p>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <StubCard
          icon={BarChart3}
          title="Efficiency Analysis"
          description="GPU utilization and efficiency metrics"
          path="/intelligence"
        />
        <StubCard
          icon={TrendingUp}
          title="Margin Analysis"
          description="Profit margin simulation and projections"
          path="/intelligence/margin"
        />
        <StubCard
          icon={Building2}
          title="Vendor Comparison"
          description="Compare cloud providers and on-prem costs"
          path="/intelligence/vendor"
        />
        <StubCard
          icon={Target}
          title="Initiative Tracker"
          description="Track FinOps and optimization initiatives"
          path="/intelligence/initiatives"
        />
      </div>

      <div className="mt-6 p-4 bg-slate-800 rounded-lg border border-slate-700">
        <h3 className="text-sm font-semibold text-slate-400 mb-2">Integration Status</h3>
        <p className="text-sm text-slate-500">
          This view will integrate with horizon-services: efficiency-intelligence, margin-intelligence, vendor-intelligence, executive-intelligence, and initiative-tracker.
        </p>
      </div>
    </div>
  );
}

function StubCard({ icon: Icon, title, description, path }: {
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  description: string;
  path: string;
}) {
  return (
    <div className="p-4 bg-slate-800 rounded-lg border border-slate-700 hover:border-violet-500/50 transition-colors">
      <div className="flex items-center gap-3 mb-2">
        <Icon className="w-5 h-5 text-violet-400" />
        <h3 className="font-semibold">{title}</h3>
      </div>
      <p className="text-sm text-slate-400">{description}</p>
      <p className="text-xs text-slate-600 mt-2 font-mono">{path}</p>
    </div>
  );
}

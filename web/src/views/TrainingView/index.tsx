import { Brain, Network, Database, BarChart3, Play } from 'lucide-react';

export default function TrainingView() {
  return (
    <div className="h-full flex flex-col bg-slate-900 text-white p-6">
      <header className="mb-6">
        <h1 className="text-2xl font-bold flex items-center gap-3">
          <Brain className="w-7 h-7 text-pink-400" />
          ML Training (RustyTorch)
        </h1>
        <p className="text-slate-400 mt-1">
          Training jobs, distributed config, model registry, and metrics
        </p>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <StubCard
          icon={Play}
          title="Training Jobs"
          description="View and manage training jobs"
          path="/training"
        />
        <StubCard
          icon={Network}
          title="Distributed Config"
          description="RMPI + Nebula distributed training setup"
          path="/training/distributed"
        />
        <StubCard
          icon={Database}
          title="Model Registry"
          description="Browse and manage trained models"
          path="/training/models"
        />
        <StubCard
          icon={BarChart3}
          title="Metrics & Loss"
          description="Real-time training metrics charts"
          path="/training/metrics"
        />
      </div>

      <div className="mt-6 p-4 bg-slate-800 rounded-lg border border-slate-700">
        <h3 className="text-sm font-semibold text-slate-400 mb-2">Integration Status</h3>
        <p className="text-sm text-slate-500">
          This view will integrate with 07-rustytorch and 02-RMPI for distributed ML training.
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
    <div className="p-4 bg-slate-800 rounded-lg border border-slate-700 hover:border-pink-500/50 transition-colors">
      <div className="flex items-center gap-3 mb-2">
        <Icon className="w-5 h-5 text-pink-400" />
        <h3 className="font-semibold">{title}</h3>
      </div>
      <p className="text-sm text-slate-400">{description}</p>
      <p className="text-xs text-slate-600 mt-2 font-mono">{path}</p>
    </div>
  );
}

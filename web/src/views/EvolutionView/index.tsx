import { Dna, Brain, Zap, Sparkles, GitBranch } from 'lucide-react';

export default function EvolutionView() {
  return (
    <div className="h-full flex flex-col bg-slate-900 text-white p-6">
      <header className="mb-6">
        <h1 className="text-2xl font-bold flex items-center gap-3">
          <Dna className="w-7 h-7 text-purple-400" />
          Evolution Engines
        </h1>
        <p className="text-slate-400 mt-1">
          ADAS, DGM, and SwarmAgentic self-improving systems
        </p>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <StubCard
          icon={Sparkles}
          title="ADAS Engine"
          description="Automated Design Architecture Search"
          path="/evolution"
        />
        <StubCard
          icon={Brain}
          title="DGM Engine"
          description="Dynamic Genetic Mutation engine"
          path="/evolution/dgm"
        />
        <StubCard
          icon={Zap}
          title="SwarmAgentic"
          description="Swarm-based agentic optimization"
          path="/evolution/swarm"
        />
        <StubCard
          icon={GitBranch}
          title="Design Explorer"
          description="Explore evolved design space"
          path="/evolution/explorer"
        />
      </div>

      <div className="mt-6 p-4 bg-slate-800 rounded-lg border border-slate-700">
        <h3 className="text-sm font-semibold text-slate-400 mb-2">Integration Status</h3>
        <p className="text-sm text-slate-500">
          This view will integrate with 05-stratoswarm evolution engines: ADAS, DGM, and SwarmAgentic.
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
    <div className="p-4 bg-slate-800 rounded-lg border border-slate-700 hover:border-purple-500/50 transition-colors">
      <div className="flex items-center gap-3 mb-2">
        <Icon className="w-5 h-5 text-purple-400" />
        <h3 className="font-semibold">{title}</h3>
      </div>
      <p className="text-sm text-slate-400">{description}</p>
      <p className="text-xs text-slate-600 mt-2 font-mono">{path}</p>
    </div>
  );
}

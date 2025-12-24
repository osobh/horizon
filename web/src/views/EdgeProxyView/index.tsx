import { Shield, Route, Brain, Zap, AlertTriangle } from 'lucide-react';

export default function EdgeProxyView() {
  return (
    <div className="h-full flex flex-col bg-slate-900 text-white p-6">
      <header className="mb-6">
        <h1 className="text-2xl font-bold flex items-center gap-3">
          <Shield className="w-7 h-7 text-red-400" />
          Edge Proxy (Vortex)
        </h1>
        <p className="text-slate-400 mt-1">
          Intelligent proxy with SLAI brain, protocol transmutation, and DDoS protection
        </p>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <StubCard
          icon={Shield}
          title="Vortex Status"
          description="Proxy health and connection metrics"
          path="/edge"
        />
        <StubCard
          icon={Route}
          title="Routing Rules"
          description="Configure request routing rules"
          path="/edge/routing"
        />
        <StubCard
          icon={Brain}
          title="SLAI Brain Link"
          description="AI-driven routing decisions"
          path="/edge/brain"
        />
        <StubCard
          icon={Zap}
          title="Protocol Transmutation"
          description="Protocol conversion statistics"
          path="/edge/transmutation"
        />
        <StubCard
          icon={AlertTriangle}
          title="DDoS Protection"
          description="Proof-of-work challenge status"
          path="/edge/ddos"
        />
      </div>

      <div className="mt-6 p-4 bg-slate-800 rounded-lg border border-slate-700">
        <h3 className="text-sm font-semibold text-slate-400 mb-2">Integration Status</h3>
        <p className="text-sm text-slate-500">
          This view will integrate with 09-vortex edge proxy services.
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
    <div className="p-4 bg-slate-800 rounded-lg border border-slate-700 hover:border-red-500/50 transition-colors">
      <div className="flex items-center gap-3 mb-2">
        <Icon className="w-5 h-5 text-red-400" />
        <h3 className="font-semibold">{title}</h3>
      </div>
      <p className="text-sm text-slate-400">{description}</p>
      <p className="text-xs text-slate-600 mt-2 font-mono">{path}</p>
    </div>
  );
}

import { Network, Server, Shield, Wifi, Activity } from 'lucide-react';

export default function NetworkView() {
  return (
    <div className="h-full flex flex-col bg-slate-900 text-white p-6">
      <header className="mb-6">
        <h1 className="text-2xl font-bold flex items-center gap-3">
          <Network className="w-7 h-7 text-cyan-400" />
          Network (Nebula)
        </h1>
        <p className="text-slate-400 mt-1">
          RDMA topology, mesh relay, and zero-knowledge proofs
        </p>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <StubCard
          icon={Network}
          title="RDMA Topology"
          description="400Gbps GPU-direct network visualization"
          path="/network"
        />
        <StubCard
          icon={Server}
          title="Mesh Relay"
          description="Stratoswarm mesh relay status"
          path="/network/mesh"
        />
        <StubCard
          icon={Shield}
          title="ZK Proofs"
          description="Zero-knowledge proof generation/verification"
          path="/network/zk"
        />
        <StubCard
          icon={Wifi}
          title="NAT Traversal"
          description="Connection status and hole punching"
          path="/network/nat"
        />
        <StubCard
          icon={Activity}
          title="eBPF Stats"
          description="Packet classification statistics"
          path="/network/ebpf"
        />
      </div>

      <div className="mt-6 p-4 bg-slate-800 rounded-lg border border-slate-700">
        <h3 className="text-sm font-semibold text-slate-400 mb-2">Integration Status</h3>
        <p className="text-sm text-slate-500">
          This view will integrate with 10-nebula services for RDMA, ZK proofs, and mesh networking.
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
    <div className="p-4 bg-slate-800 rounded-lg border border-slate-700 hover:border-cyan-500/50 transition-colors">
      <div className="flex items-center gap-3 mb-2">
        <Icon className="w-5 h-5 text-cyan-400" />
        <h3 className="font-semibold">{title}</h3>
      </div>
      <p className="text-sm text-slate-400">{description}</p>
      <p className="text-xs text-slate-600 mt-2 font-mono">{path}</p>
    </div>
  );
}

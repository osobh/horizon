import { Calendar, Clock, Users, Shield, Network, TrendingUp } from 'lucide-react';

export default function SchedulerView() {
  return (
    <div className="h-full flex flex-col bg-slate-900 text-white p-6">
      <header className="mb-6">
        <h1 className="text-2xl font-bold flex items-center gap-3">
          <Calendar className="w-7 h-7 text-blue-400" />
          GPU Scheduler (SLAI)
        </h1>
        <p className="text-slate-400 mt-1">
          Job queue management, quotas, and cluster scheduling
        </p>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <StubCard
          icon={Clock}
          title="Job Queue"
          description="View and manage pending, running, and completed jobs"
          path="/scheduler"
        />
        <StubCard
          icon={Users}
          title="Tenants & Quotas"
          description="Configure tenant resource limits and GPU quotas"
          path="/scheduler/tenants"
        />
        <StubCard
          icon={TrendingUp}
          title="Backfill Status"
          description="ML-predicted backfill scheduling status"
          path="/scheduler/backfill"
        />
        <StubCard
          icon={Calendar}
          title="Reservations"
          description="Time-based GPU reservations"
          path="/scheduler/reservations"
        />
        <StubCard
          icon={Shield}
          title="QoS Policies"
          description="Priority and quality of service policies"
          path="/scheduler/qos"
        />
        <StubCard
          icon={Network}
          title="Federation"
          description="Multi-cluster federation status"
          path="/scheduler/federation"
        />
      </div>

      <div className="mt-6 p-4 bg-slate-800 rounded-lg border border-slate-700">
        <h3 className="text-sm font-semibold text-slate-400 mb-2">Integration Status</h3>
        <p className="text-sm text-slate-500">
          This view will integrate with 03-SLAI services. Features will be migrated from 03-SLAI/dashboard/.
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
    <div className="p-4 bg-slate-800 rounded-lg border border-slate-700 hover:border-blue-500/50 transition-colors">
      <div className="flex items-center gap-3 mb-2">
        <Icon className="w-5 h-5 text-blue-400" />
        <h3 className="font-semibold">{title}</h3>
      </div>
      <p className="text-sm text-slate-400">{description}</p>
      <p className="text-xs text-slate-600 mt-2 font-mono">{path}</p>
    </div>
  );
}

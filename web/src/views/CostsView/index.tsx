import { DollarSign, TrendingUp, FileText, Bell, PieChart } from 'lucide-react';

export default function CostsView() {
  return (
    <div className="h-full flex flex-col bg-slate-900 text-white p-6">
      <header className="mb-6">
        <h1 className="text-2xl font-bold flex items-center gap-3">
          <DollarSign className="w-7 h-7 text-emerald-400" />
          Cost Intelligence
        </h1>
        <p className="text-slate-400 mt-1">
          Cost attribution, forecasting, chargeback, and showback reports
        </p>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <StubCard
          icon={PieChart}
          title="Attribution Dashboard"
          description="Cost breakdown by team, project, and resource"
          path="/costs"
        />
        <StubCard
          icon={TrendingUp}
          title="Capacity Forecasting"
          description="13-week resource and cost forecasts"
          path="/costs/forecast"
        />
        <StubCard
          icon={FileText}
          title="Chargeback Reports"
          description="Generate chargeback reports for finance"
          path="/costs/chargeback"
        />
        <StubCard
          icon={FileText}
          title="Showback Reports"
          description="Generate showback reports for teams"
          path="/costs/showback"
        />
        <StubCard
          icon={Bell}
          title="Budget Alerts"
          description="Configure budget thresholds and alerts"
          path="/costs/alerts"
        />
      </div>

      <div className="mt-6 p-4 bg-slate-800 rounded-lg border border-slate-700">
        <h3 className="text-sm font-semibold text-slate-400 mb-2">Integration Status</h3>
        <p className="text-sm text-slate-500">
          This view will integrate with horizon-services: cost-ingestor, cost-attributor, cost-reporter, and capacity-modeler.
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
    <div className="p-4 bg-slate-800 rounded-lg border border-slate-700 hover:border-emerald-500/50 transition-colors">
      <div className="flex items-center gap-3 mb-2">
        <Icon className="w-5 h-5 text-emerald-400" />
        <h3 className="font-semibold">{title}</h3>
      </div>
      <p className="text-sm text-slate-400">{description}</p>
      <p className="text-xs text-slate-600 mt-2 font-mono">{path}</p>
    </div>
  );
}

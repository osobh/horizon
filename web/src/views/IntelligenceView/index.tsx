import { useEffect, useState } from 'react';
import {
  TrendingUp,
  BarChart3,
  Building2,
  Target,
  RefreshCw,
  AlertTriangle,
  CheckCircle,
  DollarSign,
  Cpu,
  XCircle,
  Power,
} from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, PieChart, Pie, Cell } from 'recharts';
import { useIntelligenceStore, IdleResource, ProfitMargin, VendorUtilization, ExecutiveKPI, IntelligenceAlert } from '../../stores/intelligenceStore';

export default function IntelligenceView() {
  const {
    summary,
    idleResources,
    profitMargins,
    vendorUtilizations,
    executiveKpis,
    alerts,
    loading,
    error,
    fetchSummary,
    fetchIdleResources,
    fetchProfitMargins,
    fetchVendorUtilizations,
    fetchExecutiveKpis,
    fetchAlerts,
    acknowledgeAlert,
    terminateIdleResource,
  } = useIntelligenceStore();

  const [selectedTab, setSelectedTab] = useState<'overview' | 'efficiency' | 'margins' | 'vendors'>('overview');

  useEffect(() => {
    fetchSummary();
    fetchIdleResources();
    fetchProfitMargins();
    fetchVendorUtilizations();
    fetchExecutiveKpis();
    fetchAlerts();
    const interval = setInterval(() => {
      fetchSummary();
      fetchIdleResources();
    }, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="h-full flex flex-col bg-slate-900 text-white">
      {/* Header */}
      <div className="h-16 bg-slate-800 border-b border-slate-700 flex items-center px-6">
        <TrendingUp className="w-6 h-6 text-violet-400 mr-3" />
        <h1 className="text-xl font-semibold">Executive Intelligence</h1>
        <div className="flex-1" />
        <button onClick={() => { fetchSummary(); fetchIdleResources(); }} className="p-2 hover:bg-slate-700 rounded-lg transition-colors">
          <RefreshCw className={`w-5 h-5 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Summary Bar */}
      {summary && (
        <div className="bg-slate-800/50 border-b border-slate-700 px-6 py-3 flex gap-6 text-sm">
          <div>
            <span className="text-slate-400">Efficiency Score:</span>{' '}
            <span className="text-violet-400 font-semibold">{(summary.overall_efficiency * 100).toFixed(1)}%</span>
          </div>
          <div>
            <span className="text-slate-400">Idle Resources:</span>{' '}
            <span className={summary.idle_resource_count > 0 ? 'text-amber-400' : 'text-green-400'}>
              {summary.idle_resource_count}
            </span>
          </div>
          <div>
            <span className="text-slate-400">Potential Savings:</span>{' '}
            <span className="text-emerald-400">${summary.potential_savings_usd.toLocaleString()}/mo</span>
          </div>
          <div>
            <span className="text-slate-400">Avg Margin:</span>{' '}
            <span className="text-blue-400">{(summary.average_margin * 100).toFixed(1)}%</span>
          </div>
        </div>
      )}

      {/* Tabs */}
      <div className="border-b border-slate-700 px-6">
        <div className="flex gap-4">
          {['overview', 'efficiency', 'margins', 'vendors'].map((tab) => (
            <button
              key={tab}
              onClick={() => setSelectedTab(tab as any)}
              className={`py-3 px-4 border-b-2 transition-colors capitalize ${
                selectedTab === tab ? 'border-violet-400 text-violet-400' : 'border-transparent text-slate-400 hover:text-white'
              }`}
            >
              {tab}
              {tab === 'efficiency' && idleResources.length > 0 && (
                <span className="ml-2 bg-amber-500 text-white text-xs px-2 py-0.5 rounded-full">
                  {idleResources.length}
                </span>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto p-6">
        {error && <div className="mb-4 p-4 bg-red-500/20 border border-red-500 rounded-lg text-red-400">{error}</div>}

        {selectedTab === 'overview' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Executive KPIs */}
            <div className="lg:col-span-2 grid grid-cols-2 md:grid-cols-4 gap-4">
              {executiveKpis.map((kpi) => (
                <KPICard key={kpi.id} kpi={kpi} />
              ))}
            </div>

            {/* Margin Trend */}
            <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
              <h3 className="font-medium mb-4">Profit Margin Trend</h3>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={generateMarginTrendData()}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="month" stroke="#94a3b8" fontSize={12} />
                  <YAxis stroke="#94a3b8" fontSize={12} tickFormatter={(v) => `${v}%`} />
                  <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none' }} formatter={(v: number) => [`${v.toFixed(1)}%`, 'Margin']} />
                  <Line type="monotone" dataKey="margin" stroke="#8b5cf6" strokeWidth={2} dot={{ fill: '#8b5cf6' }} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Vendor Comparison */}
            <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
              <h3 className="font-medium mb-4">Vendor Utilization</h3>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={vendorUtilizations}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="vendor_name" stroke="#94a3b8" fontSize={12} />
                  <YAxis stroke="#94a3b8" fontSize={12} tickFormatter={(v) => `${v}%`} />
                  <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none' }} />
                  <Bar dataKey="utilization_pct" fill="#8b5cf6" radius={[4, 4, 0, 0]} name="Utilization %" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Alerts */}
            <div className="lg:col-span-2 bg-slate-800 rounded-lg border border-slate-700 p-4">
              <h3 className="font-medium mb-4 flex items-center gap-2">
                <AlertTriangle className="w-5 h-5 text-amber-400" />
                Intelligence Alerts
              </h3>
              <div className="space-y-2">
                {alerts.filter(a => !a.acknowledged).slice(0, 5).map((alert) => (
                  <AlertRow key={alert.id} alert={alert} onAcknowledge={() => acknowledgeAlert(alert.id)} />
                ))}
                {alerts.filter(a => !a.acknowledged).length === 0 && (
                  <div className="text-slate-500 text-center py-4">No active alerts</div>
                )}
              </div>
            </div>
          </div>
        )}

        {selectedTab === 'efficiency' && (
          <div className="space-y-6">
            <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
              <h3 className="font-medium mb-4 flex items-center gap-2">
                <Cpu className="w-5 h-5 text-amber-400" />
                Idle Resources ({idleResources.length})
              </h3>
              <div className="space-y-3">
                {idleResources.map((resource) => (
                  <IdleResourceCard
                    key={resource.id}
                    resource={resource}
                    onTerminate={() => terminateIdleResource(resource.id)}
                  />
                ))}
                {idleResources.length === 0 && (
                  <div className="text-center py-8 text-slate-500">
                    <CheckCircle className="w-12 h-12 mx-auto mb-2 text-green-400" />
                    <p>No idle resources detected. All resources are being utilized efficiently.</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {selectedTab === 'margins' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="lg:col-span-2 bg-slate-800 rounded-lg border border-slate-700 p-4">
              <h3 className="font-medium mb-4">Profit Margins by Service</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={profitMargins} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis type="number" stroke="#94a3b8" fontSize={12} tickFormatter={(v) => `${v}%`} />
                  <YAxis type="category" dataKey="service_name" stroke="#94a3b8" fontSize={12} width={120} />
                  <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none' }} formatter={(v: number) => [`${v.toFixed(1)}%`, 'Margin']} />
                  <Bar dataKey="margin_pct" radius={[0, 4, 4, 0]}>
                    {profitMargins.map((entry, index) => (
                      <Cell key={index} fill={entry.margin_pct >= 30 ? '#10b981' : entry.margin_pct >= 15 ? '#f59e0b' : '#ef4444'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {profitMargins.map((margin) => (
              <MarginCard key={margin.id} margin={margin} />
            ))}
          </div>
        )}

        {selectedTab === 'vendors' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {vendorUtilizations.map((vendor) => (
                <VendorCard key={vendor.id} vendor={vendor} />
              ))}
            </div>

            <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
              <h3 className="font-medium mb-4">Cost vs Utilization</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={vendorUtilizations}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="vendor_name" stroke="#94a3b8" fontSize={12} />
                  <YAxis yAxisId="left" stroke="#94a3b8" fontSize={12} tickFormatter={(v) => `$${v/1000}k`} />
                  <YAxis yAxisId="right" orientation="right" stroke="#94a3b8" fontSize={12} tickFormatter={(v) => `${v}%`} />
                  <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none' }} />
                  <Bar yAxisId="left" dataKey="cost_usd" fill="#3b82f6" radius={[4, 4, 0, 0]} name="Cost ($)" />
                  <Bar yAxisId="right" dataKey="utilization_pct" fill="#8b5cf6" radius={[4, 4, 0, 0]} name="Utilization (%)" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function KPICard({ kpi }: { kpi: ExecutiveKPI }) {
  const trendColors = {
    up: 'text-green-400',
    down: 'text-red-400',
    stable: 'text-slate-400',
  };

  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
      <div className="text-sm text-slate-400 mb-1">{kpi.name}</div>
      <div className="text-2xl font-bold">{kpi.value}</div>
      <div className={`text-sm flex items-center gap-1 ${trendColors[kpi.trend]}`}>
        {kpi.trend === 'up' && <TrendingUp className="w-4 h-4" />}
        {kpi.trend === 'down' && <TrendingUp className="w-4 h-4 rotate-180" />}
        {kpi.change_pct > 0 ? '+' : ''}{kpi.change_pct.toFixed(1)}% vs last period
      </div>
    </div>
  );
}

function IdleResourceCard({ resource, onTerminate }: { resource: IdleResource; onTerminate: () => void }) {
  return (
    <div className="flex items-center justify-between p-4 bg-slate-700/30 rounded-lg border border-amber-500/30">
      <div className="flex items-center gap-4">
        <div className={`p-2 rounded-lg ${resource.resource_type === 'gpu' ? 'bg-green-500/20' : 'bg-blue-500/20'}`}>
          <Cpu className={`w-5 h-5 ${resource.resource_type === 'gpu' ? 'text-green-400' : 'text-blue-400'}`} />
        </div>
        <div>
          <div className="font-medium">{resource.name}</div>
          <div className="text-sm text-slate-400">
            {resource.resource_type.toUpperCase()} • Idle for {resource.idle_hours}h
          </div>
        </div>
      </div>
      <div className="flex items-center gap-4">
        <div className="text-right">
          <div className="text-amber-400 font-semibold">${resource.wasted_cost_usd.toFixed(2)}/hr</div>
          <div className="text-sm text-slate-400">{resource.recommendation}</div>
        </div>
        <button
          onClick={onTerminate}
          className="p-2 bg-red-500/20 hover:bg-red-500/30 rounded-lg transition-colors"
          title="Terminate Resource"
        >
          <Power className="w-5 h-5 text-red-400" />
        </button>
      </div>
    </div>
  );
}

function MarginCard({ margin }: { margin: ProfitMargin }) {
  const marginColor = margin.margin_pct >= 30 ? 'text-green-400' : margin.margin_pct >= 15 ? 'text-amber-400' : 'text-red-400';

  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="font-medium">{margin.service_name}</div>
        <span className={`text-xl font-bold ${marginColor}`}>{margin.margin_pct.toFixed(1)}%</span>
      </div>
      <div className="grid grid-cols-2 gap-4 text-sm">
        <div>
          <span className="text-slate-400">Revenue</span>
          <div className="font-medium text-green-400">${margin.revenue_usd.toLocaleString()}</div>
        </div>
        <div>
          <span className="text-slate-400">Cost</span>
          <div className="font-medium text-red-400">${margin.cost_usd.toLocaleString()}</div>
        </div>
      </div>
    </div>
  );
}

function VendorCard({ vendor }: { vendor: VendorUtilization }) {
  const utilizationColor = vendor.utilization_pct >= 80 ? 'text-green-400' : vendor.utilization_pct >= 50 ? 'text-amber-400' : 'text-red-400';

  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
      <div className="flex items-center gap-3 mb-3">
        <Building2 className="w-5 h-5 text-violet-400" />
        <div>
          <div className="font-medium">{vendor.vendor_name}</div>
          <div className="text-sm text-slate-400">{vendor.contract_type}</div>
        </div>
      </div>
      <div className="space-y-2">
        <div className="flex justify-between">
          <span className="text-slate-400">Utilization</span>
          <span className={utilizationColor}>{vendor.utilization_pct.toFixed(1)}%</span>
        </div>
        <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
          <div
            className={`h-full transition-all ${vendor.utilization_pct >= 80 ? 'bg-green-500' : vendor.utilization_pct >= 50 ? 'bg-amber-500' : 'bg-red-500'}`}
            style={{ width: `${vendor.utilization_pct}%` }}
          />
        </div>
        <div className="flex justify-between text-sm">
          <span className="text-slate-400">Monthly Cost</span>
          <span>${vendor.cost_usd.toLocaleString()}</span>
        </div>
        <div className="flex justify-between text-sm">
          <span className="text-slate-400">Contract Value</span>
          <span>${vendor.contract_value_usd.toLocaleString()}</span>
        </div>
      </div>
    </div>
  );
}

function AlertRow({ alert, onAcknowledge }: { alert: IntelligenceAlert; onAcknowledge: () => void }) {
  const severityColors: Record<string, string> = {
    info: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
    warning: 'bg-amber-500/20 text-amber-400 border-amber-500/30',
    critical: 'bg-red-500/20 text-red-400 border-red-500/30',
  };

  return (
    <div className={`flex items-center justify-between p-3 rounded-lg border ${severityColors[alert.severity]}`}>
      <div className="flex items-center gap-3">
        <AlertTriangle className="w-5 h-5" />
        <div>
          <div className="font-medium">{alert.title}</div>
          <div className="text-sm opacity-80">{alert.message}</div>
        </div>
      </div>
      <button
        onClick={onAcknowledge}
        className="p-2 hover:bg-slate-700 rounded transition-colors"
        title="Acknowledge"
      >
        <CheckCircle className="w-5 h-5" />
      </button>
    </div>
  );
}

function generateMarginTrendData() {
  const months = ['Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  return months.map((month, i) => ({
    month,
    margin: 22 + i * 1.5 + Math.random() * 3,
  }));
}

import { useEffect, useState } from 'react';
import { DollarSign, TrendingUp, TrendingDown, AlertTriangle, FileText, RefreshCw, PieChart } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line } from 'recharts';
import { useCostsStore, CostAttribution, CostForecast, BudgetAlert } from '../../stores/costsStore';

export default function CostsView() {
  const {
    summary,
    attributions,
    forecasts,
    alerts,
    loading,
    error,
    fetchSummary,
    generateChargebackReport,
  } = useCostsStore();

  const [selectedTab, setSelectedTab] = useState<'overview' | 'attribution' | 'forecast' | 'alerts'>('overview');

  useEffect(() => {
    fetchSummary();
    const interval = setInterval(fetchSummary, 30000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="h-full flex flex-col bg-slate-900 text-white">
      {/* Header */}
      <div className="h-16 bg-slate-800 border-b border-slate-700 flex items-center px-6">
        <DollarSign className="w-6 h-6 text-emerald-400 mr-3" />
        <h1 className="text-xl font-semibold">Cost Intelligence</h1>
        <div className="flex-1" />
        <button
          onClick={() => generateChargebackReport('2024-12-01', '2024-12-31', 'csv')}
          className="mr-2 px-3 py-1.5 bg-emerald-600 hover:bg-emerald-500 rounded-lg text-sm transition-colors"
        >
          Generate Report
        </button>
        <button onClick={fetchSummary} className="p-2 hover:bg-slate-700 rounded-lg transition-colors">
          <RefreshCw className={`w-5 h-5 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Summary Bar */}
      {summary && (
        <div className="bg-slate-800/50 border-b border-slate-700 px-6 py-3 flex gap-6 text-sm">
          <div>
            <span className="text-slate-400">Total Cost:</span>{' '}
            <span className="text-emerald-400 font-semibold">${summary.total_cost_usd.toLocaleString()}</span>
            <span className={`ml-2 ${summary.total_cost_trend_pct >= 0 ? 'text-red-400' : 'text-green-400'}`}>
              {summary.total_cost_trend_pct >= 0 ? '+' : ''}{summary.total_cost_trend_pct.toFixed(1)}%
            </span>
          </div>
          <div><span className="text-slate-400">GPU:</span> ${summary.gpu_cost_usd.toLocaleString()}</div>
          <div><span className="text-slate-400">Storage:</span> ${summary.storage_cost_usd.toLocaleString()}</div>
          <div><span className="text-slate-400">Network:</span> ${summary.network_cost_usd.toLocaleString()}</div>
        </div>
      )}

      {/* Tabs */}
      <div className="border-b border-slate-700 px-6">
        <div className="flex gap-4">
          {['overview', 'attribution', 'forecast', 'alerts'].map((tab) => (
            <button
              key={tab}
              onClick={() => setSelectedTab(tab as any)}
              className={`py-3 px-4 border-b-2 transition-colors capitalize ${
                selectedTab === tab ? 'border-emerald-400 text-emerald-400' : 'border-transparent text-slate-400 hover:text-white'
              }`}
            >
              {tab}
              {tab === 'alerts' && alerts.filter(a => a.status !== 'ok').length > 0 && (
                <span className="ml-2 bg-amber-500 text-white text-xs px-2 py-0.5 rounded-full">
                  {alerts.filter(a => a.status !== 'ok').length}
                </span>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto p-6">
        {error && <div className="mb-4 p-4 bg-red-500/20 border border-red-500 rounded-lg text-red-400">{error}</div>}

        {selectedTab === 'overview' && summary && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
              <h3 className="font-medium mb-4">Cost by Team</h3>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={attributions.filter(a => a.type === 'team')}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="name" stroke="#94a3b8" fontSize={12} />
                  <YAxis stroke="#94a3b8" fontSize={12} tickFormatter={(v) => `$${v/1000}k`} />
                  <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none' }} formatter={(v: number) => [`$${v.toLocaleString()}`, 'Cost']} />
                  <Bar dataKey="cost_usd" fill="#10b981" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
              <h3 className="font-medium mb-4">13-Week Forecast</h3>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={forecasts}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="date" stroke="#94a3b8" fontSize={12} />
                  <YAxis stroke="#94a3b8" fontSize={12} tickFormatter={(v) => `$${v/1000}k`} />
                  <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none' }} />
                  <Line type="monotone" dataKey="predicted_cost_usd" stroke="#10b981" strokeWidth={2} dot={{ fill: '#10b981' }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {selectedTab === 'attribution' && (
          <div className="space-y-4">
            {attributions.map((attr) => (
              <AttributionCard key={attr.id} attribution={attr} />
            ))}
          </div>
        )}

        {selectedTab === 'forecast' && (
          <div className="space-y-4">
            <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={forecasts}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="date" stroke="#94a3b8" fontSize={12} />
                  <YAxis stroke="#94a3b8" fontSize={12} tickFormatter={(v) => `$${v/1000}k`} />
                  <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none' }} />
                  <Line type="monotone" dataKey="confidence_high" stroke="#334155" strokeDasharray="3 3" />
                  <Line type="monotone" dataKey="predicted_cost_usd" stroke="#10b981" strokeWidth={2} />
                  <Line type="monotone" dataKey="confidence_low" stroke="#334155" strokeDasharray="3 3" />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {forecasts.slice(0, 6).map((f) => (
                <div key={f.week} className="bg-slate-800 rounded-lg border border-slate-700 p-4">
                  <div className="text-sm text-slate-400">{f.date}</div>
                  <div className="text-xl font-bold text-emerald-400">${f.predicted_cost_usd.toLocaleString()}</div>
                  <div className="text-sm text-slate-400">
                    Range: ${f.confidence_low.toLocaleString()} - ${f.confidence_high.toLocaleString()}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {selectedTab === 'alerts' && (
          <div className="space-y-4">
            {alerts.map((alert) => (
              <AlertCard key={alert.id} alert={alert} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function AttributionCard({ attribution }: { attribution: CostAttribution }) {
  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
      <div className="flex items-center justify-between mb-3">
        <div>
          <div className="font-medium">{attribution.name}</div>
          <div className="text-sm text-slate-400 capitalize">{attribution.type} • {attribution.period}</div>
        </div>
        <div className="text-right">
          <div className="text-xl font-bold text-emerald-400">${attribution.cost_usd.toLocaleString()}</div>
          <div className={`text-sm flex items-center gap-1 ${attribution.cost_trend_pct >= 0 ? 'text-red-400' : 'text-green-400'}`}>
            {attribution.cost_trend_pct >= 0 ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
            {Math.abs(attribution.cost_trend_pct).toFixed(1)}%
          </div>
        </div>
      </div>
      <div className="grid grid-cols-3 gap-4 text-sm">
        <div><span className="text-slate-400">GPU Hours:</span> {attribution.gpu_hours.toLocaleString()}</div>
        <div><span className="text-slate-400">Storage:</span> {attribution.storage_gb.toLocaleString()} GB</div>
        <div><span className="text-slate-400">Network:</span> {attribution.network_gb.toLocaleString()} GB</div>
      </div>
    </div>
  );
}

function AlertCard({ alert }: { alert: BudgetAlert }) {
  const statusColors = { ok: 'border-green-500', warning: 'border-amber-500', critical: 'border-red-500' };
  const bgColors = { ok: 'bg-green-500/10', warning: 'bg-amber-500/10', critical: 'bg-red-500/10' };

  return (
    <div className={`rounded-lg border p-4 ${statusColors[alert.status]} ${bgColors[alert.status]}`}>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          {alert.status !== 'ok' && <AlertTriangle className={`w-5 h-5 ${alert.status === 'critical' ? 'text-red-400' : 'text-amber-400'}`} />}
          <div>
            <div className="font-medium">{alert.name}</div>
            <div className="text-sm text-slate-400">Alert at {alert.alert_at_pct}%</div>
          </div>
        </div>
        <span className={`px-3 py-1 rounded-full text-sm capitalize ${
          alert.status === 'ok' ? 'bg-green-500/20 text-green-400' :
          alert.status === 'warning' ? 'bg-amber-500/20 text-amber-400' : 'bg-red-500/20 text-red-400'
        }`}>{alert.status}</span>
      </div>
      <div className="mb-2">
        <div className="h-3 bg-slate-700 rounded-full overflow-hidden">
          <div
            className={`h-full transition-all ${
              alert.status === 'ok' ? 'bg-green-500' : alert.status === 'warning' ? 'bg-amber-500' : 'bg-red-500'
            }`}
            style={{ width: `${Math.min(alert.percentage_used, 100)}%` }}
          />
        </div>
      </div>
      <div className="flex justify-between text-sm text-slate-400">
        <span>${alert.current_usd.toLocaleString()} / ${alert.threshold_usd.toLocaleString()}</span>
        <span>{alert.percentage_used.toFixed(1)}% used</span>
      </div>
    </div>
  );
}

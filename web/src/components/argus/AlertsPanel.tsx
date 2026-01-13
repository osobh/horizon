import { useEffect } from 'react';
import { AlertTriangle, AlertCircle, Info, CheckCircle, Clock } from 'lucide-react';
import { useArgusStore, Alert } from '../../stores/argusStore';

const severityConfig = {
  critical: {
    icon: AlertTriangle,
    bgColor: 'bg-red-500/10',
    borderColor: 'border-red-500/50',
    textColor: 'text-red-400',
    label: 'Critical',
  },
  warning: {
    icon: AlertCircle,
    bgColor: 'bg-yellow-500/10',
    borderColor: 'border-yellow-500/50',
    textColor: 'text-yellow-400',
    label: 'Warning',
  },
  info: {
    icon: Info,
    bgColor: 'bg-blue-500/10',
    borderColor: 'border-blue-500/50',
    textColor: 'text-blue-400',
    label: 'Info',
  },
};

const stateConfig = {
  firing: { icon: AlertTriangle, color: 'text-red-400', label: 'Firing' },
  pending: { icon: Clock, color: 'text-yellow-400', label: 'Pending' },
  resolved: { icon: CheckCircle, color: 'text-green-400', label: 'Resolved' },
};

function AlertCard({ alert }: { alert: Alert }) {
  const severity = severityConfig[alert.severity] || severityConfig.info;
  const state = stateConfig[alert.state] || stateConfig.pending;
  const SeverityIcon = severity.icon;
  const StateIcon = state.icon;

  const formatTime = (timestamp: number) => {
    const date = new Date(timestamp * 1000);
    return date.toLocaleString();
  };

  const duration = () => {
    const now = Date.now() / 1000;
    const diff = now - alert.active_at;
    if (diff < 60) return `${Math.floor(diff)}s`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h`;
    return `${Math.floor(diff / 86400)}d`;
  };

  return (
    <div
      className={`${severity.bgColor} ${severity.borderColor} border rounded-lg p-4 mb-3`}
    >
      <div className="flex items-start justify-between">
        <div className="flex items-start gap-3">
          <SeverityIcon className={`w-5 h-5 mt-0.5 ${severity.textColor}`} />
          <div>
            <div className="flex items-center gap-2">
              <h4 className="font-medium text-white">{alert.name}</h4>
              <span
                className={`text-xs px-2 py-0.5 rounded ${severity.bgColor} ${severity.textColor}`}
              >
                {severity.label}
              </span>
            </div>
            <p className="text-sm text-slate-300 mt-1">{alert.summary}</p>
            {alert.description && (
              <p className="text-xs text-slate-400 mt-2">{alert.description}</p>
            )}
            <div className="flex flex-wrap gap-2 mt-2">
              {Object.entries(alert.labels).map(([key, value]) => (
                <span
                  key={key}
                  className="text-xs bg-slate-700 text-slate-300 px-2 py-0.5 rounded"
                >
                  {key}={value}
                </span>
              ))}
            </div>
          </div>
        </div>
        <div className="text-right">
          <div className="flex items-center gap-1">
            <StateIcon className={`w-4 h-4 ${state.color}`} />
            <span className={`text-sm ${state.color}`}>{state.label}</span>
          </div>
          <p className="text-xs text-slate-400 mt-1">Active {duration()}</p>
          <p className="text-xs text-slate-500">{formatTime(alert.active_at)}</p>
        </div>
      </div>
    </div>
  );
}

export function AlertsPanel() {
  const { alerts, loading, error, fetchAlerts } = useArgusStore();

  useEffect(() => {
    fetchAlerts();
    // Refresh every 30 seconds
    const interval = setInterval(fetchAlerts, 30000);
    return () => clearInterval(interval);
  }, [fetchAlerts]);

  const firingAlerts = alerts.filter((a) => a.state === 'firing');
  const pendingAlerts = alerts.filter((a) => a.state === 'pending');
  const resolvedAlerts = alerts.filter((a) => a.state === 'resolved');

  if (loading && alerts.length === 0) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-cyan-500" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-500/10 border border-red-500/50 rounded-lg p-4">
        <p className="text-red-400">Error loading alerts: {error}</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Summary */}
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-red-400" />
            <span className="text-2xl font-bold text-red-400">{firingAlerts.length}</span>
          </div>
          <p className="text-sm text-slate-400 mt-1">Firing</p>
        </div>
        <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-4">
          <div className="flex items-center gap-2">
            <Clock className="w-5 h-5 text-yellow-400" />
            <span className="text-2xl font-bold text-yellow-400">{pendingAlerts.length}</span>
          </div>
          <p className="text-sm text-slate-400 mt-1">Pending</p>
        </div>
        <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
          <div className="flex items-center gap-2">
            <CheckCircle className="w-5 h-5 text-green-400" />
            <span className="text-2xl font-bold text-green-400">{resolvedAlerts.length}</span>
          </div>
          <p className="text-sm text-slate-400 mt-1">Resolved</p>
        </div>
      </div>

      {/* Alert List */}
      {firingAlerts.length > 0 && (
        <div>
          <h3 className="text-lg font-medium text-white mb-3">Firing Alerts</h3>
          {firingAlerts.map((alert) => (
            <AlertCard key={alert.fingerprint} alert={alert} />
          ))}
        </div>
      )}

      {pendingAlerts.length > 0 && (
        <div>
          <h3 className="text-lg font-medium text-white mb-3">Pending Alerts</h3>
          {pendingAlerts.map((alert) => (
            <AlertCard key={alert.fingerprint} alert={alert} />
          ))}
        </div>
      )}

      {alerts.length === 0 && (
        <div className="text-center py-8">
          <CheckCircle className="w-12 h-12 text-green-400 mx-auto mb-3" />
          <p className="text-slate-300">No alerts - all systems healthy</p>
        </div>
      )}
    </div>
  );
}

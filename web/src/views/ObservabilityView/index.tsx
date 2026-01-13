import { useEffect, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  Server,
  Search,
  RefreshCw,
  Wifi,
  WifiOff,
  Settings,
} from 'lucide-react';
import { useArgusStore } from '../../stores/argusStore';
import { MetricsExplorer, AlertsPanel, TargetsPanel } from '../../components/argus';

type TabType = 'metrics' | 'alerts' | 'targets';

export default function ObservabilityView() {
  const {
    status,
    alerts,
    targets,
    loading,
    error,
    fetchStatus,
    fetchAlerts,
    fetchTargets,
    setServerUrl,
  } = useArgusStore();

  const [selectedTab, setSelectedTab] = useState<TabType>('metrics');
  const [showSettings, setShowSettings] = useState(false);
  const [serverUrlInput, setServerUrlInput] = useState('');

  useEffect(() => {
    fetchStatus();
    fetchAlerts();
    fetchTargets();

    // Refresh status every 30 seconds
    const interval = setInterval(() => {
      fetchStatus();
    }, 30000);
    return () => clearInterval(interval);
  }, [fetchStatus, fetchAlerts, fetchTargets]);

  useEffect(() => {
    if (status?.server_url) {
      setServerUrlInput(status.server_url);
    }
  }, [status?.server_url]);

  const handleServerUrlSave = async () => {
    await setServerUrl(serverUrlInput);
    setShowSettings(false);
  };

  const firingAlerts = alerts.filter((a) => a.state === 'firing');
  const healthyTargets = targets.filter((t) => t.health === 'up');

  return (
    <div className="h-full flex flex-col bg-slate-900 text-white">
      {/* Header */}
      <div className="h-16 bg-slate-800 border-b border-slate-700 flex items-center px-6">
        <Activity className="w-6 h-6 text-cyan-400 mr-3" />
        <h1 className="text-xl font-semibold">Observability</h1>
        <span className="ml-2 text-sm text-slate-400">(Argus)</span>
        <div className="flex-1" />

        {/* Connection Status */}
        <div className="flex items-center gap-2 mr-4">
          {status?.connected ? (
            <>
              <Wifi className="w-4 h-4 text-green-400" />
              <span className="text-sm text-green-400">Connected</span>
            </>
          ) : (
            <>
              <WifiOff className="w-4 h-4 text-red-400" />
              <span className="text-sm text-red-400">Disconnected</span>
            </>
          )}
        </div>

        <button
          onClick={() => setShowSettings(true)}
          className="p-2 hover:bg-slate-700 rounded-lg transition-colors mr-2"
          title="Settings"
        >
          <Settings className="w-5 h-5 text-slate-400" />
        </button>

        <button
          onClick={() => {
            fetchStatus();
            fetchAlerts();
            fetchTargets();
          }}
          className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
          title="Refresh"
        >
          <RefreshCw className={`w-5 h-5 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Status Bar */}
      {status && (
        <div className="bg-slate-800/50 border-b border-slate-700 px-6 py-3 flex gap-6 text-sm">
          <div className="flex items-center gap-2">
            <span className="text-slate-400">Server:</span>
            <span className="text-slate-300 font-mono text-xs">{status.server_url}</span>
          </div>
          <div className="flex items-center gap-2">
            <AlertTriangle
              className={`w-4 h-4 ${firingAlerts.length > 0 ? 'text-red-400' : 'text-slate-500'}`}
            />
            <span className="text-slate-400">Alerts:</span>
            <span className={firingAlerts.length > 0 ? 'text-red-400' : 'text-green-400'}>
              {firingAlerts.length} firing
            </span>
          </div>
          <div className="flex items-center gap-2">
            <Server className="w-4 h-4 text-slate-500" />
            <span className="text-slate-400">Targets:</span>
            <span className="text-green-400">{healthyTargets.length}</span>
            <span className="text-slate-400">/</span>
            <span className="text-slate-300">{targets.length}</span>
          </div>
          {status.version && (
            <div>
              <span className="text-slate-400">Version:</span>{' '}
              <span className="text-slate-300">{status.version}</span>
            </div>
          )}
        </div>
      )}

      {/* Tabs */}
      <div className="border-b border-slate-700 px-6">
        <div className="flex gap-4">
          <button
            onClick={() => setSelectedTab('metrics')}
            className={`py-3 px-4 border-b-2 transition-colors flex items-center gap-2 ${
              selectedTab === 'metrics'
                ? 'border-cyan-400 text-cyan-400'
                : 'border-transparent text-slate-400 hover:text-white'
            }`}
          >
            <Search className="w-4 h-4" />
            Metrics Explorer
          </button>
          <button
            onClick={() => setSelectedTab('alerts')}
            className={`py-3 px-4 border-b-2 transition-colors flex items-center gap-2 ${
              selectedTab === 'alerts'
                ? 'border-cyan-400 text-cyan-400'
                : 'border-transparent text-slate-400 hover:text-white'
            }`}
          >
            <AlertTriangle className="w-4 h-4" />
            Alerts
            {firingAlerts.length > 0 && (
              <span className="px-1.5 py-0.5 text-xs bg-red-500 text-white rounded-full">
                {firingAlerts.length}
              </span>
            )}
          </button>
          <button
            onClick={() => setSelectedTab('targets')}
            className={`py-3 px-4 border-b-2 transition-colors flex items-center gap-2 ${
              selectedTab === 'targets'
                ? 'border-cyan-400 text-cyan-400'
                : 'border-transparent text-slate-400 hover:text-white'
            }`}
          >
            <Server className="w-4 h-4" />
            Targets
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto p-6">
        {error && (
          <div className="mb-4 p-4 bg-red-500/20 border border-red-500 rounded-lg text-red-400">
            {error}
          </div>
        )}

        {selectedTab === 'metrics' && <MetricsExplorer />}
        {selectedTab === 'alerts' && <AlertsPanel />}
        {selectedTab === 'targets' && <TargetsPanel />}
      </div>

      {/* Settings Modal */}
      {showSettings && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-slate-800 border border-slate-700 rounded-lg p-6 w-96">
            <h3 className="text-lg font-medium mb-4">Argus Settings</h3>

            <div className="space-y-4">
              <div>
                <label className="block text-sm text-slate-400 mb-1">Server URL</label>
                <input
                  type="text"
                  value={serverUrlInput}
                  onChange={(e) => setServerUrlInput(e.target.value)}
                  placeholder="http://localhost:9090"
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2
                             text-white focus:outline-none focus:border-cyan-500"
                />
              </div>
            </div>

            <div className="flex justify-end gap-3 mt-6">
              <button
                onClick={() => setShowSettings(false)}
                className="px-4 py-2 text-slate-400 hover:text-white transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleServerUrlSave}
                className="px-4 py-2 bg-cyan-500 hover:bg-cyan-600 text-white rounded-lg
                           transition-colors"
              >
                Save
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

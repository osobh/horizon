import { useEffect, useState } from 'react';
import {
  Settings,
  Shield,
  Key,
  UserCog,
  Sliders,
  RefreshCw,
  Plus,
  Trash2,
  Edit,
  CheckCircle,
  XCircle,
  ToggleLeft,
  ToggleRight,
  Save,
} from 'lucide-react';
import { useSettingsStore, Policy, Quota, AppSettings } from '../../stores/settingsStore';

export default function SettingsView() {
  const {
    summary,
    policies,
    quotas,
    appSettings,
    loading,
    error,
    fetchSummary,
    fetchPolicies,
    fetchQuotas,
    fetchAppSettings,
    togglePolicy,
    deletePolicy,
    deleteQuota,
    updateAppSettings,
  } = useSettingsStore();

  const [selectedTab, setSelectedTab] = useState<'overview' | 'policies' | 'quotas' | 'preferences'>('overview');
  const [editingSettings, setEditingSettings] = useState<AppSettings | null>(null);

  useEffect(() => {
    fetchSummary();
    fetchPolicies();
    fetchQuotas();
    fetchAppSettings();
  }, []);

  const handleSaveSettings = () => {
    if (editingSettings) {
      updateAppSettings(editingSettings);
      setEditingSettings(null);
    }
  };

  return (
    <div className="h-full flex flex-col bg-slate-900 text-white">
      {/* Header */}
      <div className="h-16 bg-slate-800 border-b border-slate-700 flex items-center px-6">
        <Settings className="w-6 h-6 text-slate-400 mr-3" />
        <h1 className="text-xl font-semibold">Settings</h1>
        <div className="flex-1" />
        <button onClick={() => { fetchSummary(); fetchPolicies(); fetchQuotas(); }} className="p-2 hover:bg-slate-700 rounded-lg transition-colors">
          <RefreshCw className={`w-5 h-5 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Summary Bar */}
      {summary && (
        <div className="bg-slate-800/50 border-b border-slate-700 px-6 py-3 flex gap-6 text-sm">
          <div>
            <span className="text-slate-400">Active Policies:</span>{' '}
            <span className="text-green-400 font-semibold">{summary.active_policies}</span>
            <span className="text-slate-500"> / {summary.total_policies}</span>
          </div>
          <div>
            <span className="text-slate-400">Configured Quotas:</span>{' '}
            <span className="text-blue-400">{summary.total_quotas}</span>
          </div>
          <div>
            <span className="text-slate-400">Theme:</span>{' '}
            <span className="text-slate-300 capitalize">{summary.theme}</span>
          </div>
          <div>
            <span className="text-slate-400">Auto-refresh:</span>{' '}
            <span className={summary.auto_refresh ? 'text-green-400' : 'text-slate-500'}>
              {summary.auto_refresh ? 'Enabled' : 'Disabled'}
            </span>
          </div>
        </div>
      )}

      {/* Tabs */}
      <div className="border-b border-slate-700 px-6">
        <div className="flex gap-4">
          {['overview', 'policies', 'quotas', 'preferences'].map((tab) => (
            <button
              key={tab}
              onClick={() => setSelectedTab(tab as any)}
              className={`py-3 px-4 border-b-2 transition-colors capitalize ${
                selectedTab === tab ? 'border-slate-400 text-white' : 'border-transparent text-slate-400 hover:text-white'
              }`}
            >
              {tab}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto p-6">
        {error && <div className="mb-4 p-4 bg-red-500/20 border border-red-500 rounded-lg text-red-400">{error}</div>}

        {selectedTab === 'overview' && summary && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Quick Stats */}
            <div className="grid grid-cols-2 gap-4">
              <StatCard icon={Shield} label="Policies" value={summary.total_policies.toString()} color="text-green-400" />
              <StatCard icon={Sliders} label="Quotas" value={summary.total_quotas.toString()} color="text-blue-400" />
              <StatCard icon={CheckCircle} label="Active Policies" value={summary.active_policies.toString()} color="text-emerald-400" />
              <StatCard icon={XCircle} label="Inactive Policies" value={(summary.total_policies - summary.active_policies).toString()} color="text-slate-500" />
            </div>

            {/* Recent Policies */}
            <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
              <h3 className="font-medium mb-4 flex items-center gap-2">
                <Shield className="w-5 h-5 text-green-400" />
                Recent Policies
              </h3>
              <div className="space-y-2">
                {policies.slice(0, 5).map((policy) => (
                  <div key={policy.id} className="flex items-center justify-between p-2 bg-slate-700/30 rounded">
                    <div>
                      <div className="font-medium">{policy.name}</div>
                      <div className="text-sm text-slate-400">{policy.policy_type}</div>
                    </div>
                    <span className={`px-2 py-1 rounded text-xs ${policy.enabled ? 'bg-green-500/20 text-green-400' : 'bg-slate-500/20 text-slate-400'}`}>
                      {policy.enabled ? 'Active' : 'Inactive'}
                    </span>
                  </div>
                ))}
                {policies.length === 0 && (
                  <div className="text-slate-500 text-center py-4">No policies configured</div>
                )}
              </div>
            </div>

            {/* Resource Quotas Summary */}
            <div className="lg:col-span-2 bg-slate-800 rounded-lg border border-slate-700 p-4">
              <h3 className="font-medium mb-4 flex items-center gap-2">
                <Sliders className="w-5 h-5 text-blue-400" />
                Resource Quotas
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {quotas.slice(0, 6).map((quota) => (
                  <QuotaCard key={quota.id} quota={quota} />
                ))}
              </div>
            </div>
          </div>
        )}

        {selectedTab === 'policies' && (
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <h3 className="font-medium">Governance Policies</h3>
              <button className="px-3 py-1.5 bg-green-600 hover:bg-green-500 rounded-lg text-sm transition-colors flex items-center gap-2">
                <Plus className="w-4 h-4" />
                New Policy
              </button>
            </div>
            <div className="space-y-3">
              {policies.map((policy) => (
                <PolicyCard
                  key={policy.id}
                  policy={policy}
                  onToggle={() => togglePolicy(policy.id)}
                  onDelete={() => deletePolicy(policy.id)}
                />
              ))}
              {policies.length === 0 && (
                <div className="p-8 text-center text-slate-500 bg-slate-800 rounded-lg border border-slate-700">
                  No policies configured. Click "New Policy" to create one.
                </div>
              )}
            </div>
          </div>
        )}

        {selectedTab === 'quotas' && (
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <h3 className="font-medium">Resource Quotas</h3>
              <button className="px-3 py-1.5 bg-blue-600 hover:bg-blue-500 rounded-lg text-sm transition-colors flex items-center gap-2">
                <Plus className="w-4 h-4" />
                New Quota
              </button>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {quotas.map((quota) => (
                <QuotaDetailCard
                  key={quota.id}
                  quota={quota}
                  onDelete={() => deleteQuota(quota.id)}
                />
              ))}
              {quotas.length === 0 && (
                <div className="md:col-span-2 p-8 text-center text-slate-500 bg-slate-800 rounded-lg border border-slate-700">
                  No quotas configured. Click "New Quota" to create one.
                </div>
              )}
            </div>
          </div>
        )}

        {selectedTab === 'preferences' && appSettings && (
          <div className="max-w-2xl space-y-6">
            <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
              <h3 className="font-medium mb-4 flex items-center gap-2">
                <UserCog className="w-5 h-5 text-slate-400" />
                Application Settings
              </h3>
              <div className="space-y-4">
                <SettingRow
                  label="Theme"
                  description="Choose between dark and light mode"
                >
                  <select
                    value={editingSettings?.theme ?? appSettings.theme}
                    onChange={(e) => setEditingSettings({ ...appSettings, ...editingSettings, theme: e.target.value as 'dark' | 'light' | 'system' })}
                    className="bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-sm"
                  >
                    <option value="dark">Dark</option>
                    <option value="light">Light</option>
                    <option value="system">System</option>
                  </select>
                </SettingRow>

                <SettingRow
                  label="Auto-refresh"
                  description="Automatically refresh data at regular intervals"
                >
                  <button
                    onClick={() => setEditingSettings({ ...appSettings, ...editingSettings, auto_refresh: !(editingSettings?.auto_refresh ?? appSettings.auto_refresh) })}
                    className="text-2xl"
                  >
                    {(editingSettings?.auto_refresh ?? appSettings.auto_refresh) ? (
                      <ToggleRight className="w-10 h-10 text-green-400" />
                    ) : (
                      <ToggleLeft className="w-10 h-10 text-slate-500" />
                    )}
                  </button>
                </SettingRow>

                <SettingRow
                  label="Refresh Interval"
                  description="How often to refresh data (in seconds)"
                >
                  <input
                    type="number"
                    value={editingSettings?.refresh_interval_seconds ?? appSettings.refresh_interval_seconds}
                    onChange={(e) => setEditingSettings({ ...appSettings, ...editingSettings, refresh_interval_seconds: parseInt(e.target.value) || 30 })}
                    className="bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-sm w-24"
                    min="5"
                    max="300"
                  />
                </SettingRow>

                <SettingRow
                  label="Notifications"
                  description="Enable desktop notifications for alerts"
                >
                  <button
                    onClick={() => setEditingSettings({ ...appSettings, ...editingSettings, notifications_enabled: !(editingSettings?.notifications_enabled ?? appSettings.notifications_enabled) })}
                    className="text-2xl"
                  >
                    {(editingSettings?.notifications_enabled ?? appSettings.notifications_enabled) ? (
                      <ToggleRight className="w-10 h-10 text-green-400" />
                    ) : (
                      <ToggleLeft className="w-10 h-10 text-slate-500" />
                    )}
                  </button>
                </SettingRow>

                <SettingRow
                  label="Sound Alerts"
                  description="Play sounds for critical alerts"
                >
                  <button
                    onClick={() => setEditingSettings({ ...appSettings, ...editingSettings, sound_enabled: !(editingSettings?.sound_enabled ?? appSettings.sound_enabled) })}
                    className="text-2xl"
                  >
                    {(editingSettings?.sound_enabled ?? appSettings.sound_enabled) ? (
                      <ToggleRight className="w-10 h-10 text-green-400" />
                    ) : (
                      <ToggleLeft className="w-10 h-10 text-slate-500" />
                    )}
                  </button>
                </SettingRow>

                <SettingRow
                  label="Default Cluster"
                  description="The cluster to connect to on startup"
                >
                  <input
                    type="text"
                    value={editingSettings?.default_cluster ?? appSettings.default_cluster}
                    onChange={(e) => setEditingSettings({ ...appSettings, ...editingSettings, default_cluster: e.target.value })}
                    className="bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-sm w-48"
                    placeholder="localhost:8080"
                  />
                </SettingRow>
              </div>

              {editingSettings && (
                <div className="mt-6 flex gap-2">
                  <button
                    onClick={handleSaveSettings}
                    className="px-4 py-2 bg-green-600 hover:bg-green-500 rounded-lg text-sm transition-colors flex items-center gap-2"
                  >
                    <Save className="w-4 h-4" />
                    Save Changes
                  </button>
                  <button
                    onClick={() => setEditingSettings(null)}
                    className="px-4 py-2 bg-slate-600 hover:bg-slate-500 rounded-lg text-sm transition-colors"
                  >
                    Cancel
                  </button>
                </div>
              )}
            </div>

            <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
              <h3 className="font-medium mb-4 flex items-center gap-2">
                <Key className="w-5 h-5 text-amber-400" />
                API Keys
              </h3>
              <div className="space-y-3">
                {generateApiKeys().map((apiKey) => (
                  <div key={apiKey.id} className="flex items-center justify-between p-3 bg-slate-700/30 rounded-lg">
                    <div>
                      <div className="font-medium">{apiKey.name}</div>
                      <div className="text-sm text-slate-400 font-mono">{apiKey.prefix}...{apiKey.suffix}</div>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-slate-500">Created {apiKey.created}</span>
                      <button className="p-1 hover:bg-slate-600 rounded text-red-400">
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                ))}
                <button className="w-full p-3 border border-dashed border-slate-600 rounded-lg text-slate-400 hover:border-slate-500 hover:text-slate-300 transition-colors flex items-center justify-center gap-2">
                  <Plus className="w-4 h-4" />
                  Generate New API Key
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function StatCard({ icon: Icon, label, value, color }: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: string;
  color: string;
}) {
  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
      <div className="flex items-center gap-3 mb-2">
        <Icon className={`w-5 h-5 ${color}`} />
        <span className="text-slate-400 text-sm">{label}</span>
      </div>
      <div className="text-2xl font-bold">{value}</div>
    </div>
  );
}

function PolicyCard({ policy, onToggle, onDelete }: { policy: Policy; onToggle: () => void; onDelete: () => void }) {
  const priorityColors: Record<string, string> = {
    low: 'text-slate-400',
    medium: 'text-amber-400',
    high: 'text-red-400',
  };

  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <Shield className={`w-5 h-5 ${policy.enabled ? 'text-green-400' : 'text-slate-500'}`} />
          <div>
            <div className="font-medium">{policy.name}</div>
            <div className="text-sm text-slate-400">{policy.policy_type} • Priority: <span className={priorityColors[policy.priority]}>{policy.priority}</span></div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={onToggle} className="text-2xl">
            {policy.enabled ? (
              <ToggleRight className="w-8 h-8 text-green-400" />
            ) : (
              <ToggleLeft className="w-8 h-8 text-slate-500" />
            )}
          </button>
          <button className="p-2 hover:bg-slate-700 rounded-lg transition-colors">
            <Edit className="w-4 h-4 text-slate-400" />
          </button>
          <button onClick={onDelete} className="p-2 hover:bg-slate-700 rounded-lg transition-colors text-red-400">
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </div>
      <div className="text-sm text-slate-400">{policy.description}</div>
      <div className="mt-2 text-xs text-slate-500">
        Last updated: {new Date(policy.updated_at).toLocaleDateString()}
      </div>
    </div>
  );
}

function QuotaCard({ quota }: { quota: Quota }) {
  const usagePercent = (quota.current_usage / quota.limit) * 100;
  const usageColor = usagePercent >= 90 ? 'bg-red-500' : usagePercent >= 70 ? 'bg-amber-500' : 'bg-green-500';

  return (
    <div className="bg-slate-700/30 rounded-lg p-3">
      <div className="flex justify-between items-center mb-2">
        <span className="text-sm font-medium">{quota.resource_type}</span>
        <span className="text-sm text-slate-400">{quota.scope}: {quota.scope_id}</span>
      </div>
      <div className="h-2 bg-slate-600 rounded-full overflow-hidden mb-1">
        <div className={`h-full ${usageColor}`} style={{ width: `${Math.min(usagePercent, 100)}%` }} />
      </div>
      <div className="text-xs text-slate-400">
        {quota.current_usage.toLocaleString()} / {quota.limit.toLocaleString()} {quota.unit}
      </div>
    </div>
  );
}

function QuotaDetailCard({ quota, onDelete }: { quota: Quota; onDelete: () => void }) {
  const usagePercent = (quota.current_usage / quota.limit) * 100;
  const usageColor = usagePercent >= 90 ? 'text-red-400' : usagePercent >= 70 ? 'text-amber-400' : 'text-green-400';

  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
      <div className="flex items-center justify-between mb-3">
        <div>
          <div className="font-medium">{quota.resource_type}</div>
          <div className="text-sm text-slate-400">{quota.scope}: {quota.scope_id}</div>
        </div>
        <div className="flex items-center gap-2">
          <button className="p-2 hover:bg-slate-700 rounded-lg transition-colors">
            <Edit className="w-4 h-4 text-slate-400" />
          </button>
          <button onClick={onDelete} className="p-2 hover:bg-slate-700 rounded-lg transition-colors text-red-400">
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </div>
      <div className="space-y-2">
        <div className="flex justify-between text-sm">
          <span className="text-slate-400">Current Usage</span>
          <span className={usageColor}>{quota.current_usage.toLocaleString()} {quota.unit}</span>
        </div>
        <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
          <div
            className={`h-full ${usagePercent >= 90 ? 'bg-red-500' : usagePercent >= 70 ? 'bg-amber-500' : 'bg-green-500'}`}
            style={{ width: `${Math.min(usagePercent, 100)}%` }}
          />
        </div>
        <div className="flex justify-between text-sm">
          <span className="text-slate-400">Limit</span>
          <span>{quota.limit.toLocaleString()} {quota.unit}</span>
        </div>
        {quota.soft_limit && (
          <div className="flex justify-between text-sm">
            <span className="text-slate-400">Soft Limit</span>
            <span className="text-amber-400">{quota.soft_limit.toLocaleString()} {quota.unit}</span>
          </div>
        )}
      </div>
    </div>
  );
}

function SettingRow({ label, description, children }: { label: string; description: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between py-3 border-b border-slate-700 last:border-0">
      <div>
        <div className="font-medium">{label}</div>
        <div className="text-sm text-slate-400">{description}</div>
      </div>
      {children}
    </div>
  );
}

function generateApiKeys() {
  return [
    { id: '1', name: 'Production API Key', prefix: 'hzn_prod', suffix: 'x7k2', created: '2024-12-01' },
    { id: '2', name: 'Development API Key', prefix: 'hzn_dev', suffix: 'm9p1', created: '2024-11-15' },
    { id: '3', name: 'CI/CD Pipeline Key', prefix: 'hzn_ci', suffix: 'b3n8', created: '2024-10-20' },
  ];
}

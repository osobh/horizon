import { useEffect, useState } from 'react';
import {
  GitBranch,
  Server,
  Shield,
  RefreshCw,
  Plus,
  Settings,
  Wifi,
  WifiOff,
  CheckCircle,
  XCircle,
  Loader2,
  Clock,
} from 'lucide-react';
import { useHpcCiStore, TriggerParams } from '../../stores/hpcciStore';
import { PipelineList, PipelineDetail, AgentList, ApprovalPanel } from '../../components/hpcci';

type TabType = 'pipelines' | 'agents' | 'approvals';

export default function PipelinesView() {
  const {
    status,
    approvals,
    loading,
    error,
    fetchStatus,
    fetchApprovals,
    fetchDashboardSummary,
    triggerPipeline,
    setServerUrl,
  } = useHpcCiStore();

  const [selectedTab, setSelectedTab] = useState<TabType>('pipelines');
  const [selectedPipelineId, setSelectedPipelineId] = useState<string | null>(null);
  const [showSettings, setShowSettings] = useState(false);
  const [showTrigger, setShowTrigger] = useState(false);
  const [serverUrlInput, setServerUrlInput] = useState('');
  const [dashboardSummary, setDashboardSummary] = useState<{
    pipelines_running: number;
    pipelines_queued: number;
    pipelines_succeeded_24h: number;
    pipelines_failed_24h: number;
  } | null>(null);

  useEffect(() => {
    fetchStatus();
    fetchApprovals();
    fetchDashboardSummary().then(setDashboardSummary).catch(() => {});

    const interval = setInterval(() => {
      fetchStatus();
      fetchDashboardSummary().then(setDashboardSummary).catch(() => {});
    }, 30000);
    return () => clearInterval(interval);
  }, [fetchStatus, fetchApprovals, fetchDashboardSummary]);

  useEffect(() => {
    if (status?.server_url) {
      setServerUrlInput(status.server_url);
    }
  }, [status?.server_url]);

  const handleServerUrlSave = async () => {
    await setServerUrl(serverUrlInput);
    setShowSettings(false);
  };

  const pendingApprovals = approvals.filter((a) => a.status === 'pending');

  return (
    <div className="h-full flex flex-col bg-slate-900 text-white">
      {/* Header */}
      <div className="h-16 bg-slate-800 border-b border-slate-700 flex items-center px-6">
        <GitBranch className="w-6 h-6 text-purple-400 mr-3" />
        <h1 className="text-xl font-semibold">Pipelines</h1>
        <span className="ml-2 text-sm text-slate-400">(HPC-CI)</span>
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
          onClick={() => setShowTrigger(true)}
          className="flex items-center gap-2 px-3 py-1.5 bg-purple-500 hover:bg-purple-600 text-white rounded-lg transition-colors mr-2"
        >
          <Plus className="w-4 h-4" />
          Trigger
        </button>

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
            fetchDashboardSummary().then(setDashboardSummary).catch(() => {});
          }}
          className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
          title="Refresh"
        >
          <RefreshCw className={`w-5 h-5 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Status Bar */}
      {dashboardSummary && (
        <div className="bg-slate-800/50 border-b border-slate-700 px-6 py-3 flex gap-6 text-sm">
          <StatusItem
            icon={Loader2}
            label="Running"
            value={dashboardSummary.pipelines_running}
            color="text-blue-400"
            spinning={dashboardSummary.pipelines_running > 0}
          />
          <StatusItem
            icon={Clock}
            label="Queued"
            value={dashboardSummary.pipelines_queued}
            color="text-slate-400"
          />
          <StatusItem
            icon={CheckCircle}
            label="Succeeded (24h)"
            value={dashboardSummary.pipelines_succeeded_24h}
            color="text-green-400"
          />
          <StatusItem
            icon={XCircle}
            label="Failed (24h)"
            value={dashboardSummary.pipelines_failed_24h}
            color="text-red-400"
          />
        </div>
      )}

      {/* Tabs */}
      <div className="border-b border-slate-700 px-6">
        <div className="flex gap-4">
          <TabButton
            active={selectedTab === 'pipelines'}
            onClick={() => {
              setSelectedTab('pipelines');
              setSelectedPipelineId(null);
            }}
            icon={GitBranch}
            label="Pipelines"
          />
          <TabButton
            active={selectedTab === 'agents'}
            onClick={() => setSelectedTab('agents')}
            icon={Server}
            label="Agents"
          />
          <TabButton
            active={selectedTab === 'approvals'}
            onClick={() => setSelectedTab('approvals')}
            icon={Shield}
            label="Approvals"
            badge={pendingApprovals.length > 0 ? pendingApprovals.length : undefined}
          />
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto p-6">
        {error && (
          <div className="mb-4 p-4 bg-red-500/20 border border-red-500 rounded-lg text-red-400">
            {error}
          </div>
        )}

        {selectedTab === 'pipelines' && !selectedPipelineId && (
          <PipelineList onSelectPipeline={setSelectedPipelineId} />
        )}
        {selectedTab === 'pipelines' && selectedPipelineId && (
          <PipelineDetail
            pipelineId={selectedPipelineId}
            onBack={() => setSelectedPipelineId(null)}
          />
        )}
        {selectedTab === 'agents' && <AgentList />}
        {selectedTab === 'approvals' && <ApprovalPanel />}
      </div>

      {/* Settings Modal */}
      {showSettings && (
        <Modal title="HPC-CI Settings" onClose={() => setShowSettings(false)}>
          <div className="space-y-4">
            <div>
              <label className="block text-sm text-slate-400 mb-1">Server URL</label>
              <input
                type="text"
                value={serverUrlInput}
                onChange={(e) => setServerUrlInput(e.target.value)}
                placeholder="http://localhost:9000"
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2
                           text-white focus:outline-none focus:border-purple-500"
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
              className="px-4 py-2 bg-purple-500 hover:bg-purple-600 text-white rounded-lg
                         transition-colors"
            >
              Save
            </button>
          </div>
        </Modal>
      )}

      {/* Trigger Pipeline Modal */}
      {showTrigger && (
        <TriggerPipelineModal
          onClose={() => setShowTrigger(false)}
          onTrigger={async (params) => {
            await triggerPipeline(params);
            setShowTrigger(false);
          }}
        />
      )}
    </div>
  );
}

interface StatusItemProps {
  icon: React.ElementType;
  label: string;
  value: number;
  color: string;
  spinning?: boolean;
}

function StatusItem({ icon: Icon, label, value, color, spinning }: StatusItemProps) {
  return (
    <div className="flex items-center gap-2">
      <Icon className={`w-4 h-4 ${color} ${spinning ? 'animate-spin' : ''}`} />
      <span className="text-slate-400">{label}:</span>
      <span className={color}>{value}</span>
    </div>
  );
}

interface TabButtonProps {
  active: boolean;
  onClick: () => void;
  icon: React.ElementType;
  label: string;
  badge?: number;
}

function TabButton({ active, onClick, icon: Icon, label, badge }: TabButtonProps) {
  return (
    <button
      onClick={onClick}
      className={`py-3 px-4 border-b-2 transition-colors flex items-center gap-2 ${
        active
          ? 'border-purple-400 text-purple-400'
          : 'border-transparent text-slate-400 hover:text-white'
      }`}
    >
      <Icon className="w-4 h-4" />
      {label}
      {badge !== undefined && (
        <span className="px-1.5 py-0.5 text-xs bg-yellow-500 text-white rounded-full">
          {badge}
        </span>
      )}
    </button>
  );
}

interface ModalProps {
  title: string;
  onClose: () => void;
  children: React.ReactNode;
}

function Modal({ title, onClose, children }: ModalProps) {
  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={onClose}>
      <div className="bg-slate-800 border border-slate-700 rounded-lg p-6 w-96" onClick={e => e.stopPropagation()}>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-medium">{title}</h3>
          <button onClick={onClose} className="text-slate-400 hover:text-white">&times;</button>
        </div>
        {children}
      </div>
    </div>
  );
}

interface TriggerPipelineModalProps {
  onClose: () => void;
  onTrigger: (params: TriggerParams) => Promise<void>;
}

function TriggerPipelineModal({ onClose, onTrigger }: TriggerPipelineModalProps) {
  const [repo, setRepo] = useState('');
  const [branch, setBranch] = useState('main');
  const [sha, setSha] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!repo) return;

    setLoading(true);
    try {
      await onTrigger({
        repo,
        branch,
        sha: sha || undefined,
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <Modal title="Trigger Pipeline" onClose={onClose}>
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm text-slate-400 mb-1">Repository *</label>
          <input
            type="text"
            value={repo}
            onChange={(e) => setRepo(e.target.value)}
            placeholder="owner/repo"
            className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2
                       text-white focus:outline-none focus:border-purple-500"
            required
          />
        </div>
        <div>
          <label className="block text-sm text-slate-400 mb-1">Branch</label>
          <input
            type="text"
            value={branch}
            onChange={(e) => setBranch(e.target.value)}
            placeholder="main"
            className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2
                       text-white focus:outline-none focus:border-purple-500"
          />
        </div>
        <div>
          <label className="block text-sm text-slate-400 mb-1">Commit SHA (optional)</label>
          <input
            type="text"
            value={sha}
            onChange={(e) => setSha(e.target.value)}
            placeholder="abc123..."
            className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2
                       text-white focus:outline-none focus:border-purple-500 font-mono"
          />
        </div>
        <div className="flex justify-end gap-3 pt-2">
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 text-slate-400 hover:text-white transition-colors"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={loading || !repo}
            className="px-4 py-2 bg-purple-500 hover:bg-purple-600 disabled:opacity-50
                       disabled:cursor-not-allowed text-white rounded-lg transition-colors
                       flex items-center gap-2"
          >
            {loading && <Loader2 className="w-4 h-4 animate-spin" />}
            Trigger
          </button>
        </div>
      </form>
    </Modal>
  );
}

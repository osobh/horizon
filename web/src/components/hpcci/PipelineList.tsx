import { useEffect, useState } from 'react';
import {
  CheckCircle,
  XCircle,
  Clock,
  Loader2,
  GitBranch,
  RotateCcw,
  XOctagon,
  Filter,
  ChevronRight,
} from 'lucide-react';
import { useHpcCiStore, PipelineStatus, PipelineSummary } from '../../stores/hpcciStore';

interface PipelineListProps {
  onSelectPipeline: (id: string) => void;
}

const statusColors: Record<PipelineStatus, string> = {
  queued: 'text-slate-400',
  running: 'text-blue-400',
  success: 'text-green-400',
  failed: 'text-red-400',
  cancelled: 'text-slate-500',
  timeout: 'text-orange-400',
  skipped: 'text-slate-500',
};

const statusIcons: Record<PipelineStatus, React.ReactNode> = {
  queued: <Clock className="w-4 h-4" />,
  running: <Loader2 className="w-4 h-4 animate-spin" />,
  success: <CheckCircle className="w-4 h-4" />,
  failed: <XCircle className="w-4 h-4" />,
  cancelled: <XOctagon className="w-4 h-4" />,
  timeout: <Clock className="w-4 h-4" />,
  skipped: <Clock className="w-4 h-4" />,
};

function formatDuration(ms: number | null): string {
  if (ms === null) return '-';
  const seconds = Math.floor(ms / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);

  if (hours > 0) {
    return `${hours}h ${minutes % 60}m`;
  }
  if (minutes > 0) {
    return `${minutes}m ${seconds % 60}s`;
  }
  return `${seconds}s`;
}

function formatTimeAgo(ms: number): string {
  const now = Date.now();
  const diff = now - ms;
  const seconds = Math.floor(diff / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);

  if (days > 0) return `${days}d ago`;
  if (hours > 0) return `${hours}h ago`;
  if (minutes > 0) return `${minutes}m ago`;
  return 'just now';
}

export function PipelineList({ onSelectPipeline }: PipelineListProps) {
  const { pipelines, loading, fetchPipelines, cancelPipeline, retryPipeline } = useHpcCiStore();
  const [statusFilter, setStatusFilter] = useState<PipelineStatus | 'all'>('all');
  const [repoFilter, setRepoFilter] = useState('');

  useEffect(() => {
    fetchPipelines();
    const interval = setInterval(fetchPipelines, 10000);
    return () => clearInterval(interval);
  }, [fetchPipelines]);

  const filteredPipelines = pipelines.filter((p) => {
    if (statusFilter !== 'all' && p.status !== statusFilter) return false;
    if (repoFilter && !p.repo.toLowerCase().includes(repoFilter.toLowerCase())) return false;
    return true;
  });

  const handleCancel = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    await cancelPipeline(id);
  };

  const handleRetry = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    await retryPipeline(id);
  };

  return (
    <div className="space-y-4">
      {/* Filters */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          <Filter className="w-4 h-4 text-slate-400" />
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value as PipelineStatus | 'all')}
            className="bg-slate-800 border border-slate-700 rounded px-2 py-1.5 text-sm text-slate-200"
          >
            <option value="all">All Status</option>
            <option value="running">Running</option>
            <option value="queued">Queued</option>
            <option value="success">Success</option>
            <option value="failed">Failed</option>
            <option value="cancelled">Cancelled</option>
          </select>
        </div>
        <input
          type="text"
          placeholder="Filter by repository..."
          value={repoFilter}
          onChange={(e) => setRepoFilter(e.target.value)}
          className="bg-slate-800 border border-slate-700 rounded px-3 py-1.5 text-sm text-slate-200 w-64"
        />
      </div>

      {/* Pipeline Table */}
      <div className="bg-slate-800 border border-slate-700 rounded-lg overflow-hidden">
        <table className="w-full">
          <thead>
            <tr className="border-b border-slate-700 text-left text-sm text-slate-400">
              <th className="px-4 py-3">Status</th>
              <th className="px-4 py-3">Pipeline</th>
              <th className="px-4 py-3">Branch</th>
              <th className="px-4 py-3">Progress</th>
              <th className="px-4 py-3">Duration</th>
              <th className="px-4 py-3">Started</th>
              <th className="px-4 py-3">Actions</th>
            </tr>
          </thead>
          <tbody>
            {loading && pipelines.length === 0 ? (
              <tr>
                <td colSpan={7} className="px-4 py-8 text-center text-slate-400">
                  <Loader2 className="w-6 h-6 animate-spin mx-auto mb-2" />
                  Loading pipelines...
                </td>
              </tr>
            ) : filteredPipelines.length === 0 ? (
              <tr>
                <td colSpan={7} className="px-4 py-8 text-center text-slate-400">
                  No pipelines found
                </td>
              </tr>
            ) : (
              filteredPipelines.map((pipeline) => (
                <PipelineRow
                  key={pipeline.id}
                  pipeline={pipeline}
                  onSelect={() => onSelectPipeline(pipeline.id)}
                  onCancel={(e) => handleCancel(e, pipeline.id)}
                  onRetry={(e) => handleRetry(e, pipeline.id)}
                />
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

interface PipelineRowProps {
  pipeline: PipelineSummary;
  onSelect: () => void;
  onCancel: (e: React.MouseEvent) => void;
  onRetry: (e: React.MouseEvent) => void;
}

function PipelineRow({ pipeline, onSelect, onCancel, onRetry }: PipelineRowProps) {
  const canCancel = pipeline.status === 'running' || pipeline.status === 'queued';
  const canRetry = pipeline.status === 'failed' || pipeline.status === 'cancelled';

  return (
    <tr
      onClick={onSelect}
      className="border-b border-slate-700/50 hover:bg-slate-700/30 cursor-pointer transition-colors"
    >
      <td className="px-4 py-3">
        <div className={`flex items-center gap-2 ${statusColors[pipeline.status]}`}>
          {statusIcons[pipeline.status]}
          <span className="text-sm capitalize">{pipeline.status}</span>
        </div>
      </td>
      <td className="px-4 py-3">
        <div className="flex items-center gap-2">
          <span className="text-sm text-white font-medium">{pipeline.id}</span>
          <span className="text-xs text-slate-400 font-mono">{pipeline.sha_short}</span>
        </div>
        <div className="text-xs text-slate-400">{pipeline.repo}</div>
      </td>
      <td className="px-4 py-3">
        <div className="flex items-center gap-1 text-sm text-slate-300">
          <GitBranch className="w-3 h-3" />
          {pipeline.branch}
        </div>
      </td>
      <td className="px-4 py-3">
        <span className="text-sm text-slate-300">{pipeline.stages_progress}</span>
      </td>
      <td className="px-4 py-3">
        <span className="text-sm text-slate-300">{formatDuration(pipeline.duration_ms)}</span>
      </td>
      <td className="px-4 py-3">
        <span className="text-sm text-slate-400">{formatTimeAgo(pipeline.started_at_ms)}</span>
      </td>
      <td className="px-4 py-3">
        <div className="flex items-center gap-1">
          {canCancel && (
            <button
              onClick={onCancel}
              className="p-1.5 hover:bg-slate-600 rounded transition-colors"
              title="Cancel"
            >
              <XOctagon className="w-4 h-4 text-red-400" />
            </button>
          )}
          {canRetry && (
            <button
              onClick={onRetry}
              className="p-1.5 hover:bg-slate-600 rounded transition-colors"
              title="Retry"
            >
              <RotateCcw className="w-4 h-4 text-blue-400" />
            </button>
          )}
          <button
            onClick={(e) => {
              e.stopPropagation();
              onSelect();
            }}
            className="p-1.5 hover:bg-slate-600 rounded transition-colors"
            title="View Details"
          >
            <ChevronRight className="w-4 h-4 text-slate-400" />
          </button>
        </div>
      </td>
    </tr>
  );
}

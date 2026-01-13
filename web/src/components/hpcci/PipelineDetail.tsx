import { useEffect } from 'react';
import {
  CheckCircle,
  XCircle,
  Clock,
  Loader2,
  GitBranch,
  GitCommit,
  User,
  ChevronDown,
  ChevronRight,
  ArrowLeft,
  Terminal,
  Server,
} from 'lucide-react';
import { useHpcCiStore, PipelineStatus, StageInfo, JobInfo } from '../../stores/hpcciStore';

interface PipelineDetailProps {
  pipelineId: string;
  onBack: () => void;
}

const statusColors: Record<PipelineStatus, string> = {
  queued: 'bg-slate-500',
  running: 'bg-blue-500',
  success: 'bg-green-500',
  failed: 'bg-red-500',
  cancelled: 'bg-slate-600',
  timeout: 'bg-orange-500',
  skipped: 'bg-slate-600',
};

const statusTextColors: Record<PipelineStatus, string> = {
  queued: 'text-slate-400',
  running: 'text-blue-400',
  success: 'text-green-400',
  failed: 'text-red-400',
  cancelled: 'text-slate-500',
  timeout: 'text-orange-400',
  skipped: 'text-slate-500',
};

function formatDuration(ms: number | null): string {
  if (ms === null) return '-';
  const seconds = Math.floor(ms / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);

  if (hours > 0) {
    return `${hours}h ${minutes % 60}m ${seconds % 60}s`;
  }
  if (minutes > 0) {
    return `${minutes}m ${seconds % 60}s`;
  }
  return `${seconds}s`;
}

function formatTimestamp(ms: number): string {
  return new Date(ms).toLocaleString();
}

export function PipelineDetail({ pipelineId, onBack }: PipelineDetailProps) {
  const { selectedPipeline, logs, loading, fetchPipelineDetail, fetchPipelineLogs } =
    useHpcCiStore();

  useEffect(() => {
    fetchPipelineDetail(pipelineId);
    fetchPipelineLogs(pipelineId);
    const interval = setInterval(() => {
      fetchPipelineDetail(pipelineId);
    }, 5000);
    return () => clearInterval(interval);
  }, [pipelineId, fetchPipelineDetail, fetchPipelineLogs]);

  if (loading && !selectedPipeline) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-8 h-8 animate-spin text-blue-400" />
      </div>
    );
  }

  if (!selectedPipeline) {
    return (
      <div className="text-center py-8 text-slate-400">Pipeline not found</div>
    );
  }

  const pipeline = selectedPipeline;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <button
            onClick={onBack}
            className="flex items-center gap-1 text-sm text-slate-400 hover:text-white mb-2 transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Pipelines
          </button>
          <h2 className="text-xl font-semibold text-white flex items-center gap-3">
            Pipeline {pipeline.id}
            <span
              className={`px-2 py-0.5 rounded text-sm ${statusColors[pipeline.status]} text-white`}
            >
              {pipeline.status}
            </span>
          </h2>
          <p className="text-slate-400 mt-1">{pipeline.repo}</p>
        </div>
      </div>

      {/* Info Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <InfoCard icon={GitBranch} label="Branch" value={pipeline.branch} />
        <InfoCard icon={GitCommit} label="Commit" value={pipeline.sha.substring(0, 12)} mono />
        <InfoCard icon={Clock} label="Duration" value={formatDuration(pipeline.duration_ms)} />
        <InfoCard icon={User} label="Triggered by" value={pipeline.trigger.user || 'System'} />
      </div>

      {/* Trigger Info */}
      {pipeline.trigger.commit_message && (
        <div className="bg-slate-800 border border-slate-700 rounded-lg p-4">
          <div className="text-sm text-slate-400 mb-1">Commit Message</div>
          <div className="text-white">{pipeline.trigger.commit_message}</div>
          <div className="text-xs text-slate-500 mt-2">
            Trigger: {pipeline.trigger.trigger_type} | Started: {formatTimestamp(pipeline.started_at_ms)}
            {pipeline.finished_at_ms && ` | Finished: ${formatTimestamp(pipeline.finished_at_ms)}`}
          </div>
        </div>
      )}

      {/* Stages */}
      <div className="space-y-3">
        <h3 className="text-lg font-medium text-white">Stages</h3>
        {pipeline.stages.map((stage, index) => (
          <StageCard key={stage.name} stage={stage} index={index} />
        ))}
      </div>

      {/* Logs */}
      <div className="space-y-3">
        <h3 className="text-lg font-medium text-white flex items-center gap-2">
          <Terminal className="w-5 h-5" />
          Logs
        </h3>
        <div className="bg-slate-900 border border-slate-700 rounded-lg p-4 font-mono text-sm max-h-96 overflow-auto">
          {logs.length === 0 ? (
            <div className="text-slate-500">No logs available</div>
          ) : (
            logs.map((log, i) => (
              <LogLine key={i} log={log} />
            ))
          )}
        </div>
      </div>
    </div>
  );
}

interface InfoCardProps {
  icon: React.ElementType;
  label: string;
  value: string;
  mono?: boolean;
}

function InfoCard({ icon: Icon, label, value, mono }: InfoCardProps) {
  return (
    <div className="bg-slate-800 border border-slate-700 rounded-lg p-3">
      <div className="flex items-center gap-2 text-slate-400 text-sm mb-1">
        <Icon className="w-4 h-4" />
        {label}
      </div>
      <div className={`text-white ${mono ? 'font-mono' : ''}`}>{value}</div>
    </div>
  );
}

interface StageCardProps {
  stage: StageInfo;
  index: number;
}

function StageCard({ stage, index }: StageCardProps) {
  const isExpanded = stage.status === 'running' || stage.status === 'failed';

  return (
    <div className="bg-slate-800 border border-slate-700 rounded-lg overflow-hidden">
      <div className="flex items-center justify-between px-4 py-3">
        <div className="flex items-center gap-3">
          <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-medium ${statusColors[stage.status]} text-white`}>
            {index + 1}
          </div>
          <span className="text-white font-medium">{stage.name}</span>
          <span className={`text-sm ${statusTextColors[stage.status]}`}>
            {stage.status}
          </span>
        </div>
        <div className="flex items-center gap-4">
          <span className="text-sm text-slate-400">{formatDuration(stage.duration_ms)}</span>
          <span className="text-sm text-slate-500">{stage.jobs.length} jobs</span>
          {isExpanded ? (
            <ChevronDown className="w-4 h-4 text-slate-400" />
          ) : (
            <ChevronRight className="w-4 h-4 text-slate-400" />
          )}
        </div>
      </div>

      {isExpanded && (
        <div className="border-t border-slate-700 px-4 py-2 space-y-2">
          {stage.jobs.map((job) => (
            <JobRow key={job.name} job={job} />
          ))}
        </div>
      )}
    </div>
  );
}

interface JobRowProps {
  job: JobInfo;
}

function JobRow({ job }: JobRowProps) {
  return (
    <div className="flex items-center justify-between py-2 px-2 rounded hover:bg-slate-700/30">
      <div className="flex items-center gap-3">
        <StatusIcon status={job.status} />
        <span className="text-sm text-slate-200">{job.name}</span>
      </div>
      <div className="flex items-center gap-4">
        {job.agent_id && (
          <span className="flex items-center gap-1 text-xs text-slate-500">
            <Server className="w-3 h-3" />
            {job.agent_id}
          </span>
        )}
        <span className="text-sm text-slate-400">{formatDuration(job.duration_ms)}</span>
      </div>
    </div>
  );
}

function StatusIcon({ status }: { status: PipelineStatus }) {
  switch (status) {
    case 'success':
      return <CheckCircle className="w-4 h-4 text-green-400" />;
    case 'failed':
      return <XCircle className="w-4 h-4 text-red-400" />;
    case 'running':
      return <Loader2 className="w-4 h-4 text-blue-400 animate-spin" />;
    default:
      return <Clock className="w-4 h-4 text-slate-400" />;
  }
}

interface LogLineProps {
  log: { timestamp_ms: number; stage: string; job: string; level: string; content: string };
}

function LogLine({ log }: LogLineProps) {
  const levelColors: Record<string, string> = {
    info: 'text-slate-300',
    warn: 'text-yellow-400',
    error: 'text-red-400',
    debug: 'text-slate-500',
  };

  const time = new Date(log.timestamp_ms).toLocaleTimeString();

  return (
    <div className="flex gap-2 py-0.5">
      <span className="text-slate-600">{time}</span>
      <span className="text-slate-500">[{log.stage}/{log.job}]</span>
      <span className={levelColors[log.level] || 'text-slate-300'}>{log.content}</span>
    </div>
  );
}

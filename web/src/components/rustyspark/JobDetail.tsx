/**
 * RustySpark Job Detail
 *
 * Shows detailed information about a Spark job including stages and tasks.
 */

import React from 'react';
import {
  ArrowLeft,
  Clock,
  Database,
  Shuffle,
  Cpu,
  CheckCircle,
  XCircle,
  Play,
  Pause,
  Ban,
  Loader2,
} from 'lucide-react';
import { useRustySparkStore, SparkJob, SparkStage, SparkStageStatus, SparkTaskStatus } from '../../stores/rustysparkStore';

// Format bytes to human readable
function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
}

// Format duration in ms
function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  if (ms < 3600000) return `${Math.floor(ms / 60000)}m ${Math.floor((ms % 60000) / 1000)}s`;
  return `${Math.floor(ms / 3600000)}h ${Math.floor((ms % 3600000) / 60000)}m`;
}

// Stage status badge
function StageStatusBadge({ status }: { status: SparkStageStatus }) {
  const configs: Record<SparkStageStatus, { icon: React.ReactNode; color: string; bg: string }> = {
    pending: { icon: <Clock size={12} />, color: 'text-yellow-400', bg: 'bg-yellow-400/10' },
    active: { icon: <Play size={12} />, color: 'text-blue-400', bg: 'bg-blue-400/10' },
    complete: { icon: <CheckCircle size={12} />, color: 'text-green-400', bg: 'bg-green-400/10' },
    failed: { icon: <XCircle size={12} />, color: 'text-red-400', bg: 'bg-red-400/10' },
    skipped: { icon: <Pause size={12} />, color: 'text-gray-400', bg: 'bg-gray-400/10' },
  };

  const config = configs[status];
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium ${config.color} ${config.bg}`}>
      {config.icon}
      {status}
    </span>
  );
}

// Task status badge
function TaskStatusBadge({ status }: { status: SparkTaskStatus }) {
  const configs: Record<SparkTaskStatus, { icon: React.ReactNode; color: string; bg: string }> = {
    running: { icon: <Play size={12} />, color: 'text-blue-400', bg: 'bg-blue-400/10' },
    success: { icon: <CheckCircle size={12} />, color: 'text-green-400', bg: 'bg-green-400/10' },
    failed: { icon: <XCircle size={12} />, color: 'text-red-400', bg: 'bg-red-400/10' },
    killed: { icon: <Ban size={12} />, color: 'text-gray-400', bg: 'bg-gray-400/10' },
  };

  const config = configs[status];
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium ${config.color} ${config.bg}`}>
      {config.icon}
      {status}
    </span>
  );
}

// Stat card component
function StatCard({ icon, label, value, subValue }: { icon: React.ReactNode; label: string; value: string; subValue?: string }) {
  return (
    <div className="bg-[#1e293b] rounded-lg p-3">
      <div className="flex items-center gap-2 text-gray-400 text-xs mb-1">
        {icon}
        {label}
      </div>
      <div className="text-lg font-semibold text-white">{value}</div>
      {subValue && <div className="text-xs text-gray-500">{subValue}</div>}
    </div>
  );
}

interface JobDetailProps {
  job: SparkJob;
  onBack: () => void;
}

export function JobDetail({ job, onBack }: JobDetailProps) {
  const { stages, tasks, fetchJobStages, fetchStageTasks, isLoading, cancelJob } = useRustySparkStore();
  const [selectedStage, setSelectedStage] = React.useState<SparkStage | null>(null);

  React.useEffect(() => {
    fetchJobStages(job.id);
  }, [job.id, fetchJobStages]);

  const handleStageClick = (stage: SparkStage) => {
    setSelectedStage(stage);
    fetchStageTasks(stage.stage_id);
  };

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <button
            onClick={onBack}
            className="p-2 text-gray-400 hover:text-white hover:bg-[#1e293b] rounded-lg transition-colors"
          >
            <ArrowLeft size={18} />
          </button>
          <div>
            <h2 className="text-lg font-semibold text-white">{job.name}</h2>
            <p className="text-xs text-gray-500">{job.id}</p>
          </div>
        </div>
        {(job.status === 'running' || job.status === 'pending') && (
          <button
            onClick={() => cancelJob(job.id)}
            className="flex items-center gap-2 px-3 py-1.5 bg-red-500/10 text-red-400 hover:bg-red-500/20 rounded-lg text-sm transition-colors"
          >
            <Ban size={14} />
            Cancel Job
          </button>
        )}
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-4 gap-3">
        <StatCard
          icon={<Clock size={14} />}
          label="Duration"
          value={job.started_at ? formatDuration(
            (job.completed_at ? new Date(job.completed_at).getTime() : Date.now()) -
            new Date(job.started_at).getTime()
          ) : '-'}
          subValue={job.started_at ? `Started ${new Date(job.started_at).toLocaleTimeString()}` : 'Not started'}
        />
        <StatCard
          icon={<Database size={14} />}
          label="Input/Output"
          value={formatBytes(job.input_bytes)}
          subValue={`Out: ${formatBytes(job.output_bytes)}`}
        />
        <StatCard
          icon={<Shuffle size={14} />}
          label="Shuffle"
          value={formatBytes(job.shuffle_read_bytes + job.shuffle_write_bytes)}
          subValue={`R: ${formatBytes(job.shuffle_read_bytes)} / W: ${formatBytes(job.shuffle_write_bytes)}`}
        />
        <StatCard
          icon={<Cpu size={14} />}
          label="Progress"
          value={`${job.completed_stages}/${job.num_stages} stages`}
          subValue={`${job.completed_tasks}/${job.num_tasks} tasks`}
        />
      </div>

      {/* Stages Table */}
      <div className="bg-[#0f1729] rounded-lg border border-[#1e293b]">
        <div className="px-4 py-3 border-b border-[#1e293b]">
          <h3 className="text-sm font-medium text-white">Stages</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="text-xs text-gray-400 border-b border-[#1e293b]">
                <th className="text-left px-4 py-2 font-medium">Stage</th>
                <th className="text-left px-4 py-2 font-medium">Status</th>
                <th className="text-left px-4 py-2 font-medium">Tasks</th>
                <th className="text-right px-4 py-2 font-medium">Input</th>
                <th className="text-right px-4 py-2 font-medium">Output</th>
                <th className="text-right px-4 py-2 font-medium">Shuffle R/W</th>
                <th className="text-right px-4 py-2 font-medium">Duration</th>
              </tr>
            </thead>
            <tbody>
              {isLoading && stages.length === 0 ? (
                <tr>
                  <td colSpan={7} className="px-4 py-6 text-center">
                    <Loader2 className="h-5 w-5 animate-spin text-[#00d4ff] mx-auto" />
                  </td>
                </tr>
              ) : stages.length === 0 ? (
                <tr>
                  <td colSpan={7} className="px-4 py-6 text-center text-gray-500 text-sm">
                    No stages
                  </td>
                </tr>
              ) : (
                stages.map((stage) => (
                  <tr
                    key={stage.stage_id}
                    className={`border-b border-[#1e293b] cursor-pointer transition-colors ${
                      selectedStage?.stage_id === stage.stage_id
                        ? 'bg-[#00d4ff]/10'
                        : 'hover:bg-[#1e293b]/50'
                    }`}
                    onClick={() => handleStageClick(stage)}
                  >
                    <td className="px-4 py-3">
                      <div className="flex flex-col">
                        <span className="text-sm font-medium text-white">Stage {stage.stage_id}</span>
                        <span className="text-xs text-gray-500 truncate max-w-[200px]">{stage.name}</span>
                      </div>
                    </td>
                    <td className="px-4 py-3">
                      <StageStatusBadge status={stage.status} />
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-1 text-sm">
                        <span className="text-green-400">{stage.completed_tasks}</span>
                        <span className="text-gray-500">/</span>
                        <span className="text-white">{stage.num_tasks}</span>
                        {stage.failed_tasks > 0 && (
                          <span className="text-red-400 ml-1">({stage.failed_tasks} failed)</span>
                        )}
                      </div>
                    </td>
                    <td className="px-4 py-3 text-right text-sm text-gray-300">
                      {formatBytes(stage.input_bytes)}
                    </td>
                    <td className="px-4 py-3 text-right text-sm text-gray-300">
                      {formatBytes(stage.output_bytes)}
                    </td>
                    <td className="px-4 py-3 text-right text-xs text-gray-400">
                      {formatBytes(stage.shuffle_read_bytes)} / {formatBytes(stage.shuffle_write_bytes)}
                    </td>
                    <td className="px-4 py-3 text-right text-sm text-gray-300">
                      {formatDuration(stage.executor_run_time_ms)}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Tasks Table (shown when stage is selected) */}
      {selectedStage && (
        <div className="bg-[#0f1729] rounded-lg border border-[#1e293b]">
          <div className="px-4 py-3 border-b border-[#1e293b] flex items-center justify-between">
            <h3 className="text-sm font-medium text-white">
              Tasks for Stage {selectedStage.stage_id}
            </h3>
            <button
              onClick={() => setSelectedStage(null)}
              className="text-xs text-gray-400 hover:text-white transition-colors"
            >
              Close
            </button>
          </div>
          <div className="overflow-x-auto max-h-64 overflow-y-auto">
            <table className="w-full">
              <thead className="sticky top-0 bg-[#0f1729]">
                <tr className="text-xs text-gray-400 border-b border-[#1e293b]">
                  <th className="text-left px-4 py-2 font-medium">Task ID</th>
                  <th className="text-left px-4 py-2 font-medium">Status</th>
                  <th className="text-left px-4 py-2 font-medium">Executor</th>
                  <th className="text-left px-4 py-2 font-medium">Host</th>
                  <th className="text-right px-4 py-2 font-medium">Duration</th>
                  <th className="text-right px-4 py-2 font-medium">Input</th>
                  <th className="text-right px-4 py-2 font-medium">Output</th>
                </tr>
              </thead>
              <tbody>
                {tasks.length === 0 ? (
                  <tr>
                    <td colSpan={7} className="px-4 py-4 text-center text-gray-500 text-sm">
                      {isLoading ? (
                        <Loader2 className="h-4 w-4 animate-spin text-[#00d4ff] mx-auto" />
                      ) : (
                        'No tasks'
                      )}
                    </td>
                  </tr>
                ) : (
                  tasks.map((task) => (
                    <tr key={task.task_id} className="border-b border-[#1e293b] hover:bg-[#1e293b]/50">
                      <td className="px-4 py-2 text-sm text-white">{task.task_id}</td>
                      <td className="px-4 py-2">
                        <TaskStatusBadge status={task.status} />
                      </td>
                      <td className="px-4 py-2 text-sm text-gray-300">{task.executor_id}</td>
                      <td className="px-4 py-2 text-xs text-gray-400">{task.host}</td>
                      <td className="px-4 py-2 text-right text-sm text-gray-300">
                        {task.duration_ms !== null ? formatDuration(task.duration_ms) : '-'}
                      </td>
                      <td className="px-4 py-2 text-right text-xs text-gray-400">
                        {formatBytes(task.input_bytes)}
                      </td>
                      <td className="px-4 py-2 text-right text-xs text-gray-400">
                        {formatBytes(task.output_bytes)}
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

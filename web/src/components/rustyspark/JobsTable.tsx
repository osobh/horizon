/**
 * RustySpark Jobs Table
 *
 * Displays Spark jobs with status, progress, and actions.
 */

import React from 'react';
import {
  Play,
  CheckCircle,
  XCircle,
  Clock,
  Ban,
  ChevronRight,
  Loader2,
  Filter,
} from 'lucide-react';
import { useRustySparkStore, SparkJob, SparkJobStatus } from '../../stores/rustysparkStore';

// Format bytes to human readable
function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
}

// Format duration
function formatDuration(startedAt: string | null, completedAt: string | null): string {
  if (!startedAt) return '-';
  const start = new Date(startedAt).getTime();
  const end = completedAt ? new Date(completedAt).getTime() : Date.now();
  const durationMs = end - start;

  if (durationMs < 1000) return `${durationMs}ms`;
  if (durationMs < 60000) return `${(durationMs / 1000).toFixed(1)}s`;
  if (durationMs < 3600000) return `${Math.floor(durationMs / 60000)}m ${Math.floor((durationMs % 60000) / 1000)}s`;
  return `${Math.floor(durationMs / 3600000)}h ${Math.floor((durationMs % 3600000) / 60000)}m`;
}

// Status badge component
function StatusBadge({ status }: { status: SparkJobStatus }) {
  const configs: Record<SparkJobStatus, { icon: React.ReactNode; color: string; bg: string }> = {
    pending: { icon: <Clock size={14} />, color: 'text-yellow-400', bg: 'bg-yellow-400/10' },
    running: { icon: <Play size={14} />, color: 'text-blue-400', bg: 'bg-blue-400/10' },
    succeeded: { icon: <CheckCircle size={14} />, color: 'text-green-400', bg: 'bg-green-400/10' },
    failed: { icon: <XCircle size={14} />, color: 'text-red-400', bg: 'bg-red-400/10' },
    cancelled: { icon: <Ban size={14} />, color: 'text-gray-400', bg: 'bg-gray-400/10' },
  };

  const config = configs[status];
  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium ${config.color} ${config.bg}`}>
      {config.icon}
      {status}
    </span>
  );
}

// Progress bar component
function ProgressBar({ completed, total }: { completed: number; total: number }) {
  const percent = total > 0 ? Math.round((completed / total) * 100) : 0;
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 h-1.5 bg-[#1e293b] rounded-full overflow-hidden">
        <div
          className="h-full bg-[#00d4ff] transition-all duration-300"
          style={{ width: `${percent}%` }}
        />
      </div>
      <span className="text-xs text-gray-400 w-12 text-right">{percent}%</span>
    </div>
  );
}

interface JobsTableProps {
  onSelectJob?: (job: SparkJob) => void;
}

export function JobsTable({ onSelectJob }: JobsTableProps) {
  const {
    jobs,
    isLoading,
    jobStatusFilter,
    setJobStatusFilter,
    cancelJob,
    fetchJobs,
  } = useRustySparkStore();

  React.useEffect(() => {
    fetchJobs();
    const interval = setInterval(() => fetchJobs(), 10000);
    return () => clearInterval(interval);
  }, [fetchJobs]);

  const statusOptions: (SparkJobStatus | null)[] = [null, 'running', 'pending', 'succeeded', 'failed', 'cancelled'];

  return (
    <div className="bg-[#0f1729] rounded-lg border border-[#1e293b]">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-[#1e293b]">
        <h3 className="text-sm font-medium text-white">Spark Jobs</h3>
        <div className="flex items-center gap-2">
          <Filter size={14} className="text-gray-400" />
          <select
            value={jobStatusFilter || ''}
            onChange={(e) => setJobStatusFilter(e.target.value as SparkJobStatus | null || null)}
            className="bg-[#1e293b] text-gray-300 text-xs rounded px-2 py-1 border border-[#334155] focus:outline-none focus:ring-1 focus:ring-[#00d4ff]"
          >
            {statusOptions.map((status) => (
              <option key={status || 'all'} value={status || ''}>
                {status ? status.charAt(0).toUpperCase() + status.slice(1) : 'All Status'}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="text-xs text-gray-400 border-b border-[#1e293b]">
              <th className="text-left px-4 py-2 font-medium">Job</th>
              <th className="text-left px-4 py-2 font-medium">Status</th>
              <th className="text-left px-4 py-2 font-medium">Progress</th>
              <th className="text-left px-4 py-2 font-medium">Duration</th>
              <th className="text-right px-4 py-2 font-medium">I/O</th>
              <th className="text-right px-4 py-2 font-medium">Actions</th>
            </tr>
          </thead>
          <tbody>
            {isLoading && jobs.length === 0 ? (
              <tr>
                <td colSpan={6} className="px-4 py-8 text-center">
                  <Loader2 className="h-5 w-5 animate-spin text-[#00d4ff] mx-auto" />
                </td>
              </tr>
            ) : jobs.length === 0 ? (
              <tr>
                <td colSpan={6} className="px-4 py-8 text-center text-gray-500 text-sm">
                  No jobs found
                </td>
              </tr>
            ) : (
              jobs.map((job) => (
                <tr
                  key={job.id}
                  className="border-b border-[#1e293b] hover:bg-[#1e293b]/50 cursor-pointer transition-colors"
                  onClick={() => onSelectJob?.(job)}
                >
                  <td className="px-4 py-3">
                    <div className="flex flex-col">
                      <span className="text-sm font-medium text-white">{job.name}</span>
                      <span className="text-xs text-gray-500">{job.id}</span>
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <StatusBadge status={job.status} />
                  </td>
                  <td className="px-4 py-3 w-48">
                    <div className="space-y-1">
                      <div className="flex items-center justify-between text-xs text-gray-400">
                        <span>Stages</span>
                        <span>{job.completed_stages}/{job.num_stages}</span>
                      </div>
                      <ProgressBar completed={job.completed_stages} total={job.num_stages} />
                    </div>
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-300">
                    {formatDuration(job.started_at, job.completed_at)}
                  </td>
                  <td className="px-4 py-3 text-right">
                    <div className="text-xs text-gray-400">
                      <div>In: {formatBytes(job.input_bytes)}</div>
                      <div>Out: {formatBytes(job.output_bytes)}</div>
                    </div>
                  </td>
                  <td className="px-4 py-3 text-right">
                    <div className="flex items-center justify-end gap-2">
                      {(job.status === 'running' || job.status === 'pending') && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            cancelJob(job.id);
                          }}
                          className="p-1 text-red-400 hover:bg-red-400/10 rounded transition-colors"
                          title="Cancel job"
                        >
                          <Ban size={14} />
                        </button>
                      )}
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          onSelectJob?.(job);
                        }}
                        className="p-1 text-gray-400 hover:text-white transition-colors"
                        title="View details"
                      >
                        <ChevronRight size={14} />
                      </button>
                    </div>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

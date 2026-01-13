/**
 * SchedulerView - SLAI GPU Scheduler Dashboard
 *
 * Provides a unified view of GPU cluster scheduling:
 * - GPU inventory with vendor/memory details
 * - Job queue management (submit, cancel, view status)
 * - Tenant fair-share allocation
 * - Real-time scheduler statistics
 */

import { useEffect, useCallback } from 'react';
import {
  Cpu,
  Users,
  Layers,
  Play,
  Trash2,
  Plus,
  RefreshCw,
  CheckCircle,
  Clock,
} from 'lucide-react';
import { useSchedulerStore, GpuInfo, JobInfo, TenantInfo, FairShareInfo } from '../../stores/schedulerStore';

const VENDOR_COLORS: Record<string, string> = {
  nvidia: 'text-green-400 bg-green-500/10 border-green-500/30',
  amd: 'text-red-400 bg-red-500/10 border-red-500/30',
  apple: 'text-blue-400 bg-blue-500/10 border-blue-500/30',
  intel: 'text-cyan-400 bg-cyan-500/10 border-cyan-500/30',
};

const STATUS_COLORS: Record<string, string> = {
  queued: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
  running: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
  completed: 'bg-green-500/20 text-green-400 border-green-500/30',
  failed: 'bg-red-500/20 text-red-400 border-red-500/30',
  cancelled: 'bg-slate-500/20 text-slate-400 border-slate-500/30',
};

const PRIORITY_COLORS: Record<string, string> = {
  low: 'bg-slate-500/20 text-slate-400',
  normal: 'bg-blue-500/20 text-blue-400',
  high: 'bg-orange-500/20 text-orange-400',
  critical: 'bg-red-500/20 text-red-400',
};

function formatBytes(bytes: number): string {
  const gb = bytes / (1024 * 1024 * 1024);
  return `${gb.toFixed(0)} GB`;
}

function formatTimestamp(ts: number): string {
  return new Date(ts * 1000).toLocaleTimeString();
}

export default function SchedulerView() {
  const {
    stats,
    gpus,
    jobs,
    tenants,
    fairShare,
    loading,
    error,
    fetchStats,
    fetchGpus,
    fetchJobs,
    fetchTenants,
    fetchFairShare,
    submitJob,
    cancelJob,
  } = useSchedulerStore();

  useEffect(() => {
    fetchStats();
    fetchGpus();
    fetchJobs();
    fetchTenants();
    fetchFairShare();
    const interval = setInterval(() => {
      fetchStats();
      fetchJobs();
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleSubmitDemo = useCallback(() => {
    if (!tenants.length) return;
    submitJob({
      name: `demo-job-${Date.now().toString(36)}`,
      tenant_id: tenants[0].id,
      gpus_requested: 1,
      priority: 'normal',
    });
  }, [submitJob, tenants]);

  const handleCancelJob = useCallback((jobId: string) => {
    cancelJob(jobId);
  }, [cancelJob]);

  const handleRefresh = useCallback(() => {
    fetchStats();
    fetchGpus();
    fetchJobs();
    fetchTenants();
    fetchFairShare();
  }, [fetchStats, fetchGpus, fetchJobs, fetchTenants, fetchFairShare]);

  return (
    <div className="h-full overflow-auto bg-slate-900 text-white">
      {/* Header */}
      <div className="h-16 bg-slate-800 border-b border-slate-700 flex items-center px-6">
        <div className="flex items-center gap-3">
          <Layers className="h-6 w-6 text-purple-400" />
          <h1 className="text-xl font-semibold">GPU Scheduler</h1>
          <span className="bg-purple-500/20 text-purple-400 border border-purple-500/30 px-2 py-0.5 rounded text-xs">
            SLAI
          </span>
        </div>
        <div className="flex-1" />
        <div className="flex items-center gap-2">
          <button
            onClick={handleSubmitDemo}
            disabled={!tenants.length}
            className="px-3 py-1.5 bg-purple-600 hover:bg-purple-500 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg text-sm transition-colors flex items-center gap-1"
          >
            <Plus className="h-3 w-3" />
            Demo Job
          </button>
          <button
            onClick={handleRefresh}
            className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
          >
            <RefreshCw className={`w-5 h-5 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      <div className="p-6 space-y-6">
        {error && (
          <div className="p-4 bg-red-500/20 border border-red-500 rounded-lg text-red-400">{error}</div>
        )}

        {/* Stats Overview */}
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
          <StatCard
            icon={Cpu}
            label="Total GPUs"
            value={stats?.total_gpus ?? 0}
            color="text-green-400"
          />
          <StatCard
            icon={Cpu}
            label="Available"
            value={stats?.available_gpus ?? 0}
            color="text-blue-400"
          />
          <StatCard
            icon={Clock}
            label="Queued"
            value={stats?.queued_jobs ?? 0}
            color="text-yellow-400"
          />
          <StatCard
            icon={Play}
            label="Running"
            value={stats?.running_jobs ?? 0}
            color="text-purple-400"
          />
          <StatCard
            icon={CheckCircle}
            label="Completed"
            value={stats?.completed_jobs ?? 0}
            color="text-emerald-400"
          />
          <StatCard
            icon={Users}
            label="Tenants"
            value={stats?.tenant_count ?? 0}
            color="text-cyan-400"
          />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* GPU Inventory */}
          <div className="bg-slate-800 rounded-lg border border-slate-700">
            <div className="p-4 border-b border-slate-700">
              <h3 className="text-sm font-medium flex items-center gap-2">
                <Cpu className="h-4 w-4 text-green-400" />
                GPU Inventory ({gpus.length})
              </h3>
            </div>
            <div className="p-4 space-y-2">
              {loading && !gpus.length ? (
                <div className="flex items-center justify-center py-8">
                  <RefreshCw className="h-5 w-5 animate-spin text-slate-500" />
                </div>
              ) : gpus.length ? (
                gpus.map((gpu) => (
                  <GpuCard key={gpu.index} gpu={gpu} />
                ))
              ) : (
                <div className="text-center py-4 text-slate-500 text-sm">
                  No GPUs detected
                </div>
              )}
            </div>
          </div>

          {/* Job Queue */}
          <div className="bg-slate-800 rounded-lg border border-slate-700 lg:col-span-2">
            <div className="p-4 border-b border-slate-700">
              <h3 className="text-sm font-medium flex items-center gap-2">
                <Layers className="h-4 w-4 text-purple-400" />
                Job Queue
                <span className="ml-2 bg-slate-700 px-2 py-0.5 rounded text-xs">
                  {(jobs?.queued.length ?? 0) + (jobs?.running.length ?? 0)} active
                </span>
              </h3>
            </div>
            <div className="p-4">
              {loading && !jobs ? (
                <div className="flex items-center justify-center py-8">
                  <RefreshCw className="h-5 w-5 animate-spin text-slate-500" />
                </div>
              ) : (
                <div className="space-y-4">
                  {/* Running Jobs */}
                  {jobs?.running.length ? (
                    <div className="space-y-2">
                      <div className="text-xs text-slate-500 uppercase tracking-wider">
                        Running ({jobs.running.length})
                      </div>
                      {jobs.running.map((job) => (
                        <JobCard key={job.id} job={job} onCancel={handleCancelJob} />
                      ))}
                    </div>
                  ) : null}

                  {/* Queued Jobs */}
                  {jobs?.queued.length ? (
                    <div className="space-y-2">
                      <div className="text-xs text-slate-500 uppercase tracking-wider">
                        Queued ({jobs.queued.length})
                      </div>
                      {jobs.queued.map((job) => (
                        <JobCard key={job.id} job={job} onCancel={handleCancelJob} />
                      ))}
                    </div>
                  ) : null}

                  {/* No Jobs */}
                  {!jobs?.running.length && !jobs?.queued.length && (
                    <div className="text-center py-8 text-slate-500 text-sm">
                      No jobs in queue. Click "Demo Job" to submit one.
                    </div>
                  )}

                  {/* Completed Jobs (last 5) */}
                  {jobs?.completed.length ? (
                    <div className="space-y-2 pt-4 border-t border-slate-700">
                      <div className="text-xs text-slate-500 uppercase tracking-wider">
                        Recently Completed ({jobs.completed.length})
                      </div>
                      {jobs.completed.slice(0, 5).map((job) => (
                        <JobCard key={job.id} job={job} compact />
                      ))}
                    </div>
                  ) : null}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Tenants and Fair-Share */}
        <div className="bg-slate-800 rounded-lg border border-slate-700">
          <div className="p-4 border-b border-slate-700">
            <h3 className="text-sm font-medium flex items-center gap-2">
              <Users className="h-4 w-4 text-cyan-400" />
              Tenant Fair-Share Allocation
            </h3>
          </div>
          <div className="p-4">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {tenants.map((tenant) => {
                const share = fairShare.find(fs => fs.tenant_name === tenant.name);
                return (
                  <TenantCard
                    key={tenant.id}
                    tenant={tenant}
                    fairShare={share}
                  />
                );
              })}
              {!tenants.length && (
                <div className="col-span-full text-center py-4 text-slate-500 text-sm">
                  No tenants registered
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// Sub-components

interface StatCardProps {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: number | string;
  color: string;
}

function StatCard({ icon: Icon, label, value, color }: StatCardProps) {
  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
      <div className="flex items-center gap-3">
        <Icon className={`h-5 w-5 ${color}`} />
        <div>
          <div className="text-2xl font-bold">{value}</div>
          <div className="text-xs text-slate-400">{label}</div>
        </div>
      </div>
    </div>
  );
}

function GpuCard({ gpu }: { gpu: GpuInfo }) {
  const usedPercent = ((gpu.memory_total - gpu.memory_available) / gpu.memory_total) * 100;
  const vendorColor = VENDOR_COLORS[gpu.vendor] ?? VENDOR_COLORS.intel;

  return (
    <div className={`border rounded-lg p-3 ${vendorColor}`}>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className={`px-2 py-0.5 rounded text-xs border ${vendorColor}`}>
            {gpu.vendor.toUpperCase()}
          </span>
          <span className="text-sm font-medium truncate">{gpu.name}</span>
        </div>
        {gpu.is_primary && (
          <span className="bg-slate-700 px-2 py-0.5 rounded text-xs">Primary</span>
        )}
      </div>
      <div className="space-y-1">
        <div className="flex justify-between text-xs text-slate-400">
          <span>Memory</span>
          <span>{formatBytes(gpu.memory_total - gpu.memory_available)} / {formatBytes(gpu.memory_total)}</span>
        </div>
        <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
          <div
            className="h-full bg-current transition-all"
            style={{ width: `${usedPercent}%` }}
          />
        </div>
      </div>
      <div className="text-xs text-slate-400 mt-1">
        {gpu.compute_units} compute units
      </div>
    </div>
  );
}

interface JobCardProps {
  job: JobInfo;
  onCancel?: (id: string) => void;
  compact?: boolean;
}

function JobCard({ job, onCancel, compact }: JobCardProps) {
  const statusColor = STATUS_COLORS[job.status] ?? STATUS_COLORS.queued;
  const priorityColor = PRIORITY_COLORS[job.priority] ?? PRIORITY_COLORS.normal;

  if (compact) {
    return (
      <div className="flex items-center justify-between py-1 text-sm">
        <div className="flex items-center gap-2">
          <span className={`px-2 py-0.5 rounded text-xs border ${statusColor}`}>
            {job.status}
          </span>
          <span className="text-slate-400 truncate max-w-[200px]">{job.name}</span>
        </div>
        <span className="text-xs text-slate-500">
          {formatTimestamp(job.submitted_at)}
        </span>
      </div>
    );
  }

  return (
    <div className="border border-slate-700 rounded-lg p-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className={`px-2 py-0.5 rounded text-xs border ${statusColor}`}>
            {job.status}
          </span>
          <span className="font-medium">{job.name}</span>
          <span className={`px-2 py-0.5 rounded text-xs ${priorityColor}`}>
            {job.priority}
          </span>
        </div>
        <div className="flex items-center gap-2">
          {job.status !== 'completed' && job.status !== 'cancelled' && job.status !== 'failed' && onCancel && (
            <button
              onClick={() => onCancel(job.id)}
              className="p-1.5 text-red-400 hover:text-red-300 hover:bg-red-500/10 rounded transition-colors"
            >
              <Trash2 className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
      </div>
      <div className="grid grid-cols-3 gap-2 mt-2 text-xs text-slate-400">
        <div>
          <span className="text-slate-500">Tenant:</span> {job.tenant_id}
        </div>
        <div>
          <span className="text-slate-500">GPUs:</span> {job.gpus_requested}
        </div>
        <div>
          <span className="text-slate-500">Submitted:</span> {formatTimestamp(job.submitted_at)}
        </div>
      </div>
      {job.assigned_gpus.length > 0 && (
        <div className="mt-2 text-xs">
          <span className="text-slate-500">Assigned GPUs:</span>{' '}
          <span className="text-blue-400">{job.assigned_gpus.join(', ')}</span>
        </div>
      )}
    </div>
  );
}

interface TenantCardProps {
  tenant: TenantInfo;
  fairShare?: FairShareInfo;
}

function TenantCard({ tenant, fairShare }: TenantCardProps) {
  const gpuUsage = fairShare?.current_gpus ?? tenant.current_gpus;
  const usagePercent = (gpuUsage / tenant.max_gpus) * 100;

  return (
    <div className="border border-slate-700 rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <div>
          <div className="font-medium">{tenant.name}</div>
          <div className="text-xs text-slate-500">{tenant.id}</div>
        </div>
        <span className={`px-2 py-0.5 rounded text-xs border ${
          tenant.status === 'active' ? 'text-green-400 border-green-500/30' : 'text-slate-400 border-slate-500/30'
        }`}>
          {tenant.status}
        </span>
      </div>
      <div className="space-y-2">
        <div className="flex justify-between text-sm">
          <span className="text-slate-400">GPU Allocation</span>
          <span>{gpuUsage} / {tenant.max_gpus}</span>
        </div>
        <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
          <div
            className="h-full bg-cyan-500 transition-all"
            style={{ width: `${usagePercent}%` }}
          />
        </div>
      </div>
      <div className="grid grid-cols-2 gap-2 mt-3 text-xs">
        <div className="flex items-center gap-1">
          <div className="w-2 h-2 rounded-full bg-yellow-500" />
          <span className="text-slate-400">Queued:</span>
          <span>{fairShare?.queued_jobs ?? 0}</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-2 h-2 rounded-full bg-blue-500" />
          <span className="text-slate-400">Running:</span>
          <span>{fairShare?.running_jobs ?? 0}</span>
        </div>
      </div>
      <div className="text-xs text-slate-400 mt-2">
        Priority weight: {tenant.priority_weight}
      </div>
    </div>
  );
}

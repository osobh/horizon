/**
 * SchedulerView - SLAI GPU Scheduler Dashboard
 *
 * Provides a unified view of GPU cluster scheduling:
 * - GPU inventory with vendor/memory details
 * - Job queue management (submit, cancel, view status)
 * - Tenant fair-share allocation
 * - Real-time scheduler statistics
 */

import { useCallback } from 'react'
import {
  Cpu,
  Users,
  Layers,
  Play,
  Trash2,
  Plus,
  RefreshCw,
  CheckCircle2,
  Clock,
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import { cn } from '@/lib/utils'
import {
  useSchedulerStats,
  useSlaiGpus,
  useFairShare,
  useSlaiTenants,
  useSlaiJobs,
  useSubmitJob,
  useCancelJob,
  useScheduleNext,
} from '@/features/scheduler/api'
import type { GpuInfo, JobInfo, TenantInfo } from '@/features/scheduler/types'

const VENDOR_COLORS: Record<string, string> = {
  nvidia: 'text-green-400 bg-green-500/10 border-green-500/30',
  amd: 'text-red-400 bg-red-500/10 border-red-500/30',
  apple: 'text-blue-400 bg-blue-500/10 border-blue-500/30',
  intel: 'text-cyan-400 bg-cyan-500/10 border-cyan-500/30',
}

const STATUS_COLORS: Record<string, string> = {
  queued: 'bg-yellow-500/10 text-yellow-500 border-yellow-500/30',
  running: 'bg-blue-500/10 text-blue-500 border-blue-500/30',
  completed: 'bg-green-500/10 text-green-500 border-green-500/30',
  failed: 'bg-red-500/10 text-red-500 border-red-500/30',
  cancelled: 'bg-slate-500/10 text-slate-500 border-slate-500/30',
}

const PRIORITY_COLORS: Record<string, string> = {
  low: 'bg-slate-500/10 text-slate-400',
  normal: 'bg-blue-500/10 text-blue-400',
  high: 'bg-orange-500/10 text-orange-400',
  critical: 'bg-red-500/10 text-red-400',
}

function formatBytes(bytes: number): string {
  const gb = bytes / (1024 * 1024 * 1024)
  return `${gb.toFixed(0)} GB`
}

function formatTimestamp(ts: number): string {
  return new Date(ts * 1000).toLocaleTimeString()
}

export default function SchedulerView() {
  // React Query hooks
  const { data: stats, isLoading: statsLoading } = useSchedulerStats()
  const { data: gpus, isLoading: gpusLoading } = useSlaiGpus()
  const { data: fairShare } = useFairShare()
  const { data: tenants } = useSlaiTenants()
  const { data: jobs, isLoading: jobsLoading } = useSlaiJobs()
  const submitJob = useSubmitJob()
  const cancelJob = useCancelJob()
  const scheduleNext = useScheduleNext()

  const handleSubmitDemo = useCallback(() => {
    if (!tenants?.length) return
    submitJob.mutate({
      name: `demo-job-${Date.now().toString(36)}`,
      tenant_id: tenants[0].id,
      gpus: 1,
      priority: 'normal',
    })
  }, [submitJob, tenants])

  const handleCancelJob = useCallback((jobId: string) => {
    cancelJob.mutate(jobId)
  }, [cancelJob])

  const handleScheduleNext = useCallback(() => {
    scheduleNext.mutate()
  }, [scheduleNext])

  const isLoading = statsLoading || gpusLoading || jobsLoading

  return (
    <div className="h-full overflow-auto bg-slate-900">
      {/* Header */}
      <div className="h-16 bg-slate-800 border-b border-slate-700 flex items-center px-6">
        <div className="flex items-center gap-3">
          <Layers className="h-6 w-6 text-purple-400" />
          <h1 className="text-xl font-semibold">GPU Scheduler</h1>
          <Badge variant="outline" className="bg-purple-500/10 text-purple-400 border-purple-500/30">
            SLAI
          </Badge>
        </div>
        <div className="flex-1" />
        <div className="flex items-center gap-2">
          <Button
            size="sm"
            variant="outline"
            onClick={handleScheduleNext}
            disabled={scheduleNext.isPending || !jobs?.queued.length}
            className="text-blue-400 border-blue-500/30 hover:bg-blue-500/10"
          >
            <Play className="h-3 w-3 mr-1" />
            Schedule Next
          </Button>
          <Button
            size="sm"
            onClick={handleSubmitDemo}
            disabled={submitJob.isPending || !tenants?.length}
            className="bg-purple-600 hover:bg-purple-500"
          >
            <Plus className="h-3 w-3 mr-1" />
            Demo Job
          </Button>
        </div>
      </div>

      <div className="p-6 space-y-6">
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
            icon={CheckCircle2}
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
          <Card className="bg-slate-800 border-slate-700">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm flex items-center gap-2">
                <Cpu className="h-4 w-4 text-green-400" />
                GPU Inventory ({gpus?.length ?? 0})
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {isLoading ? (
                <div className="flex items-center justify-center py-8">
                  <RefreshCw className="h-5 w-5 animate-spin text-muted-foreground" />
                </div>
              ) : gpus?.length ? (
                gpus.map((gpu) => (
                  <GpuCard key={gpu.index} gpu={gpu} />
                ))
              ) : (
                <div className="text-center py-4 text-muted-foreground text-sm">
                  No GPUs detected
                </div>
              )}
            </CardContent>
          </Card>

          {/* Job Queue */}
          <Card className="bg-slate-800 border-slate-700 lg:col-span-2">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm flex items-center gap-2">
                <Layers className="h-4 w-4 text-purple-400" />
                Job Queue
                <Badge variant="secondary" className="ml-2">
                  {(jobs?.queued.length ?? 0) + (jobs?.running.length ?? 0)} active
                </Badge>
              </CardTitle>
            </CardHeader>
            <CardContent>
              {isLoading ? (
                <div className="flex items-center justify-center py-8">
                  <RefreshCw className="h-5 w-5 animate-spin text-muted-foreground" />
                </div>
              ) : (
                <div className="space-y-4">
                  {/* Running Jobs */}
                  {jobs?.running.length ? (
                    <div className="space-y-2">
                      <div className="text-xs text-muted-foreground uppercase tracking-wider">
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
                      <div className="text-xs text-muted-foreground uppercase tracking-wider">
                        Queued ({jobs.queued.length})
                      </div>
                      {jobs.queued.map((job) => (
                        <JobCard key={job.id} job={job} onCancel={handleCancelJob} />
                      ))}
                    </div>
                  ) : null}

                  {/* No Jobs */}
                  {!jobs?.running.length && !jobs?.queued.length && (
                    <div className="text-center py-8 text-muted-foreground text-sm">
                      No jobs in queue. Click "Demo Job" to submit one.
                    </div>
                  )}

                  {/* Completed Jobs (last 5) */}
                  {jobs?.completed.length ? (
                    <div className="space-y-2 pt-4 border-t border-slate-700">
                      <div className="text-xs text-muted-foreground uppercase tracking-wider">
                        Recently Completed ({jobs.completed.length})
                      </div>
                      {jobs.completed.slice(0, 5).map((job) => (
                        <JobCard key={job.id} job={job} compact />
                      ))}
                    </div>
                  ) : null}
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Tenants and Fair-Share */}
        <Card className="bg-slate-800 border-slate-700">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm flex items-center gap-2">
              <Users className="h-4 w-4 text-cyan-400" />
              Tenant Fair-Share Allocation
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {tenants?.map((tenant) => (
                <TenantCard
                  key={tenant.id}
                  tenant={tenant}
                  fairShare={fairShare?.[tenant.id]}
                />
              ))}
              {!tenants?.length && (
                <div className="col-span-full text-center py-4 text-muted-foreground text-sm">
                  No tenants registered
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

// Sub-components

interface StatCardProps {
  icon: React.ElementType
  label: string
  value: number | string
  color: string
}

function StatCard({ icon: Icon, label, value, color }: StatCardProps) {
  return (
    <Card className="bg-slate-800 border-slate-700">
      <CardContent className="p-4">
        <div className="flex items-center gap-3">
          <Icon className={cn('h-5 w-5', color)} />
          <div>
            <div className="text-2xl font-bold">{value}</div>
            <div className="text-xs text-muted-foreground">{label}</div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

function GpuCard({ gpu }: { gpu: GpuInfo }) {
  const usedPercent = ((gpu.memory_total - gpu.memory_available) / gpu.memory_total) * 100
  const vendorColor = VENDOR_COLORS[gpu.vendor] ?? VENDOR_COLORS.intel

  return (
    <Card className={cn('border', vendorColor)}>
      <CardContent className="p-3">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <Badge variant="outline" className={vendorColor}>
              {gpu.vendor.toUpperCase()}
            </Badge>
            <span className="text-sm font-medium truncate">{gpu.name}</span>
          </div>
          {gpu.is_primary && (
            <Badge variant="secondary" className="text-xs">Primary</Badge>
          )}
        </div>
        <div className="space-y-1">
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>Memory</span>
            <span>{formatBytes(gpu.memory_total - gpu.memory_available)} / {formatBytes(gpu.memory_total)}</span>
          </div>
          <Progress value={usedPercent} className="h-1.5" />
        </div>
        <div className="text-xs text-muted-foreground mt-1">
          {gpu.compute_units} compute units
        </div>
      </CardContent>
    </Card>
  )
}

interface JobCardProps {
  job: JobInfo
  onCancel?: (id: string) => void
  compact?: boolean
}

function JobCard({ job, onCancel, compact }: JobCardProps) {
  const statusColor = STATUS_COLORS[job.status] ?? STATUS_COLORS.queued
  const priorityColor = PRIORITY_COLORS[job.priority] ?? PRIORITY_COLORS.normal

  if (compact) {
    return (
      <div className="flex items-center justify-between py-1 text-sm">
        <div className="flex items-center gap-2">
          <Badge variant="outline" className={cn('text-xs', statusColor)}>
            {job.status}
          </Badge>
          <span className="text-muted-foreground truncate max-w-[200px]">{job.name}</span>
        </div>
        <span className="text-xs text-muted-foreground">
          {formatTimestamp(job.submitted_at)}
        </span>
      </div>
    )
  }

  return (
    <Card className="border-slate-700">
      <CardContent className="p-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Badge variant="outline" className={statusColor}>
              {job.status}
            </Badge>
            <span className="font-medium">{job.name}</span>
            <Badge variant="secondary" className={priorityColor}>
              {job.priority}
            </Badge>
          </div>
          <div className="flex items-center gap-2">
            {job.status !== 'completed' && job.status !== 'cancelled' && job.status !== 'failed' && onCancel && (
              <Button
                size="sm"
                variant="ghost"
                onClick={() => onCancel(job.id)}
                className="h-7 w-7 p-0 text-red-400 hover:text-red-300 hover:bg-red-500/10"
              >
                <Trash2 className="h-3.5 w-3.5" />
              </Button>
            )}
          </div>
        </div>
        <div className="grid grid-cols-3 gap-2 mt-2 text-xs text-muted-foreground">
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
      </CardContent>
    </Card>
  )
}

interface TenantCardProps {
  tenant: TenantInfo
  fairShare?: { current_gpus: number; queued_jobs: number; running_jobs: number }
}

function TenantCard({ tenant, fairShare }: TenantCardProps) {
  const gpuUsage = fairShare?.current_gpus ?? 0
  const usagePercent = (gpuUsage / tenant.max_gpus) * 100

  return (
    <Card className="border-slate-700">
      <CardContent className="p-4">
        <div className="flex items-center justify-between mb-3">
          <div>
            <div className="font-medium">{tenant.name}</div>
            <div className="text-xs text-muted-foreground">{tenant.id}</div>
          </div>
          <Badge
            variant="outline"
            className={tenant.status === 'active' ? 'text-green-400 border-green-500/30' : 'text-slate-400'}
          >
            {tenant.status}
          </Badge>
        </div>
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span className="text-muted-foreground">GPU Allocation</span>
            <span>{gpuUsage} / {tenant.max_gpus}</span>
          </div>
          <Progress value={usagePercent} className="h-2" />
        </div>
        <div className="grid grid-cols-2 gap-2 mt-3 text-xs">
          <div className="flex items-center gap-1">
            <div className="w-2 h-2 rounded-full bg-yellow-500" />
            <span className="text-muted-foreground">Queued:</span>
            <span>{fairShare?.queued_jobs ?? 0}</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-2 h-2 rounded-full bg-blue-500" />
            <span className="text-muted-foreground">Running:</span>
            <span>{fairShare?.running_jobs ?? 0}</span>
          </div>
        </div>
        <div className="text-xs text-muted-foreground mt-2">
          Priority weight: {tenant.priority_weight}
        </div>
      </CardContent>
    </Card>
  )
}

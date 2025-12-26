/**
 * SLAI Scheduler TypeScript Types
 *
 * These types match the Rust types in slai_bridge.rs
 */

/** GPU vendor types */
export type GpuVendor = 'nvidia' | 'amd' | 'apple' | 'intel'

/** Job priority levels */
export type JobPriority = 'low' | 'normal' | 'high' | 'critical'

/** Job status */
export type JobStatus = 'queued' | 'running' | 'completed' | 'failed' | 'cancelled'

/** GPU information */
export interface GpuInfo {
  /** GPU index */
  index: number
  /** GPU name/model */
  name: string
  /** Vendor (nvidia, amd, apple, intel) */
  vendor: string
  /** Total memory in bytes */
  memory_total: number
  /** Available memory in bytes */
  memory_available: number
  /** Number of compute units */
  compute_units: number
  /** Is this the primary GPU */
  is_primary: boolean
}

/** Scheduler statistics */
export interface SchedulerStats {
  /** Total number of GPUs */
  total_gpus: number
  /** Available (unallocated) GPUs */
  available_gpus: number
  /** Jobs waiting in queue */
  queued_jobs: number
  /** Currently running jobs */
  running_jobs: number
  /** Completed jobs in history */
  completed_jobs: number
  /** Registered tenants */
  tenant_count: number
}

/** Fair-share allocation info for a tenant */
export interface FairShareInfo {
  /** Tenant name */
  tenant_name: string
  /** Priority weight (1-1000) */
  priority_weight: number
  /** Maximum GPUs allowed */
  max_gpus: number
  /** Currently allocated GPUs */
  current_gpus: number
  /** Number of queued jobs */
  queued_jobs: number
  /** Number of running jobs */
  running_jobs: number
}

/** Tenant information */
export interface TenantInfo {
  /** Tenant ID */
  id: string
  /** Tenant name */
  name: string
  /** Priority weight */
  priority_weight: number
  /** Maximum GPUs allowed */
  max_gpus: number
  /** Maximum concurrent jobs */
  max_concurrent_jobs: number
  /** Currently allocated GPUs */
  current_gpus: number
  /** Status (active, suspended) */
  status: string
}

/** Job information */
export interface JobInfo {
  /** Job ID */
  id: string
  /** Job name */
  name: string
  /** Tenant ID */
  tenant_id: string
  /** Job status */
  status: string
  /** Priority level */
  priority: string
  /** GPUs requested */
  gpus_requested: number
  /** Submitted timestamp (unix epoch) */
  submitted_at: number
  /** Started timestamp (if running) */
  started_at?: number
  /** Assigned GPU indices (if running) */
  assigned_gpus: number[]
}

/** Job list with queued, running, and completed jobs */
export interface JobList {
  /** Jobs waiting in queue */
  queued: JobInfo[]
  /** Currently running jobs */
  running: JobInfo[]
  /** Recently completed jobs */
  completed: JobInfo[]
}

/** Job submission request */
export interface JobRequest {
  /** Job name */
  name: string
  /** Tenant ID */
  tenant_id: string
  /** Number of GPUs required */
  gpus: number
  /** Priority level (low, normal, high, critical) */
  priority: string
}

/** Fair-share allocation map */
export type FairShareMap = Record<string, FairShareInfo>

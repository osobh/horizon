/**
 * SLAI Scheduler Query Hooks
 *
 * TanStack Query hooks for SLAI scheduler Tauri commands.
 * Provides real-time GPU detection, job management, and tenant fair-share.
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { invoke } from '@tauri-apps/api/core'
import { queryKeys } from '@/lib/query-keys'
import type {
  SchedulerStats,
  GpuInfo,
  FairShareMap,
  TenantInfo,
  JobList,
  JobRequest,
  JobInfo,
} from '../types'

/**
 * Hook to get scheduler statistics
 * Refreshes every 2 seconds for real-time updates
 */
export function useSchedulerStats() {
  return useQuery({
    queryKey: queryKeys.slai.stats(),
    queryFn: () => invoke<SchedulerStats>('get_slai_stats'),
    refetchInterval: 2000,
  })
}

/**
 * Hook to get GPU inventory
 * Refreshes every 5 seconds
 */
export function useSlaiGpus() {
  return useQuery({
    queryKey: queryKeys.slai.gpus(),
    queryFn: () => invoke<GpuInfo[]>('get_slai_gpus'),
    refetchInterval: 5000,
  })
}

/**
 * Hook to get fair-share allocation per tenant
 * Refreshes every 2 seconds
 */
export function useFairShare() {
  return useQuery({
    queryKey: queryKeys.slai.fairShare(),
    queryFn: () => invoke<FairShareMap>('get_slai_fair_share'),
    refetchInterval: 2000,
  })
}

/**
 * Hook to list all tenants
 */
export function useSlaiTenants() {
  return useQuery({
    queryKey: queryKeys.slai.tenants(),
    queryFn: () => invoke<TenantInfo[]>('list_slai_tenants'),
    refetchInterval: 10000,
  })
}

/**
 * Hook to create a new tenant
 */
export function useCreateTenant() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ name, maxGpus, maxConcurrentJobs }: {
      name: string
      maxGpus: number
      maxConcurrentJobs: number
    }) => invoke<TenantInfo>('create_slai_tenant', {
      name,
      maxGpus,
      maxConcurrentJobs,
    }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.slai.tenants() })
      queryClient.invalidateQueries({ queryKey: queryKeys.slai.fairShare() })
      queryClient.invalidateQueries({ queryKey: queryKeys.slai.stats() })
    },
  })
}

/**
 * Hook to list all jobs (queued, running, completed)
 * Refreshes every 2 seconds
 */
export function useSlaiJobs() {
  return useQuery({
    queryKey: queryKeys.slai.jobs(),
    queryFn: () => invoke<JobList>('list_slai_jobs'),
    refetchInterval: 2000,
  })
}

/**
 * Hook to submit a new job
 */
export function useSubmitJob() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (job: JobRequest) =>
      invoke<string>('submit_slai_job', { job }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.slai.jobs() })
      queryClient.invalidateQueries({ queryKey: queryKeys.slai.stats() })
      queryClient.invalidateQueries({ queryKey: queryKeys.slai.fairShare() })
    },
  })
}

/**
 * Hook to cancel a job
 */
export function useCancelJob() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (jobId: string) =>
      invoke('cancel_slai_job', { jobId }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.slai.jobs() })
      queryClient.invalidateQueries({ queryKey: queryKeys.slai.stats() })
      queryClient.invalidateQueries({ queryKey: queryKeys.slai.fairShare() })
    },
  })
}

/**
 * Hook to schedule the next pending job (for demo/testing)
 */
export function useScheduleNext() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: () => invoke<JobInfo | null>('schedule_slai_next'),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.slai.jobs() })
      queryClient.invalidateQueries({ queryKey: queryKeys.slai.stats() })
      queryClient.invalidateQueries({ queryKey: queryKeys.slai.fairShare() })
    },
  })
}

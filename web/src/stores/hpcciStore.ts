import { create } from 'zustand';
import { invoke } from '@tauri-apps/api/core';

// Types matching the Rust backend
export type PipelineStatus = 'queued' | 'running' | 'success' | 'failed' | 'cancelled' | 'timeout' | 'skipped';
export type AgentStatus = 'online' | 'offline' | 'draining' | 'maintenance';
export type ApprovalStatus = 'pending' | 'approved' | 'rejected' | 'expired';

export interface PipelineSummary {
  id: string;
  repo: string;
  branch: string;
  sha_short: string;
  status: PipelineStatus;
  duration_ms: number | null;
  started_at_ms: number;
  stages_progress: string;
}

export interface StageInfo {
  name: string;
  status: PipelineStatus;
  duration_ms: number | null;
  jobs: JobInfo[];
}

export interface JobInfo {
  name: string;
  status: PipelineStatus;
  duration_ms: number | null;
  agent_id: string | null;
}

export interface TriggerInfo {
  trigger_type: string;
  user: string | null;
  commit_message: string | null;
}

export interface PipelineDetail {
  id: string;
  repo: string;
  branch: string;
  sha: string;
  status: PipelineStatus;
  duration_ms: number | null;
  started_at_ms: number;
  finished_at_ms: number | null;
  stages: StageInfo[];
  trigger: TriggerInfo;
}

export interface AgentSummary {
  id: string;
  status: AgentStatus;
  current_jobs: number;
  max_jobs: number;
  capabilities: string[];
  gpu_count: number | null;
}

export interface DashboardSummary {
  pipelines_running: number;
  pipelines_queued: number;
  pipelines_succeeded_24h: number;
  pipelines_failed_24h: number;
  agents_online: number;
  agents_total: number;
  recent_pipelines: PipelineSummary[];
}

export interface ApprovalRequest {
  id: string;
  pipeline_id: string;
  environment: string;
  requested_by: string;
  requested_at_ms: number;
  status: ApprovalStatus;
}

export interface PipelineFilter {
  status?: PipelineStatus;
  repo?: string;
  branch?: string;
  limit?: number;
  offset?: number;
}

export interface TriggerParams {
  repo: string;
  branch: string;
  sha?: string;
  inputs?: Record<string, unknown>;
}

export interface LogEntry {
  timestamp_ms: number;
  stage: string;
  job: string;
  level: string;
  content: string;
}

export interface LogsResponse {
  pipeline_id: string;
  entries: LogEntry[];
  has_more: boolean;
  next_offset: number | null;
}

export interface HpcCiStatus {
  connected: boolean;
  server_url: string;
  version: string;
  summary: DashboardSummary | null;
}

interface HpcCiState {
  status: HpcCiStatus | null;
  pipelines: PipelineSummary[];
  selectedPipeline: PipelineDetail | null;
  agents: AgentSummary[];
  approvals: ApprovalRequest[];
  logs: LogEntry[];
  loading: boolean;
  error: string | null;

  // Actions
  fetchStatus: () => Promise<void>;
  setServerUrl: (url: string) => Promise<void>;
  fetchPipelines: (filter?: PipelineFilter) => Promise<void>;
  fetchPipelineDetail: (id: string) => Promise<void>;
  triggerPipeline: (params: TriggerParams) => Promise<string>;
  cancelPipeline: (id: string) => Promise<void>;
  retryPipeline: (id: string) => Promise<string>;
  fetchAgents: () => Promise<void>;
  drainAgent: (id: string) => Promise<void>;
  enableAgent: (id: string) => Promise<void>;
  fetchApprovals: () => Promise<void>;
  submitApproval: (id: string, approved: boolean, comment?: string) => Promise<void>;
  fetchDashboardSummary: () => Promise<DashboardSummary>;
  fetchPipelineLogs: (id: string, offset?: number) => Promise<void>;
}

export const useHpcCiStore = create<HpcCiState>((set, get) => ({
  status: null,
  pipelines: [],
  selectedPipeline: null,
  agents: [],
  approvals: [],
  logs: [],
  loading: false,
  error: null,

  fetchStatus: async () => {
    try {
      const status = await invoke<HpcCiStatus>('get_hpcci_status');
      set({ status, error: null });
    } catch (e) {
      set({ error: String(e) });
    }
  },

  setServerUrl: async (url: string) => {
    try {
      await invoke('set_hpcci_server_url', { url });
      await get().fetchStatus();
    } catch (e) {
      set({ error: String(e) });
    }
  },

  fetchPipelines: async (filter?: PipelineFilter) => {
    set({ loading: true });
    try {
      const pipelines = await invoke<PipelineSummary[]>('list_hpcci_pipelines', { filter });
      set({ pipelines, loading: false, error: null });
    } catch (e) {
      set({ loading: false, error: String(e) });
    }
  },

  fetchPipelineDetail: async (id: string) => {
    set({ loading: true });
    try {
      const selectedPipeline = await invoke<PipelineDetail>('get_hpcci_pipeline', { id });
      set({ selectedPipeline, loading: false, error: null });
    } catch (e) {
      set({ loading: false, error: String(e) });
    }
  },

  triggerPipeline: async (params: TriggerParams) => {
    set({ loading: true });
    try {
      const pipelineId = await invoke<string>('trigger_hpcci_pipeline', { params });
      await get().fetchPipelines();
      set({ loading: false, error: null });
      return pipelineId;
    } catch (e) {
      set({ loading: false, error: String(e) });
      throw e;
    }
  },

  cancelPipeline: async (id: string) => {
    try {
      await invoke('cancel_hpcci_pipeline', { id });
      await get().fetchPipelines();
      if (get().selectedPipeline?.id === id) {
        await get().fetchPipelineDetail(id);
      }
    } catch (e) {
      set({ error: String(e) });
    }
  },

  retryPipeline: async (id: string) => {
    try {
      const newId = await invoke<string>('retry_hpcci_pipeline', { id });
      await get().fetchPipelines();
      return newId;
    } catch (e) {
      set({ error: String(e) });
      throw e;
    }
  },

  fetchAgents: async () => {
    try {
      const agents = await invoke<AgentSummary[]>('list_hpcci_agents');
      set({ agents, error: null });
    } catch (e) {
      set({ error: String(e) });
    }
  },

  drainAgent: async (id: string) => {
    try {
      await invoke('drain_hpcci_agent', { id });
      await get().fetchAgents();
    } catch (e) {
      set({ error: String(e) });
    }
  },

  enableAgent: async (id: string) => {
    try {
      await invoke('enable_hpcci_agent', { id });
      await get().fetchAgents();
    } catch (e) {
      set({ error: String(e) });
    }
  },

  fetchApprovals: async () => {
    try {
      const approvals = await invoke<ApprovalRequest[]>('get_hpcci_approvals');
      set({ approvals, error: null });
    } catch (e) {
      set({ error: String(e) });
    }
  },

  submitApproval: async (id: string, approved: boolean, comment?: string) => {
    try {
      await invoke('submit_hpcci_approval', { id, approved, comment });
      await get().fetchApprovals();
    } catch (e) {
      set({ error: String(e) });
    }
  },

  fetchDashboardSummary: async () => {
    try {
      const summary = await invoke<DashboardSummary>('get_hpcci_dashboard_summary');
      set({ error: null });
      return summary;
    } catch (e) {
      set({ error: String(e) });
      throw e;
    }
  },

  fetchPipelineLogs: async (id: string, offset?: number) => {
    try {
      const response = await invoke<LogsResponse>('get_hpcci_pipeline_logs', { id, offset });
      if (offset) {
        // Append to existing logs
        set((state) => ({ logs: [...state.logs, ...response.entries], error: null }));
      } else {
        set({ logs: response.entries, error: null });
      }
    } catch (e) {
      set({ error: String(e) });
    }
  },
}));

import { create } from 'zustand';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';

// Types matching the Rust backend
export type TransferOperation = 'upload' | 'download' | 'sync';
export type TransferStatus = 'queued' | 'analyzing' | 'transferring' | 'verifying' | 'completed' | 'failed' | 'cancelled' | 'paused';

export interface TransferProgress {
  bytes_transferred: number;
  total_bytes: number;
  chunks_completed: number;
  total_chunks: number;
  current_file: string | null;
  bytes_per_second: number;
  completion_percentage: number;
  eta_seconds: number | null;
}

export interface Transfer {
  id: string;
  name: string;
  operation: TransferOperation;
  status: TransferStatus;
  progress: TransferProgress;
  source: string;
  destination: string;
  created_at: number;
  started_at: number | null;
  completed_at: number | null;
  merkle_root: string | null;
  error: string | null;
}

export interface StorageStats {
  total_uploads: number;
  total_downloads: number;
  bytes_uploaded: number;
  bytes_downloaded: number;
  active_transfers: number;
  completed_transfers: number;
  failed_transfers: number;
}

export interface FileInfo {
  name: string;
  path: string;
  size: number;
  is_directory: boolean;
  modified_at: number | null;
  merkle_root: string | null;
}

export interface TransferConfig {
  max_concurrent_chunks: number;
  enable_gpu: boolean;
  compression: string;
  compression_level: number;
  verify_on_complete: boolean;
}

interface StorageState {
  transfers: Transfer[];
  activeTransfers: Transfer[];
  files: FileInfo[];
  stats: StorageStats | null;
  currentPath: string;
  loading: boolean;
  error: string | null;

  // Actions
  fetchTransfers: () => Promise<void>;
  fetchActiveTransfers: () => Promise<void>;
  fetchStats: () => Promise<void>;
  listFiles: (path: string) => Promise<void>;
  uploadFile: (source: string, destination: string, config?: TransferConfig) => Promise<Transfer>;
  downloadFile: (source: string, destination: string, config?: TransferConfig) => Promise<Transfer>;
  pauseTransfer: (transferId: string) => Promise<void>;
  resumeTransfer: (transferId: string) => Promise<void>;
  cancelTransfer: (transferId: string) => Promise<void>;
  getTransfer: (transferId: string) => Promise<Transfer>;
  updateTransferProgress: (transferId: string, progress: TransferProgress) => void;
  initEventListeners: () => Promise<() => void>;
}

export const useStorageStore = create<StorageState>((set, get) => ({
  transfers: [],
  activeTransfers: [],
  files: [],
  stats: null,
  currentPath: '/',
  loading: false,
  error: null,

  fetchTransfers: async () => {
    try {
      set({ loading: true, error: null });
      const transfers = await invoke<Transfer[]>('list_transfers');
      set({ transfers, loading: false });
    } catch (error) {
      set({ error: String(error), loading: false });
    }
  },

  fetchActiveTransfers: async () => {
    try {
      const activeTransfers = await invoke<Transfer[]>('list_active_transfers');
      set({ activeTransfers });
    } catch (error) {
      set({ error: String(error) });
    }
  },

  fetchStats: async () => {
    try {
      const stats = await invoke<StorageStats>('get_storage_stats');
      set({ stats });
    } catch (error) {
      set({ error: String(error) });
    }
  },

  listFiles: async (path: string) => {
    try {
      set({ loading: true, error: null });
      const files = await invoke<FileInfo[]>('list_files', { path });
      set({ files, currentPath: path, loading: false });
    } catch (error) {
      set({ error: String(error), loading: false });
    }
  },

  uploadFile: async (source: string, destination: string, config?: TransferConfig) => {
    try {
      set({ loading: true, error: null });
      const transfer = await invoke<Transfer>('upload_file', {
        input: { source, destination, config },
      });
      await get().fetchActiveTransfers();
      set({ loading: false });
      return transfer;
    } catch (error) {
      set({ error: String(error), loading: false });
      throw error;
    }
  },

  downloadFile: async (source: string, destination: string, config?: TransferConfig) => {
    try {
      set({ loading: true, error: null });
      const transfer = await invoke<Transfer>('download_file', {
        input: { source, destination, config },
      });
      await get().fetchActiveTransfers();
      set({ loading: false });
      return transfer;
    } catch (error) {
      set({ error: String(error), loading: false });
      throw error;
    }
  },

  pauseTransfer: async (transferId: string) => {
    try {
      await invoke('pause_transfer', { transferId });
      await get().fetchActiveTransfers();
    } catch (error) {
      set({ error: String(error) });
    }
  },

  resumeTransfer: async (transferId: string) => {
    try {
      await invoke('resume_transfer', { transferId });
      await get().fetchActiveTransfers();
    } catch (error) {
      set({ error: String(error) });
    }
  },

  cancelTransfer: async (transferId: string) => {
    try {
      await invoke('cancel_transfer', { transferId });
      await get().fetchActiveTransfers();
    } catch (error) {
      set({ error: String(error) });
    }
  },

  getTransfer: async (transferId: string) => {
    const transfer = await invoke<Transfer>('get_transfer', { transferId });
    return transfer;
  },

  updateTransferProgress: (transferId: string, progress: TransferProgress) => {
    set((state) => ({
      activeTransfers: state.activeTransfers.map((t) =>
        t.id === transferId ? { ...t, progress } : t
      ),
      transfers: state.transfers.map((t) =>
        t.id === transferId ? { ...t, progress } : t
      ),
    }));
  },

  initEventListeners: async () => {
    const unlisten = await listen<{ transfer_id: string; progress: TransferProgress }>(
      'transfer:progress',
      (event) => {
        get().updateTransferProgress(event.payload.transfer_id, event.payload.progress);
      }
    );
    return unlisten;
  },
}));

import { useEffect, useState } from 'react';
import {
  HardDrive,
  Upload,
  Download,
  Pause,
  Play,
  X,
  Folder,
  File,
  RefreshCw,
  ChevronRight,
} from 'lucide-react';
import { useStorageStore, Transfer, FileInfo } from '../../stores/storageStore';

export default function StorageView() {
  const {
    transfers,
    activeTransfers,
    files,
    stats,
    currentPath,
    loading,
    error,
    fetchTransfers,
    fetchActiveTransfers,
    fetchStats,
    listFiles,
    pauseTransfer,
    resumeTransfer,
    cancelTransfer,
    initEventListeners,
  } = useStorageStore();

  const [selectedTab, setSelectedTab] = useState<'browser' | 'transfers'>('browser');

  useEffect(() => {
    fetchStats();
    fetchActiveTransfers();
    listFiles('/');

    // Initialize event listeners for real-time updates
    let cleanup: (() => void) | undefined;
    initEventListeners().then((unlisten) => {
      cleanup = unlisten;
    });

    // Poll for active transfers
    const interval = setInterval(() => {
      fetchActiveTransfers();
    }, 2000);

    return () => {
      if (cleanup) cleanup();
      clearInterval(interval);
    };
  }, []);

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatSpeed = (bytesPerSecond: number) => {
    return formatBytes(bytesPerSecond) + '/s';
  };

  return (
    <div className="h-full flex flex-col bg-slate-900 text-white">
      {/* Header */}
      <div className="h-16 bg-slate-800 border-b border-slate-700 flex items-center px-6">
        <HardDrive className="w-6 h-6 text-green-400 mr-3" />
        <h1 className="text-xl font-semibold">Storage (WARP)</h1>
        <div className="flex-1" />
        <button
          onClick={() => { fetchStats(); fetchActiveTransfers(); }}
          className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
        >
          <RefreshCw className={`w-5 h-5 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Stats Bar */}
      {stats && (
        <div className="bg-slate-800/50 border-b border-slate-700 px-6 py-3 flex gap-6 text-sm">
          <div>
            <span className="text-slate-400">Uploads:</span>{' '}
            <span className="text-green-400">{stats.total_uploads}</span>
          </div>
          <div>
            <span className="text-slate-400">Downloads:</span>{' '}
            <span className="text-blue-400">{stats.total_downloads}</span>
          </div>
          <div>
            <span className="text-slate-400">Uploaded:</span>{' '}
            <span className="text-white">{formatBytes(stats.bytes_uploaded)}</span>
          </div>
          <div>
            <span className="text-slate-400">Downloaded:</span>{' '}
            <span className="text-white">{formatBytes(stats.bytes_downloaded)}</span>
          </div>
          <div>
            <span className="text-slate-400">Active:</span>{' '}
            <span className="text-amber-400">{stats.active_transfers}</span>
          </div>
        </div>
      )}

      {/* Tabs */}
      <div className="border-b border-slate-700 px-6">
        <div className="flex gap-4">
          <button
            onClick={() => setSelectedTab('browser')}
            className={`py-3 px-4 border-b-2 transition-colors ${
              selectedTab === 'browser'
                ? 'border-green-400 text-green-400'
                : 'border-transparent text-slate-400 hover:text-white'
            }`}
          >
            <Folder className="w-4 h-4 inline mr-2" />
            File Browser
          </button>
          <button
            onClick={() => setSelectedTab('transfers')}
            className={`py-3 px-4 border-b-2 transition-colors ${
              selectedTab === 'transfers'
                ? 'border-green-400 text-green-400'
                : 'border-transparent text-slate-400 hover:text-white'
            }`}
          >
            <RefreshCw className="w-4 h-4 inline mr-2" />
            Active Transfers
            {activeTransfers.length > 0 && (
              <span className="ml-2 bg-green-500 text-white text-xs px-2 py-0.5 rounded-full">
                {activeTransfers.length}
              </span>
            )}
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto p-6">
        {error && (
          <div className="mb-4 p-4 bg-red-500/20 border border-red-500 rounded-lg text-red-400">
            {error}
          </div>
        )}

        {selectedTab === 'browser' && (
          <div className="space-y-4">
            {/* Breadcrumb */}
            <div className="flex items-center gap-2 text-sm text-slate-400">
              <button
                onClick={() => listFiles('/')}
                className="hover:text-white transition-colors"
              >
                Root
              </button>
              {currentPath !== '/' &&
                currentPath.split('/').filter(Boolean).map((part, i, arr) => (
                  <span key={i} className="flex items-center gap-2">
                    <ChevronRight className="w-4 h-4" />
                    <button
                      onClick={() => listFiles('/' + arr.slice(0, i + 1).join('/'))}
                      className="hover:text-white transition-colors"
                    >
                      {part}
                    </button>
                  </span>
                ))}
            </div>

            {/* File List */}
            <div className="bg-slate-800 rounded-lg border border-slate-700">
              <div className="grid grid-cols-12 gap-4 p-3 border-b border-slate-700 text-sm text-slate-400">
                <div className="col-span-6">Name</div>
                <div className="col-span-2">Size</div>
                <div className="col-span-3">Modified</div>
                <div className="col-span-1">Actions</div>
              </div>
              {files.length === 0 ? (
                <div className="p-8 text-center text-slate-500">
                  {loading ? 'Loading...' : 'No files in this directory'}
                </div>
              ) : (
                files.map((file) => (
                  <FileRow
                    key={file.path}
                    file={file}
                    onNavigate={() => file.is_directory && listFiles(file.path)}
                  />
                ))
              )}
            </div>
          </div>
        )}

        {selectedTab === 'transfers' && (
          <div className="space-y-4">
            {activeTransfers.length === 0 ? (
              <div className="p-8 text-center text-slate-500 bg-slate-800 rounded-lg border border-slate-700">
                No active transfers
              </div>
            ) : (
              activeTransfers.map((transfer) => (
                <TransferCard
                  key={transfer.id}
                  transfer={transfer}
                  onPause={() => pauseTransfer(transfer.id)}
                  onResume={() => resumeTransfer(transfer.id)}
                  onCancel={() => cancelTransfer(transfer.id)}
                />
              ))
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function FileRow({ file, onNavigate }: { file: FileInfo; onNavigate: () => void }) {
  const formatDate = (timestamp: number | null) => {
    if (!timestamp) return '-';
    return new Date(timestamp * 1000).toLocaleDateString();
  };

  const formatSize = (bytes: number) => {
    if (bytes === 0) return '-';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  return (
    <div
      className={`grid grid-cols-12 gap-4 p-3 border-b border-slate-700/50 hover:bg-slate-700/30 transition-colors ${
        file.is_directory ? 'cursor-pointer' : ''
      }`}
      onClick={onNavigate}
    >
      <div className="col-span-6 flex items-center gap-3">
        {file.is_directory ? (
          <Folder className="w-5 h-5 text-amber-400" />
        ) : (
          <File className="w-5 h-5 text-slate-400" />
        )}
        <span className={file.is_directory ? 'text-amber-400' : ''}>{file.name}</span>
      </div>
      <div className="col-span-2 text-slate-400">{file.is_directory ? '-' : formatSize(file.size)}</div>
      <div className="col-span-3 text-slate-400">{formatDate(file.modified_at)}</div>
      <div className="col-span-1 flex gap-2">
        {!file.is_directory && (
          <button className="p-1 hover:bg-slate-600 rounded transition-colors">
            <Download className="w-4 h-4 text-blue-400" />
          </button>
        )}
      </div>
    </div>
  );
}

function TransferCard({
  transfer,
  onPause,
  onResume,
  onCancel,
}: {
  transfer: Transfer;
  onPause: () => void;
  onResume: () => void;
  onCancel: () => void;
}) {
  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const statusColors: Record<string, string> = {
    queued: 'text-slate-400',
    analyzing: 'text-blue-400',
    transferring: 'text-green-400',
    verifying: 'text-amber-400',
    completed: 'text-green-400',
    failed: 'text-red-400',
    cancelled: 'text-slate-500',
    paused: 'text-amber-400',
  };

  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          {transfer.operation === 'upload' ? (
            <Upload className="w-5 h-5 text-green-400" />
          ) : (
            <Download className="w-5 h-5 text-blue-400" />
          )}
          <div>
            <div className="font-medium">{transfer.name}</div>
            <div className="text-sm text-slate-400">{transfer.source} → {transfer.destination}</div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className={`text-sm ${statusColors[transfer.status] || 'text-slate-400'}`}>
            {transfer.status.charAt(0).toUpperCase() + transfer.status.slice(1)}
          </span>
          {transfer.status === 'transferring' && (
            <button onClick={onPause} className="p-2 hover:bg-slate-700 rounded-lg transition-colors">
              <Pause className="w-4 h-4" />
            </button>
          )}
          {transfer.status === 'paused' && (
            <button onClick={onResume} className="p-2 hover:bg-slate-700 rounded-lg transition-colors">
              <Play className="w-4 h-4" />
            </button>
          )}
          {['transferring', 'paused', 'queued'].includes(transfer.status) && (
            <button onClick={onCancel} className="p-2 hover:bg-slate-700 rounded-lg transition-colors text-red-400">
              <X className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      {/* Progress Bar */}
      <div className="mb-2">
        <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
          <div
            className="h-full bg-green-500 transition-all duration-300"
            style={{ width: `${transfer.progress.completion_percentage}%` }}
          />
        </div>
      </div>

      {/* Progress Info */}
      <div className="flex justify-between text-sm text-slate-400">
        <span>
          {formatBytes(transfer.progress.bytes_transferred)} / {formatBytes(transfer.progress.total_bytes)}
        </span>
        <span>{transfer.progress.completion_percentage.toFixed(1)}%</span>
        <span>{formatBytes(transfer.progress.bytes_per_second)}/s</span>
        {transfer.progress.eta_seconds && (
          <span>ETA: {Math.ceil(transfer.progress.eta_seconds / 60)} min</span>
        )}
      </div>
    </div>
  );
}

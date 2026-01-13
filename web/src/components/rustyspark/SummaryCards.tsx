/**
 * RustySpark Summary Cards
 *
 * Displays summary statistics for Spark processing.
 */

import React from 'react';
import {
  Briefcase,
  Play,
  Layers,
  Cpu,
  Database,
  Shuffle,
  Loader2,
  RefreshCw,
} from 'lucide-react';
import { useRustySparkStore } from '../../stores/rustysparkStore';

// Format bytes to human readable
function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
}

interface StatCardProps {
  icon: React.ReactNode;
  iconColor: string;
  iconBg: string;
  label: string;
  value: string | number;
  subValue?: string;
}

function StatCard({ icon, iconColor, iconBg, label, value, subValue }: StatCardProps) {
  return (
    <div className="bg-[#0f1729] rounded-lg border border-[#1e293b] p-4">
      <div className="flex items-start justify-between">
        <div className={`p-2 rounded-lg ${iconBg}`}>
          <div className={iconColor}>{icon}</div>
        </div>
      </div>
      <div className="mt-3">
        <div className="text-2xl font-bold text-white">{value}</div>
        <div className="text-sm text-gray-400">{label}</div>
        {subValue && <div className="text-xs text-gray-500 mt-1">{subValue}</div>}
      </div>
    </div>
  );
}

export function SummaryCards() {
  const { summary, status, isLoading, fetchSummary, fetchStatus } = useRustySparkStore();

  React.useEffect(() => {
    fetchStatus();
    fetchSummary();
    const interval = setInterval(() => {
      fetchSummary();
    }, 15000);
    return () => clearInterval(interval);
  }, [fetchStatus, fetchSummary]);

  const handleRefresh = () => {
    fetchStatus();
    fetchSummary();
  };

  if (isLoading && !summary) {
    return (
      <div className="flex items-center justify-center h-32">
        <Loader2 className="h-6 w-6 animate-spin text-[#00d4ff]" />
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Status Banner */}
      <div className="bg-[#0f1729] rounded-lg border border-[#1e293b] p-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className={`w-2 h-2 rounded-full ${status.connected ? 'bg-green-400' : 'bg-red-400'}`} />
          <span className="text-sm text-gray-300">
            {status.connected ? 'Connected to RustySpark' : 'Disconnected'}
          </span>
          {status.version && (
            <span className="text-xs text-gray-500">v{status.version}</span>
          )}
        </div>
        <button
          onClick={handleRefresh}
          disabled={isLoading}
          className="p-1.5 text-gray-400 hover:text-white hover:bg-[#1e293b] rounded transition-colors disabled:opacity-50"
        >
          <RefreshCw size={14} className={isLoading ? 'animate-spin' : ''} />
        </button>
      </div>

      {/* Summary Cards Grid */}
      {summary && (
        <div className="grid grid-cols-4 gap-4">
          <StatCard
            icon={<Briefcase size={20} />}
            iconColor="text-blue-400"
            iconBg="bg-blue-400/10"
            label="Total Jobs"
            value={summary.total_jobs}
            subValue={`${summary.running_jobs} running, ${summary.completed_jobs} completed`}
          />
          <StatCard
            icon={<Play size={20} />}
            iconColor="text-green-400"
            iconBg="bg-green-400/10"
            label="Active Jobs"
            value={summary.running_jobs}
            subValue={`${summary.failed_jobs} failed`}
          />
          <StatCard
            icon={<Layers size={20} />}
            iconColor="text-purple-400"
            iconBg="bg-purple-400/10"
            label="Stages"
            value={summary.total_stages}
            subValue={`${summary.active_stages} active`}
          />
          <StatCard
            icon={<Cpu size={20} />}
            iconColor="text-yellow-400"
            iconBg="bg-yellow-400/10"
            label="Tasks"
            value={summary.total_tasks}
            subValue={`${summary.active_tasks} active`}
          />
        </div>
      )}

      {/* Data Volume Cards */}
      {summary && (
        <div className="grid grid-cols-3 gap-4">
          <StatCard
            icon={<Database size={20} />}
            iconColor="text-cyan-400"
            iconBg="bg-cyan-400/10"
            label="Total Input"
            value={formatBytes(summary.total_input_bytes)}
          />
          <StatCard
            icon={<Database size={20} />}
            iconColor="text-emerald-400"
            iconBg="bg-emerald-400/10"
            label="Total Output"
            value={formatBytes(summary.total_output_bytes)}
          />
          <StatCard
            icon={<Shuffle size={20} />}
            iconColor="text-orange-400"
            iconBg="bg-orange-400/10"
            label="Shuffle Data"
            value={formatBytes(summary.total_shuffle_bytes)}
          />
        </div>
      )}
    </div>
  );
}

import { useEffect } from 'react';
import { Server, CheckCircle, XCircle, HelpCircle, RefreshCw } from 'lucide-react';
import { useArgusStore, Target } from '../../stores/argusStore';

const healthConfig = {
  up: { icon: CheckCircle, color: 'text-green-400', bgColor: 'bg-green-500/10' },
  down: { icon: XCircle, color: 'text-red-400', bgColor: 'bg-red-500/10' },
  unknown: { icon: HelpCircle, color: 'text-slate-400', bgColor: 'bg-slate-500/10' },
};

function TargetCard({ target }: { target: Target }) {
  const health = healthConfig[target.health] || healthConfig.unknown;
  const HealthIcon = health.icon;

  const formatDuration = (seconds: number) => {
    if (seconds < 0.001) return '<1ms';
    if (seconds < 1) return `${Math.round(seconds * 1000)}ms`;
    return `${seconds.toFixed(2)}s`;
  };

  const formatLastScrape = (timestamp: number) => {
    const now = Date.now() / 1000;
    const diff = now - timestamp;
    if (diff < 60) return `${Math.floor(diff)}s ago`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    return new Date(timestamp * 1000).toLocaleTimeString();
  };

  return (
    <div className={`${health.bgColor} border border-slate-700 rounded-lg p-4`}>
      <div className="flex items-start justify-between">
        <div className="flex items-start gap-3">
          <Server className="w-5 h-5 text-slate-400 mt-0.5" />
          <div>
            <div className="flex items-center gap-2">
              <h4 className="font-medium text-white">{target.instance}</h4>
              <HealthIcon className={`w-4 h-4 ${health.color}`} />
            </div>
            <p className="text-sm text-slate-400">{target.job}</p>
            <p className="text-xs text-slate-500 mt-1 font-mono">{target.scrape_url}</p>
          </div>
        </div>
        <div className="text-right text-sm">
          <p className={health.color}>
            {target.health === 'up' ? 'Healthy' : target.health === 'down' ? 'Down' : 'Unknown'}
          </p>
          <p className="text-slate-400 text-xs mt-1">
            Last: {formatLastScrape(target.last_scrape)}
          </p>
          <p className="text-slate-500 text-xs">
            Duration: {formatDuration(target.last_scrape_duration)}
          </p>
        </div>
      </div>
      {target.last_error && (
        <div className="mt-3 p-2 bg-red-500/10 border border-red-500/30 rounded text-xs text-red-400">
          Error: {target.last_error}
        </div>
      )}
      {Object.entries(target.labels).length > 0 && (
        <div className="flex flex-wrap gap-1 mt-3">
          {Object.entries(target.labels).map(([key, value]) => (
            <span
              key={key}
              className="text-xs bg-slate-700 text-slate-300 px-2 py-0.5 rounded"
            >
              {key}={value}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

export function TargetsPanel() {
  const { targets, loading, error, fetchTargets } = useArgusStore();

  useEffect(() => {
    fetchTargets();
    // Refresh every 15 seconds
    const interval = setInterval(fetchTargets, 15000);
    return () => clearInterval(interval);
  }, [fetchTargets]);

  const upTargets = targets.filter((t) => t.health === 'up');
  const downTargets = targets.filter((t) => t.health === 'down');
  const unknownTargets = targets.filter((t) => t.health === 'unknown');

  // Group targets by job
  const targetsByJob = targets.reduce((acc, target) => {
    const job = target.job || 'unknown';
    if (!acc[job]) acc[job] = [];
    acc[job].push(target);
    return acc;
  }, {} as Record<string, Target[]>);

  if (loading && targets.length === 0) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-cyan-500" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-500/10 border border-red-500/50 rounded-lg p-4">
        <p className="text-red-400">Error loading targets: {error}</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Summary */}
      <div className="grid grid-cols-4 gap-4">
        <div className="bg-slate-800 border border-slate-700 rounded-lg p-4">
          <div className="flex items-center gap-2">
            <Server className="w-5 h-5 text-slate-400" />
            <span className="text-2xl font-bold text-white">{targets.length}</span>
          </div>
          <p className="text-sm text-slate-400 mt-1">Total Targets</p>
        </div>
        <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
          <div className="flex items-center gap-2">
            <CheckCircle className="w-5 h-5 text-green-400" />
            <span className="text-2xl font-bold text-green-400">{upTargets.length}</span>
          </div>
          <p className="text-sm text-slate-400 mt-1">Healthy</p>
        </div>
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4">
          <div className="flex items-center gap-2">
            <XCircle className="w-5 h-5 text-red-400" />
            <span className="text-2xl font-bold text-red-400">{downTargets.length}</span>
          </div>
          <p className="text-sm text-slate-400 mt-1">Down</p>
        </div>
        <div className="bg-slate-700/50 border border-slate-600 rounded-lg p-4">
          <div className="flex items-center gap-2">
            <HelpCircle className="w-5 h-5 text-slate-400" />
            <span className="text-2xl font-bold text-slate-400">{unknownTargets.length}</span>
          </div>
          <p className="text-sm text-slate-400 mt-1">Unknown</p>
        </div>
      </div>

      {/* Refresh button */}
      <div className="flex justify-end">
        <button
          onClick={() => fetchTargets()}
          disabled={loading}
          className="flex items-center gap-2 px-3 py-1.5 bg-slate-700 hover:bg-slate-600
                     text-slate-300 rounded-lg text-sm transition-colors disabled:opacity-50"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* Targets by Job */}
      {Object.entries(targetsByJob).map(([job, jobTargets]) => (
        <div key={job}>
          <h3 className="text-lg font-medium text-white mb-3 flex items-center gap-2">
            <span>{job}</span>
            <span className="text-sm text-slate-400">({jobTargets.length})</span>
          </h3>
          <div className="grid gap-3">
            {jobTargets.map((target) => (
              <TargetCard key={`${target.job}-${target.instance}`} target={target} />
            ))}
          </div>
        </div>
      ))}

      {targets.length === 0 && (
        <div className="text-center py-8">
          <Server className="w-12 h-12 text-slate-500 mx-auto mb-3" />
          <p className="text-slate-400">No scrape targets configured</p>
        </div>
      )}
    </div>
  );
}

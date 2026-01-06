import { useEffect, useState } from 'react';
import {
  Database,
  Code,
  Radio,
  CheckCircle,
  RefreshCw,
  Play,
  Pause,
  Square,
  Activity,
  Layers,
  Zap,
} from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, AreaChart, Area } from 'recharts';
import { useDataPipelineStore, PipelineStage, PipelineJob } from '../../stores/dataPipelineStore';

export default function DataProcessingView() {
  const {
    status,
    stats,
    stages,
    jobs,
    loading,
    error,
    fetchStatus,
    fetchStats,
    fetchStages,
    fetchJobs,
    simulateActivity,
  } = useDataPipelineStore();

  const [selectedTab, setSelectedTab] = useState<'overview' | 'pipelines' | 'stages' | 'jobs'>('overview');

  useEffect(() => {
    fetchStatus();
    fetchStats();
    fetchStages();
    fetchJobs();
    const interval = setInterval(() => {
      fetchStatus();
      fetchStats();
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="h-full flex flex-col bg-slate-900 text-white">
      {/* Header */}
      <div className="h-16 bg-slate-800 border-b border-slate-700 flex items-center px-6">
        <Database className="w-6 h-6 text-orange-400 mr-3" />
        <h1 className="text-xl font-semibold">Data Processing (RustySpark)</h1>
        <div className="flex-1" />
        <button
          onClick={simulateActivity}
          className="mr-2 px-3 py-1.5 bg-orange-600 hover:bg-orange-500 rounded-lg text-sm transition-colors"
        >
          Simulate Pipeline
        </button>
        <button onClick={() => { fetchStatus(); fetchStats(); }} className="p-2 hover:bg-slate-700 rounded-lg transition-colors">
          <RefreshCw className={`w-5 h-5 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Stats Bar */}
      {stats && (
        <div className="bg-slate-800/50 border-b border-slate-700 px-6 py-3 flex gap-6 text-sm">
          <div>
            <span className="text-slate-400">Active Pipelines:</span>{' '}
            <span className="text-orange-400 font-semibold">{stats.active_pipelines}</span>
          </div>
          <div>
            <span className="text-slate-400">Records Processed:</span>{' '}
            <span className="text-green-400">{stats.records_processed.toLocaleString()}</span>
          </div>
          <div>
            <span className="text-slate-400">Throughput:</span>{' '}
            <span className="text-blue-400">{stats.throughput_records_per_sec.toLocaleString()} rec/s</span>
          </div>
          <div>
            <span className="text-slate-400">GPU Accel:</span>{' '}
            <span className={stats.gpu_acceleration_enabled ? 'text-green-400' : 'text-slate-500'}>
              {stats.gpu_acceleration_enabled ? 'Enabled' : 'Disabled'}
            </span>
          </div>
        </div>
      )}

      {/* Tabs */}
      <div className="border-b border-slate-700 px-6">
        <div className="flex gap-4">
          {['overview', 'pipelines', 'stages', 'jobs'].map((tab) => (
            <button
              key={tab}
              onClick={() => setSelectedTab(tab as any)}
              className={`py-3 px-4 border-b-2 transition-colors capitalize ${
                selectedTab === tab ? 'border-orange-400 text-orange-400' : 'border-transparent text-slate-400 hover:text-white'
              }`}
            >
              {tab}
              {tab === 'jobs' && jobs.filter(j => j.status === 'running').length > 0 && (
                <span className="ml-2 bg-orange-500 text-white text-xs px-2 py-0.5 rounded-full">
                  {jobs.filter(j => j.status === 'running').length}
                </span>
              )}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto p-6">
        {error && <div className="mb-4 p-4 bg-red-500/20 border border-red-500 rounded-lg text-red-400">{error}</div>}

        {selectedTab === 'overview' && stats && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Throughput Chart */}
            <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
              <h3 className="font-medium mb-4">Throughput Over Time</h3>
              <ResponsiveContainer width="100%" height={250}>
                <AreaChart data={generateThroughputData()}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="time" stroke="#94a3b8" fontSize={12} />
                  <YAxis stroke="#94a3b8" fontSize={12} tickFormatter={(v) => `${v/1000}k`} />
                  <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none' }} formatter={(v: number) => [`${v.toLocaleString()} rec/s`, 'Throughput']} />
                  <Area type="monotone" dataKey="throughput" stroke="#f97316" fill="#f97316" fillOpacity={0.3} />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            {/* Stage Distribution */}
            <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
              <h3 className="font-medium mb-4">Records by Stage</h3>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={stages}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="name" stroke="#94a3b8" fontSize={12} />
                  <YAxis stroke="#94a3b8" fontSize={12} tickFormatter={(v) => `${v/1000}k`} />
                  <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none' }} />
                  <Bar dataKey="records_in" fill="#10b981" name="Records In" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="records_out" fill="#3b82f6" name="Records Out" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Stats Grid */}
            <div className="lg:col-span-2 grid grid-cols-2 md:grid-cols-4 gap-4">
              <StatCard
                icon={Layers}
                label="Total Stages"
                value={stages.length.toString()}
                color="text-orange-400"
              />
              <StatCard
                icon={Activity}
                label="Avg Latency"
                value={`${stats.average_latency_ms.toFixed(1)}ms`}
                color="text-blue-400"
              />
              <StatCard
                icon={Zap}
                label="GPU Speedup"
                value={`${stats.gpu_speedup_factor.toFixed(1)}x`}
                color="text-green-400"
              />
              <StatCard
                icon={Database}
                label="Data Processed"
                value={formatBytes(stats.bytes_processed)}
                color="text-purple-400"
              />
            </div>
          </div>
        )}

        {selectedTab === 'pipelines' && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {generatePipelineData().map((pipeline) => (
              <PipelineCard key={pipeline.id} pipeline={pipeline} />
            ))}
          </div>
        )}

        {selectedTab === 'stages' && (
          <div className="space-y-4">
            {stages.map((stage) => (
              <StageCard key={stage.id} stage={stage} />
            ))}
            {stages.length === 0 && (
              <div className="p-8 text-center text-slate-500 bg-slate-800 rounded-lg border border-slate-700">
                No pipeline stages active
              </div>
            )}
          </div>
        )}

        {selectedTab === 'jobs' && (
          <div className="space-y-4">
            {jobs.map((job) => (
              <JobCard key={job.id} job={job} />
            ))}
            {jobs.length === 0 && (
              <div className="p-8 text-center text-slate-500 bg-slate-800 rounded-lg border border-slate-700">
                No pipeline jobs found
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function StatCard({ icon: Icon, label, value, color }: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: string;
  color: string;
}) {
  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
      <div className="flex items-center gap-3 mb-2">
        <Icon className={`w-5 h-5 ${color}`} />
        <span className="text-slate-400 text-sm">{label}</span>
      </div>
      <div className="text-2xl font-bold">{value}</div>
    </div>
  );
}

function PipelineCard({ pipeline }: { pipeline: { id: string; name: string; status: string; stages: number; throughput: number; records_processed: number } }) {
  const statusColors: Record<string, string> = {
    running: 'bg-green-500/20 text-green-400',
    paused: 'bg-amber-500/20 text-amber-400',
    stopped: 'bg-slate-500/20 text-slate-400',
    failed: 'bg-red-500/20 text-red-400',
  };

  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <Database className="w-5 h-5 text-orange-400" />
          <div>
            <div className="font-medium">{pipeline.name}</div>
            <div className="text-sm text-slate-400">{pipeline.stages} stages</div>
          </div>
        </div>
        <span className={`px-3 py-1 rounded-full text-sm ${statusColors[pipeline.status]}`}>
          {pipeline.status}
        </span>
      </div>
      <div className="grid grid-cols-2 gap-4 text-sm">
        <div>
          <span className="text-slate-400">Throughput</span>
          <div className="font-medium">{pipeline.throughput.toLocaleString()} rec/s</div>
        </div>
        <div>
          <span className="text-slate-400">Processed</span>
          <div className="font-medium">{pipeline.records_processed.toLocaleString()}</div>
        </div>
      </div>
      <div className="mt-3 flex gap-2">
        {pipeline.status === 'running' ? (
          <button className="px-3 py-1 bg-amber-600 hover:bg-amber-500 rounded text-sm">Pause</button>
        ) : (
          <button className="px-3 py-1 bg-green-600 hover:bg-green-500 rounded text-sm">Start</button>
        )}
        <button className="px-3 py-1 bg-slate-600 hover:bg-slate-500 rounded text-sm">Configure</button>
      </div>
    </div>
  );
}

function StageCard({ stage }: { stage: PipelineStage }) {
  const healthColors: Record<string, string> = {
    healthy: 'text-green-400',
    degraded: 'text-amber-400',
    unhealthy: 'text-red-400',
  };

  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-3">
          <Layers className="w-5 h-5 text-orange-400" />
          <div>
            <div className="font-medium">{stage.name}</div>
            <div className="text-sm text-slate-400">{stage.stage_type}</div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className={`${healthColors[stage.health]}`}>
            {stage.health === 'healthy' ? <CheckCircle className="w-5 h-5" /> : <Radio className="w-5 h-5" />}
          </span>
        </div>
      </div>
      <div className="grid grid-cols-4 gap-4 text-sm">
        <div>
          <span className="text-slate-400">Records In</span>
          <div className="font-medium">{stage.records_in.toLocaleString()}</div>
        </div>
        <div>
          <span className="text-slate-400">Records Out</span>
          <div className="font-medium">{stage.records_out.toLocaleString()}</div>
        </div>
        <div>
          <span className="text-slate-400">Latency</span>
          <div className="font-medium">{stage.latency_ms.toFixed(1)}ms</div>
        </div>
        <div>
          <span className="text-slate-400">Errors</span>
          <div className={stage.errors > 0 ? 'font-medium text-red-400' : 'font-medium'}>{stage.errors}</div>
        </div>
      </div>
    </div>
  );
}

function JobCard({ job }: { job: PipelineJob }) {
  const statusColors: Record<string, string> = {
    pending: 'bg-slate-500/20 text-slate-400',
    running: 'bg-green-500/20 text-green-400',
    completed: 'bg-blue-500/20 text-blue-400',
    failed: 'bg-red-500/20 text-red-400',
  };

  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
      <div className="flex items-center justify-between mb-3">
        <div>
          <div className="font-medium">{job.name}</div>
          <div className="text-sm text-slate-400">{job.pipeline_id}</div>
        </div>
        <span className={`px-3 py-1 rounded-full text-sm ${statusColors[job.status]}`}>
          {job.status}
        </span>
      </div>
      {job.status === 'running' && (
        <div className="mb-3">
          <div className="flex justify-between text-sm text-slate-400 mb-1">
            <span>Progress</span>
            <span>{job.progress.toFixed(1)}%</span>
          </div>
          <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
            <div className="h-full bg-orange-500 transition-all" style={{ width: `${job.progress}%` }} />
          </div>
        </div>
      )}
      <div className="grid grid-cols-3 gap-4 text-sm">
        <div>
          <span className="text-slate-400">Records</span>
          <div className="font-medium">{job.records_processed.toLocaleString()}</div>
        </div>
        <div>
          <span className="text-slate-400">Started</span>
          <div className="font-medium">{new Date(job.started_at).toLocaleTimeString()}</div>
        </div>
        <div>
          <span className="text-slate-400">Duration</span>
          <div className="font-medium">{job.duration_seconds ? `${job.duration_seconds}s` : 'Running...'}</div>
        </div>
      </div>
    </div>
  );
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

function generateThroughputData() {
  const times = ['5m', '10m', '15m', '20m', '25m', '30m', '35m', '40m', '45m', '50m', '55m', '60m'];
  return times.map(time => ({
    time,
    throughput: 8000 + Math.random() * 4000,
  }));
}

function generatePipelineData() {
  return [
    { id: '1', name: 'ETL-Main', status: 'running', stages: 5, throughput: 12500, records_processed: 45000000 },
    { id: '2', name: 'Stream-Kafka', status: 'running', stages: 3, throughput: 8200, records_processed: 12000000 },
    { id: '3', name: 'Batch-Daily', status: 'paused', stages: 7, throughput: 0, records_processed: 89000000 },
    { id: '4', name: 'ML-Features', status: 'running', stages: 4, throughput: 5600, records_processed: 7800000 },
  ];
}

import { useEffect, useState } from 'react';
import {
  Brain,
  Network,
  Database,
  BarChart3,
  Play,
  Pause,
  Square,
  RefreshCw,
  Plus,
  CheckCircle,
  XCircle,
  Clock,
  Cpu,
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { useTrainingStore, TrainingJob, TrainingMetrics } from '../../stores/trainingStore';

export default function TrainingView() {
  const {
    jobs,
    summary,
    loading,
    error,
    fetchJobs,
    fetchSummary,
    pauseJob,
    resumeJob,
    cancelJob,
  } = useTrainingStore();

  const [selectedTab, setSelectedTab] = useState<'jobs' | 'metrics' | 'models'>('jobs');
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);

  useEffect(() => {
    fetchJobs();
    fetchSummary();
    const interval = setInterval(() => {
      fetchJobs();
      fetchSummary();
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  const selectedJob = jobs.find(j => j.id === selectedJobId);

  return (
    <div className="h-full flex flex-col bg-slate-900 text-white">
      {/* Header */}
      <div className="h-16 bg-slate-800 border-b border-slate-700 flex items-center px-6">
        <Brain className="w-6 h-6 text-pink-400 mr-3" />
        <h1 className="text-xl font-semibold">ML Training (RustyTorch)</h1>
        <div className="flex-1" />
        <button className="mr-2 px-3 py-1.5 bg-pink-600 hover:bg-pink-500 rounded-lg text-sm transition-colors flex items-center gap-2">
          <Plus className="w-4 h-4" />
          New Job
        </button>
        <button onClick={() => { fetchJobs(); fetchSummary(); }} className="p-2 hover:bg-slate-700 rounded-lg transition-colors">
          <RefreshCw className={`w-5 h-5 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Summary Bar */}
      {summary && (
        <div className="bg-slate-800/50 border-b border-slate-700 px-6 py-3 flex gap-6 text-sm">
          <div>
            <span className="text-slate-400">Active Jobs:</span>{' '}
            <span className="text-pink-400 font-semibold">{summary.active_jobs}</span>
          </div>
          <div>
            <span className="text-slate-400">Completed:</span>{' '}
            <span className="text-green-400">{summary.completed_jobs}</span>
          </div>
          <div>
            <span className="text-slate-400">Failed:</span>{' '}
            <span className="text-red-400">{summary.failed_jobs}</span>
          </div>
          <div>
            <span className="text-slate-400">GPUs Used:</span>{' '}
            <span className="text-amber-400">{summary.gpus_used} / {summary.gpus_total}</span>
          </div>
          <div>
            <span className="text-slate-400">Total Epochs:</span>{' '}
            <span className="text-white">{summary.total_epochs_completed.toLocaleString()}</span>
          </div>
        </div>
      )}

      {/* Tabs */}
      <div className="border-b border-slate-700 px-6">
        <div className="flex gap-4">
          {['jobs', 'metrics', 'models'].map((tab) => (
            <button
              key={tab}
              onClick={() => setSelectedTab(tab as any)}
              className={`py-3 px-4 border-b-2 transition-colors capitalize ${
                selectedTab === tab ? 'border-pink-400 text-pink-400' : 'border-transparent text-slate-400 hover:text-white'
              }`}
            >
              {tab}
              {tab === 'jobs' && jobs.filter(j => j.status === 'running').length > 0 && (
                <span className="ml-2 bg-pink-500 text-white text-xs px-2 py-0.5 rounded-full">
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

        {selectedTab === 'jobs' && (
          <div className="flex gap-6 h-full">
            {/* Jobs List */}
            <div className="w-1/2 space-y-4">
              {jobs.map((job) => (
                <JobCard
                  key={job.id}
                  job={job}
                  selected={selectedJobId === job.id}
                  onSelect={() => setSelectedJobId(job.id)}
                  onPause={() => pauseJob(job.id)}
                  onResume={() => resumeJob(job.id)}
                  onCancel={() => cancelJob(job.id)}
                />
              ))}
              {jobs.length === 0 && (
                <div className="p-8 text-center text-slate-500 bg-slate-800 rounded-lg border border-slate-700">
                  No training jobs. Click "New Job" to start training.
                </div>
              )}
            </div>

            {/* Job Details */}
            <div className="w-1/2">
              {selectedJob ? (
                <JobDetails job={selectedJob} />
              ) : (
                <div className="h-full flex items-center justify-center text-slate-500 bg-slate-800 rounded-lg border border-slate-700">
                  Select a job to view details
                </div>
              )}
            </div>
          </div>
        )}

        {selectedTab === 'metrics' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
              <h3 className="font-medium mb-4">Loss Over Time (All Jobs)</h3>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={generateLossData()}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="epoch" stroke="#94a3b8" fontSize={12} />
                  <YAxis stroke="#94a3b8" fontSize={12} />
                  <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none' }} />
                  <Line type="monotone" dataKey="train_loss" stroke="#ec4899" strokeWidth={2} name="Train Loss" />
                  <Line type="monotone" dataKey="val_loss" stroke="#8b5cf6" strokeWidth={2} name="Val Loss" />
                </LineChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
              <h3 className="font-medium mb-4">GPU Utilization</h3>
              <ResponsiveContainer width="100%" height={250}>
                <AreaChart data={generateGpuData()}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="time" stroke="#94a3b8" fontSize={12} />
                  <YAxis stroke="#94a3b8" fontSize={12} domain={[0, 100]} tickFormatter={(v) => `${v}%`} />
                  <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none' }} />
                  <Area type="monotone" dataKey="utilization" stroke="#10b981" fill="#10b981" fillOpacity={0.3} />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            <div className="lg:col-span-2 bg-slate-800 rounded-lg border border-slate-700 p-4">
              <h3 className="font-medium mb-4">Training Throughput</h3>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={generateThroughputData()}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="time" stroke="#94a3b8" fontSize={12} />
                  <YAxis stroke="#94a3b8" fontSize={12} tickFormatter={(v) => `${v} samples/s`} />
                  <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none' }} />
                  <Line type="monotone" dataKey="throughput" stroke="#f59e0b" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {selectedTab === 'models' && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {generateModelData().map((model) => (
              <ModelCard key={model.id} model={model} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function JobCard({
  job,
  selected,
  onSelect,
  onPause,
  onResume,
  onCancel,
}: {
  job: TrainingJob;
  selected: boolean;
  onSelect: () => void;
  onPause: () => void;
  onResume: () => void;
  onCancel: () => void;
}) {
  const statusColors: Record<string, string> = {
    pending: 'text-slate-400',
    running: 'text-green-400',
    paused: 'text-amber-400',
    completed: 'text-blue-400',
    failed: 'text-red-400',
    cancelled: 'text-slate-500',
  };

  const statusIcons: Record<string, React.ReactNode> = {
    pending: <Clock className="w-4 h-4" />,
    running: <Play className="w-4 h-4" />,
    paused: <Pause className="w-4 h-4" />,
    completed: <CheckCircle className="w-4 h-4" />,
    failed: <XCircle className="w-4 h-4" />,
    cancelled: <Square className="w-4 h-4" />,
  };

  return (
    <div
      onClick={onSelect}
      className={`bg-slate-800 rounded-lg border p-4 cursor-pointer transition-colors ${
        selected ? 'border-pink-400' : 'border-slate-700 hover:border-slate-600'
      }`}
    >
      <div className="flex items-center justify-between mb-3">
        <div>
          <div className="font-medium">{job.name}</div>
          <div className="text-sm text-slate-400">{job.model_type}</div>
        </div>
        <div className="flex items-center gap-2">
          <span className={`flex items-center gap-1 text-sm ${statusColors[job.status]}`}>
            {statusIcons[job.status]}
            {job.status}
          </span>
        </div>
      </div>

      {/* Progress Bar */}
      {job.status === 'running' && (
        <div className="mb-3">
          <div className="flex justify-between text-sm text-slate-400 mb-1">
            <span>Epoch {job.current_epoch}/{job.total_epochs}</span>
            <span>{job.progress.toFixed(1)}%</span>
          </div>
          <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
            <div className="h-full bg-pink-500 transition-all" style={{ width: `${job.progress}%` }} />
          </div>
        </div>
      )}

      <div className="flex justify-between items-center text-sm">
        <div className="text-slate-400">
          <Cpu className="w-4 h-4 inline mr-1" />
          {job.gpus_allocated} GPUs
        </div>
        {job.status === 'running' && (
          <div className="flex gap-2">
            <button onClick={(e) => { e.stopPropagation(); onPause(); }} className="p-1 hover:bg-slate-700 rounded">
              <Pause className="w-4 h-4" />
            </button>
            <button onClick={(e) => { e.stopPropagation(); onCancel(); }} className="p-1 hover:bg-slate-700 rounded text-red-400">
              <Square className="w-4 h-4" />
            </button>
          </div>
        )}
        {job.status === 'paused' && (
          <button onClick={(e) => { e.stopPropagation(); onResume(); }} className="p-1 hover:bg-slate-700 rounded text-green-400">
            <Play className="w-4 h-4" />
          </button>
        )}
      </div>
    </div>
  );
}

function JobDetails({ job }: { job: TrainingJob }) {
  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 p-4 h-full overflow-auto">
      <h3 className="font-medium mb-4">{job.name}</h3>

      <div className="grid grid-cols-2 gap-4 mb-6">
        <div>
          <span className="text-slate-400 text-sm">Model Type</span>
          <div className="font-medium">{job.model_type}</div>
        </div>
        <div>
          <span className="text-slate-400 text-sm">Dataset</span>
          <div className="font-medium">{job.dataset}</div>
        </div>
        <div>
          <span className="text-slate-400 text-sm">Batch Size</span>
          <div className="font-medium">{job.batch_size}</div>
        </div>
        <div>
          <span className="text-slate-400 text-sm">Learning Rate</span>
          <div className="font-medium">{job.learning_rate}</div>
        </div>
        <div>
          <span className="text-slate-400 text-sm">Current Loss</span>
          <div className="font-medium text-amber-400">{job.current_loss?.toFixed(4) || 'N/A'}</div>
        </div>
        <div>
          <span className="text-slate-400 text-sm">Best Loss</span>
          <div className="font-medium text-green-400">{job.best_loss?.toFixed(4) || 'N/A'}</div>
        </div>
      </div>

      {job.metrics && job.metrics.length > 0 && (
        <div>
          <h4 className="text-sm text-slate-400 mb-2">Training Curve</h4>
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={job.metrics}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis dataKey="epoch" stroke="#94a3b8" fontSize={12} />
              <YAxis stroke="#94a3b8" fontSize={12} />
              <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none' }} />
              <Line type="monotone" dataKey="train_loss" stroke="#ec4899" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="val_loss" stroke="#8b5cf6" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

function ModelCard({ model }: { model: { id: string; name: string; version: string; accuracy: number; size_mb: number; created_at: string } }) {
  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
      <div className="flex items-center gap-3 mb-3">
        <Database className="w-5 h-5 text-pink-400" />
        <div>
          <div className="font-medium">{model.name}</div>
          <div className="text-sm text-slate-400">v{model.version}</div>
        </div>
      </div>
      <div className="grid grid-cols-2 gap-2 text-sm">
        <div>
          <span className="text-slate-400">Accuracy</span>
          <div className="text-green-400">{(model.accuracy * 100).toFixed(1)}%</div>
        </div>
        <div>
          <span className="text-slate-400">Size</span>
          <div>{model.size_mb} MB</div>
        </div>
      </div>
      <div className="mt-2 text-xs text-slate-500">
        Created {new Date(model.created_at).toLocaleDateString()}
      </div>
    </div>
  );
}

// Helper functions to generate mock chart data
function generateLossData() {
  return Array.from({ length: 20 }, (_, i) => ({
    epoch: i + 1,
    train_loss: 2.5 * Math.exp(-0.1 * i) + 0.1 + Math.random() * 0.05,
    val_loss: 2.5 * Math.exp(-0.08 * i) + 0.15 + Math.random() * 0.08,
  }));
}

function generateGpuData() {
  const times = ['00:00', '02:00', '04:00', '06:00', '08:00', '10:00', '12:00', '14:00', '16:00', '18:00', '20:00', '22:00'];
  return times.map(time => ({
    time,
    utilization: 70 + Math.random() * 25,
  }));
}

function generateThroughputData() {
  const times = ['5m', '10m', '15m', '20m', '25m', '30m'];
  return times.map(time => ({
    time,
    throughput: 1500 + Math.random() * 500,
  }));
}

function generateModelData() {
  return [
    { id: '1', name: 'ResNet-152', version: '2.1.0', accuracy: 0.943, size_mb: 234, created_at: '2024-12-15' },
    { id: '2', name: 'GPT-Mini', version: '1.0.0', accuracy: 0.891, size_mb: 1240, created_at: '2024-12-10' },
    { id: '3', name: 'BERT-Base', version: '3.0.1', accuracy: 0.912, size_mb: 440, created_at: '2024-12-05' },
    { id: '4', name: 'ViT-Large', version: '1.2.0', accuracy: 0.956, size_mb: 890, created_at: '2024-11-28' },
  ];
}

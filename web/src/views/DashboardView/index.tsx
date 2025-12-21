import { useEffect } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LineChart,
  Line,
} from 'recharts';
import {
  Activity,
  DollarSign,
  Zap,
  Clock,
  TrendingUp,
  AlertCircle,
} from 'lucide-react';
import { useClusterStore } from '../../stores/clusterStore';
import { useTrainingStore } from '../../stores/trainingStore';
import SystemInfoPanel from '../../components/SystemInfoPanel';
import RealTimeMetrics from '../../components/RealTimeMetrics';

// Mock data for visualization
const gpuUtilization = [
  { name: 'gpu-server-1', utilization: 87 },
  { name: 'gpu-server-2', utilization: 92 },
  { name: 'workstation', utilization: 45 },
];

const costData = [
  { name: 'Mon', cost: 124 },
  { name: 'Tue', cost: 156 },
  { name: 'Wed', cost: 189 },
  { name: 'Thu', cost: 145 },
  { name: 'Fri', cost: 178 },
  { name: 'Sat', cost: 89 },
  { name: 'Sun', cost: 67 },
];

const evolutionEvents = [
  { time: '10:34', event: 'DGM improved model checkpoint efficiency by 12%' },
  { time: '09:15', event: 'SwarmAgentic rebalanced GPU workloads across 3 nodes' },
  { time: '08:45', event: 'ADAS discovered optimal batch size for dataset X' },
  { time: 'Yesterday', event: 'Behavioral learning prevented 2 predicted failures' },
];

export default function DashboardView() {
  const { connected, nodeCount, healthyNodes } = useClusterStore();
  const { jobs, fetchJobs } = useTrainingStore();

  useEffect(() => {
    fetchJobs();
  }, [fetchJobs]);

  const activeJobs = jobs.filter((j) => j.status === 'running').length;
  const completedJobs = jobs.filter((j) => j.status === 'completed').length;

  return (
    <div className="h-full overflow-auto bg-slate-900">
      {/* Header */}
      <div className="h-16 bg-slate-800 border-b border-slate-700 flex items-center px-6">
        <h1 className="text-xl font-semibold">Executive Dashboard</h1>
        <div className="flex-1" />
        <div className="text-sm text-slate-400">
          Last updated: {new Date().toLocaleTimeString()}
        </div>
      </div>

      <div className="p-6 space-y-6">
        {/* Top Stats Row */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricCard
            icon={Activity}
            label="Cluster Health"
            value={connected ? `${Math.round((healthyNodes / nodeCount) * 100)}%` : 'N/A'}
            trend={connected ? '+2%' : undefined}
            trendUp={true}
            color="text-green-400"
          />
          <MetricCard
            icon={Zap}
            label="Active Training Jobs"
            value={activeJobs.toString()}
            trend={completedJobs > 0 ? `${completedJobs} completed` : undefined}
            color="text-blue-400"
          />
          <MetricCard
            icon={DollarSign}
            label="Weekly Compute Cost"
            value="$948"
            trend="-8% vs last week"
            trendUp={false}
            color="text-purple-400"
          />
          <MetricCard
            icon={Clock}
            label="Avg. Training Time"
            value="2.4h"
            trend="-15% improvement"
            trendUp={false}
            color="text-amber-400"
          />
        </div>

        {/* Charts Row */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* GPU Utilization */}
          <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
            <h3 className="font-medium mb-4">GPU Utilization</h3>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={gpuUtilization}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="name" stroke="#94a3b8" fontSize={12} />
                <YAxis stroke="#94a3b8" fontSize={12} domain={[0, 100]} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1e293b', border: 'none' }}
                />
                <Bar dataKey="utilization" fill="#3b82f6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Cost Trend */}
          <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
            <h3 className="font-medium mb-4">Daily Compute Cost</h3>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={costData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="name" stroke="#94a3b8" fontSize={12} />
                <YAxis stroke="#94a3b8" fontSize={12} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1e293b', border: 'none' }}
                  formatter={(value) => [`$${value}`, 'Cost']}
                />
                <Line
                  type="monotone"
                  dataKey="cost"
                  stroke="#a855f7"
                  strokeWidth={2}
                  dot={{ fill: '#a855f7' }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Bottom Row */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Real-Time Metrics */}
          <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
            <RealTimeMetrics />
          </div>

          {/* Evolution Insights */}
          <div className="lg:col-span-2 bg-slate-800 rounded-lg border border-slate-700 p-4">
            <div className="flex items-center gap-2 mb-4">
              <TrendingUp className="w-4 h-4 text-green-400" />
              <h3 className="font-medium">Evolution Insights</h3>
              <span className="text-xs bg-green-500/20 text-green-400 px-2 py-0.5 rounded">
                Self-improving
              </span>
            </div>
            <div className="space-y-3">
              {evolutionEvents.map((event, i) => (
                <div
                  key={i}
                  className="flex items-start gap-3 p-3 bg-slate-700/50 rounded-lg"
                >
                  <div className="w-12 text-xs text-slate-400 flex-shrink-0">
                    {event.time}
                  </div>
                  <div className="text-sm">{event.event}</div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Bottom Section: Local System + Incidents */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Local System Info */}
          <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
            <SystemInfoPanel />
          </div>

          {/* Active Incidents */}
          <div className="lg:col-span-2 bg-slate-800 rounded-lg border border-slate-700 p-4">
            <div className="flex items-center gap-2 mb-4">
              <AlertCircle className="w-4 h-4 text-amber-400" />
              <h3 className="font-medium">Active Incidents</h3>
            </div>
            <div className="text-sm text-slate-400 text-center py-8">
              No active incidents. System operating normally.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

interface MetricCardProps {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: string;
  trend?: string;
  trendUp?: boolean;
  color?: string;
}

function MetricCard({
  icon: Icon,
  label,
  value,
  trend,
  trendUp,
  color = 'text-white',
}: MetricCardProps) {
  return (
    <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
      <div className="flex items-center gap-2 text-slate-400 text-sm mb-2">
        <Icon className={`w-4 h-4 ${color}`} />
        {label}
      </div>
      <div className={`text-3xl font-bold ${color}`}>{value}</div>
      {trend && (
        <div
          className={`text-xs mt-1 ${
            trendUp === undefined
              ? 'text-slate-400'
              : trendUp
              ? 'text-green-400'
              : 'text-green-400'
          }`}
        >
          {trend}
        </div>
      )}
    </div>
  );
}

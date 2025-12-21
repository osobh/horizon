/**
 * RealTimeMetrics - Displays live system metrics from Tauri events
 *
 * Shows CPU, memory, and GPU usage with real-time charts.
 */

import { useEffect, useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  ResponsiveContainer,
  Tooltip,
} from 'recharts';
import { Activity, Cpu, HardDrive, Zap } from 'lucide-react';
import { useMetrics } from '../hooks/useMetrics';

interface MetricGaugeProps {
  label: string;
  value: number;
  icon: React.ComponentType<{ className?: string }>;
  color: string;
  unit?: string;
}

function MetricGauge({ label, value, icon: Icon, color, unit = '%' }: MetricGaugeProps) {
  const displayValue = Math.round(value);
  const barWidth = Math.min(100, Math.max(0, value));

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-sm">
        <div className="flex items-center gap-2 text-slate-400">
          <Icon className={`w-4 h-4 ${color}`} />
          {label}
        </div>
        <span className={`font-mono ${color}`}>
          {displayValue}{unit}
        </span>
      </div>
      <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${
            value > 90 ? 'bg-red-500' :
            value > 70 ? 'bg-yellow-500' :
            color.replace('text-', 'bg-')
          }`}
          style={{ width: `${barWidth}%` }}
        />
      </div>
    </div>
  );
}

interface ChartDataPoint {
  time: string;
  cpu: number;
  memory: number;
  gpu: number;
}

export default function RealTimeMetrics() {
  const { metrics, history } = useMetrics();
  const [chartData, setChartData] = useState<ChartDataPoint[]>([]);

  // Transform history into chart data
  useEffect(() => {
    const data = history.map((m, i) => ({
      time: i.toString(),
      cpu: Math.round(m.cpu_usage),
      memory: Math.round(m.memory_usage),
      gpu: m.gpu_usage.length > 0 ? Math.round(m.gpu_usage[0]) : 0,
    }));
    setChartData(data);
  }, [history]);

  // If no metrics yet, show placeholder
  if (!metrics) {
    return (
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <Activity className="w-5 h-5 text-slate-400" />
          <h3 className="font-medium">Real-Time Metrics</h3>
          <span className="text-xs text-slate-500 ml-auto">Waiting for data...</span>
        </div>
        <div className="h-32 flex items-center justify-center text-slate-500">
          <div className="animate-pulse flex items-center gap-2">
            <div className="w-4 h-4 border-2 border-slate-500 border-t-transparent rounded-full animate-spin" />
            Connecting...
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center gap-2">
        <Activity className="w-5 h-5 text-green-400" />
        <h3 className="font-medium">Real-Time Metrics</h3>
        <span className="flex items-center gap-1 text-xs text-green-400 ml-auto">
          <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
          Live
        </span>
      </div>

      {/* Current Values */}
      <div className="space-y-3">
        <MetricGauge
          label="CPU"
          value={metrics.cpu_usage}
          icon={Cpu}
          color="text-blue-400"
        />
        <MetricGauge
          label="Memory"
          value={metrics.memory_usage}
          icon={HardDrive}
          color="text-purple-400"
        />
        {metrics.gpu_usage.length > 0 && (
          <MetricGauge
            label="GPU"
            value={metrics.gpu_usage[0]}
            icon={Zap}
            color="text-amber-400"
          />
        )}
      </div>

      {/* Mini Chart */}
      {chartData.length > 5 && (
        <div className="pt-2">
          <div className="text-xs text-slate-500 mb-2">Last 60 seconds</div>
          <ResponsiveContainer width="100%" height={80}>
            <LineChart data={chartData}>
              <XAxis dataKey="time" hide />
              <YAxis domain={[0, 100]} hide />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1e293b',
                  border: '1px solid #334155',
                  borderRadius: '0.5rem',
                }}
                formatter={(value: number, name: string) => [
                  `${value}%`,
                  name.toUpperCase(),
                ]}
                labelFormatter={() => ''}
              />
              <Line
                type="monotone"
                dataKey="cpu"
                stroke="#60a5fa"
                strokeWidth={1.5}
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="memory"
                stroke="#a78bfa"
                strokeWidth={1.5}
                dot={false}
              />
              {chartData[0]?.gpu > 0 && (
                <Line
                  type="monotone"
                  dataKey="gpu"
                  stroke="#fbbf24"
                  strokeWidth={1.5}
                  dot={false}
                />
              )}
            </LineChart>
          </ResponsiveContainer>
          <div className="flex justify-center gap-4 text-xs mt-1">
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-blue-400" />
              CPU
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-purple-400" />
              Memory
            </span>
            {chartData[0]?.gpu > 0 && (
              <span className="flex items-center gap-1">
                <span className="w-2 h-2 rounded-full bg-amber-400" />
                GPU
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

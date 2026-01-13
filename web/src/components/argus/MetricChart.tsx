import { useMemo } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { QueryResult } from '../../stores/argusStore';

interface MetricChartProps {
  result: QueryResult;
  height?: number;
}

const COLORS = [
  '#06b6d4', // cyan
  '#8b5cf6', // violet
  '#22c55e', // green
  '#f59e0b', // amber
  '#ef4444', // red
  '#ec4899', // pink
  '#3b82f6', // blue
  '#14b8a6', // teal
];

export function MetricChart({ result, height = 300 }: MetricChartProps) {
  const { chartData, seriesNames } = useMemo(() => {
    if (!result || result.data.length === 0) {
      return { chartData: [], seriesNames: [] };
    }

    // For matrix (range query) results
    if (result.result_type === 'matrix') {
      const seriesNames: string[] = [];
      const timeMap = new Map<number, Record<string, number>>();

      result.data.forEach((sample, idx) => {
        // Create series name from labels
        const labelStr = Object.entries(sample.metric)
          .filter(([k]) => k !== '__name__')
          .map(([k, v]) => `${k}="${v}"`)
          .join(', ');
        const name = sample.metric.__name__
          ? `${sample.metric.__name__}{${labelStr}}`
          : labelStr || `series_${idx}`;
        seriesNames.push(name);

        // Add values to time map
        sample.values.forEach(([timestamp, value]) => {
          const ts = Math.floor(timestamp);
          if (!timeMap.has(ts)) {
            timeMap.set(ts, { timestamp: ts });
          }
          timeMap.get(ts)![name] = parseFloat(value);
        });
      });

      // Sort by timestamp and convert to array
      const chartData = Array.from(timeMap.values()).sort(
        (a, b) => a.timestamp - b.timestamp
      );

      return { chartData, seriesNames };
    }

    // For vector (instant query) results
    if (result.result_type === 'vector') {
      const chartData = result.data.map((sample, idx) => {
        const labelStr = Object.entries(sample.metric)
          .filter(([k]) => k !== '__name__')
          .map(([k, v]) => `${k}="${v}"`)
          .join(', ');
        const name = sample.metric.__name__ || labelStr || `metric_${idx}`;
        return {
          name,
          value: sample.value ? parseFloat(sample.value[1]) : 0,
        };
      });

      return { chartData, seriesNames: ['value'] };
    }

    return { chartData: [], seriesNames: [] };
  }, [result]);

  const formatTimestamp = (ts: number) => {
    const date = new Date(ts * 1000);
    return date.toLocaleTimeString();
  };

  const formatValue = (value: number) => {
    if (Math.abs(value) >= 1e9) return `${(value / 1e9).toFixed(2)}G`;
    if (Math.abs(value) >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
    if (Math.abs(value) >= 1e3) return `${(value / 1e3).toFixed(2)}K`;
    return value.toFixed(2);
  };

  if (chartData.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 bg-slate-800 rounded-lg">
        <p className="text-slate-400">No data to display</p>
      </div>
    );
  }

  // Render bar chart for instant vectors
  if (result.result_type === 'vector') {
    return (
      <div className="bg-slate-800 rounded-lg p-4">
        <div className="space-y-2">
          {chartData.map((item, idx) => (
            <div key={idx} className="flex items-center gap-3">
              <div className="flex-1">
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-slate-300 truncate max-w-xs">{item.name}</span>
                  <span className="text-cyan-400 font-mono">{formatValue(item.value)}</span>
                </div>
                <div className="h-2 bg-slate-700 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-cyan-500 rounded-full transition-all"
                    style={{
                      width: `${Math.min(100, (item.value / Math.max(...chartData.map((d) => d.value))) * 100)}%`,
                    }}
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  // Render line chart for matrix results
  return (
    <div className="bg-slate-800 rounded-lg p-4">
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={chartData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis
            dataKey="timestamp"
            tickFormatter={formatTimestamp}
            stroke="#64748b"
            tick={{ fill: '#94a3b8', fontSize: 12 }}
          />
          <YAxis
            tickFormatter={formatValue}
            stroke="#64748b"
            tick={{ fill: '#94a3b8', fontSize: 12 }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#1e293b',
              border: '1px solid #334155',
              borderRadius: '8px',
            }}
            labelFormatter={(ts) => new Date(ts * 1000).toLocaleString()}
            formatter={(value: number) => [formatValue(value), '']}
          />
          {seriesNames.length > 1 && <Legend />}
          {seriesNames.map((name, idx) => (
            <Line
              key={name}
              type="monotone"
              dataKey={name}
              stroke={COLORS[idx % COLORS.length]}
              strokeWidth={2}
              dot={false}
              name={name.length > 30 ? name.slice(0, 30) + '...' : name}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

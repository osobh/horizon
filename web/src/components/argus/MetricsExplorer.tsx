import { useState } from 'react';
import { Search, Play, Clock, BarChart3 } from 'lucide-react';
import { useArgusStore } from '../../stores/argusStore';
import { MetricChart } from './MetricChart';

const COMMON_QUERIES = [
  { label: 'CPU Usage', query: 'rate(node_cpu_seconds_total{mode!="idle"}[5m]) * 100' },
  { label: 'Memory Usage', query: '(1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100' },
  { label: 'Disk Usage', query: '(1 - node_filesystem_avail_bytes / node_filesystem_size_bytes) * 100' },
  { label: 'GPU Utilization', query: 'DCGM_FI_DEV_GPU_UTIL' },
  { label: 'GPU Memory', query: 'DCGM_FI_DEV_FB_USED / DCGM_FI_DEV_FB_TOTAL * 100' },
  { label: 'Network RX', query: 'rate(node_network_receive_bytes_total[5m])' },
  { label: 'Network TX', query: 'rate(node_network_transmit_bytes_total[5m])' },
  { label: 'Up Targets', query: 'up' },
];

const TIME_RANGES = [
  { label: '15m', value: 15 * 60 },
  { label: '1h', value: 60 * 60 },
  { label: '3h', value: 3 * 60 * 60 },
  { label: '6h', value: 6 * 60 * 60 },
  { label: '12h', value: 12 * 60 * 60 },
  { label: '24h', value: 24 * 60 * 60 },
];

export function MetricsExplorer() {
  const [query, setQuery] = useState('up');
  const [timeRange, setTimeRange] = useState(60 * 60); // 1 hour default
  const [queryType, setQueryType] = useState<'instant' | 'range'>('range');

  const { queryResult, loading, error, queryInstant, queryRange } = useArgusStore();

  const handleExecute = async () => {
    if (!query.trim()) return;

    if (queryType === 'instant') {
      await queryInstant(query);
    } else {
      const now = Math.floor(Date.now() / 1000);
      const start = now - timeRange;
      const step = Math.max(15, Math.floor(timeRange / 100)); // Dynamic step
      await queryRange(query, start, now, step);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      handleExecute();
    }
  };

  return (
    <div className="space-y-6">
      {/* Query Input */}
      <div className="bg-slate-800 border border-slate-700 rounded-lg p-4">
        <div className="flex items-center gap-2 mb-3">
          <Search className="w-5 h-5 text-slate-400" />
          <h3 className="text-lg font-medium text-white">PromQL Query</h3>
        </div>

        <div className="space-y-3">
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Enter PromQL query (e.g., rate(http_requests_total[5m]))"
            className="w-full h-24 bg-slate-900 border border-slate-700 rounded-lg p-3
                       text-white font-mono text-sm resize-none focus:outline-none
                       focus:border-cyan-500"
          />

          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              {/* Query Type Toggle */}
              <div className="flex bg-slate-900 rounded-lg p-1">
                <button
                  onClick={() => setQueryType('instant')}
                  className={`px-3 py-1.5 text-sm rounded transition-colors ${
                    queryType === 'instant'
                      ? 'bg-cyan-500 text-white'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  <BarChart3 className="w-4 h-4 inline mr-1" />
                  Instant
                </button>
                <button
                  onClick={() => setQueryType('range')}
                  className={`px-3 py-1.5 text-sm rounded transition-colors ${
                    queryType === 'range'
                      ? 'bg-cyan-500 text-white'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  <Clock className="w-4 h-4 inline mr-1" />
                  Range
                </button>
              </div>

              {/* Time Range Selector (only for range queries) */}
              {queryType === 'range' && (
                <div className="flex items-center gap-1">
                  {TIME_RANGES.map((range) => (
                    <button
                      key={range.value}
                      onClick={() => setTimeRange(range.value)}
                      className={`px-2 py-1 text-xs rounded transition-colors ${
                        timeRange === range.value
                          ? 'bg-slate-700 text-white'
                          : 'text-slate-400 hover:text-white'
                      }`}
                    >
                      {range.label}
                    </button>
                  ))}
                </div>
              )}
            </div>

            <button
              onClick={handleExecute}
              disabled={loading || !query.trim()}
              className="flex items-center gap-2 px-4 py-2 bg-cyan-500 hover:bg-cyan-600
                         text-white rounded-lg font-medium transition-colors
                         disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Play className="w-4 h-4" />
              Execute
            </button>
          </div>
        </div>
      </div>

      {/* Common Queries */}
      <div className="bg-slate-800 border border-slate-700 rounded-lg p-4">
        <h4 className="text-sm font-medium text-slate-400 mb-3">Common Queries</h4>
        <div className="flex flex-wrap gap-2">
          {COMMON_QUERIES.map((q) => (
            <button
              key={q.label}
              onClick={() => setQuery(q.query)}
              className="px-3 py-1.5 text-sm bg-slate-700 hover:bg-slate-600
                         text-slate-300 rounded-lg transition-colors"
            >
              {q.label}
            </button>
          ))}
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/50 rounded-lg p-4">
          <p className="text-red-400">Query Error: {error}</p>
        </div>
      )}

      {/* Loading State */}
      {loading && (
        <div className="flex items-center justify-center h-64 bg-slate-800 rounded-lg">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-cyan-500" />
        </div>
      )}

      {/* Results */}
      {queryResult && !loading && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-medium text-white">Results</h3>
            <span className="text-sm text-slate-400">
              {queryResult.data.length} series, type: {queryResult.result_type}
            </span>
          </div>
          <MetricChart result={queryResult} height={350} />

          {/* Raw Data Table */}
          <div className="bg-slate-800 border border-slate-700 rounded-lg overflow-hidden">
            <div className="p-3 border-b border-slate-700">
              <h4 className="text-sm font-medium text-slate-400">Raw Data</h4>
            </div>
            <div className="max-h-64 overflow-auto">
              <table className="w-full text-sm">
                <thead className="bg-slate-900 sticky top-0">
                  <tr>
                    <th className="text-left p-2 text-slate-400">Metric</th>
                    <th className="text-right p-2 text-slate-400">Value</th>
                  </tr>
                </thead>
                <tbody>
                  {queryResult.data.map((sample, idx) => {
                    const labelStr = Object.entries(sample.metric)
                      .map(([k, v]) => `${k}="${v}"`)
                      .join(', ');
                    const value = sample.value
                      ? sample.value[1]
                      : sample.values.length > 0
                      ? sample.values[sample.values.length - 1][1]
                      : 'N/A';

                    return (
                      <tr key={idx} className="border-t border-slate-700">
                        <td className="p-2 text-slate-300 font-mono text-xs truncate max-w-md">
                          {labelStr || `series_${idx}`}
                        </td>
                        <td className="p-2 text-right text-cyan-400 font-mono">
                          {parseFloat(value).toFixed(4)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

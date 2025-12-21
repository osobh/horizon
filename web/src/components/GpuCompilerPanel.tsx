/**
 * GpuCompilerPanel - Displays GPU compiler status and compilation metrics
 *
 * Uses Tauri IPC to interact with the rustg GPU compiler backend.
 * Shows compilation benchmarks, GPU utilization, and speedup metrics.
 */

import { useEffect, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import {
  Zap,
  RefreshCw,
  Play,
  Gauge,
  Clock,
  Cpu,
  MemoryStick,
} from 'lucide-react';

interface GpuCompilerStatus {
  available: boolean;
  initialized: boolean;
  backend: string;
  memory_limit: number;
  cpu_fallback_enabled: boolean;
  device_name: string | null;
}

interface BenchmarkResult {
  gpu_time_ms: number;
  cpu_time_ms: number;
  speedup: number;
  gpu_accelerated: boolean;
  gpu_utilization: number;
}

interface GpuCompilationResult {
  success: boolean;
  total_time_ms: number;
  parsing_time_ms: number;
  type_check_time_ms: number;
  codegen_time_ms: number;
  token_count: number;
  gpu_memory_used: number;
  gpu_utilization: number;
  error: string | null;
  gpu_accelerated: boolean;
}

const BACKEND_COLORS: Record<string, string> = {
  Metal: 'text-slate-300 bg-gradient-to-r from-slate-700 to-slate-600 border-slate-500',
  CUDA: 'text-green-300 bg-gradient-to-r from-green-800 to-green-700 border-green-500',
  OpenCL: 'text-blue-300 bg-gradient-to-r from-blue-800 to-blue-700 border-blue-500',
  CPU: 'text-yellow-300 bg-gradient-to-r from-yellow-800 to-yellow-700 border-yellow-500',
  none: 'text-slate-400 bg-slate-800 border-slate-600',
};

interface GpuCompilerPanelProps {
  compact?: boolean;
}

const SAMPLE_CODE = `// GPU Compiler Benchmark
fn fibonacci(n: u64) -> u64 {
    match n {
        0 => 0,
        1 => 1,
        _ => fibonacci(n - 1) + fibonacci(n - 2),
    }
}

struct Matrix<const N: usize> {
    data: [[f64; N]; N],
}

impl<const N: usize> Matrix<N> {
    fn multiply(&self, other: &Self) -> Self {
        let mut result = Matrix { data: [[0.0; N]; N] };
        for i in 0..N {
            for j in 0..N {
                for k in 0..N {
                    result.data[i][j] += self.data[i][k] * other.data[k][j];
                }
            }
        }
        result
    }
}

fn main() {
    let fib_10 = fibonacci(10);
    println!("Fibonacci(10) = {}", fib_10);
}
`;

export default function GpuCompilerPanel({ compact = false }: GpuCompilerPanelProps) {
  const [status, setStatus] = useState<GpuCompilerStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [benchmarking, setBenchmarking] = useState(false);
  const [lastResult, setLastResult] = useState<GpuCompilationResult | null>(null);
  const [lastBenchmark, setLastBenchmark] = useState<BenchmarkResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchStatus = async () => {
    setLoading(true);
    setError(null);
    try {
      const compilerStatus = await invoke<GpuCompilerStatus>('get_gpu_compiler_status');
      setStatus(compilerStatus);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  };

  const runBenchmark = async () => {
    setBenchmarking(true);
    setError(null);
    try {
      // First compile and get detailed result
      const result = await invoke<GpuCompilationResult>('gpu_compile', { source: SAMPLE_CODE });
      setLastResult(result);

      // Then run benchmark for speedup comparison
      const benchmark = await invoke<BenchmarkResult>('benchmark_gpu_compiler', { source: SAMPLE_CODE });
      setLastBenchmark(benchmark);
    } catch (err) {
      setError(String(err));
    } finally {
      setBenchmarking(false);
    }
  };

  useEffect(() => {
    fetchStatus();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <RefreshCw className="w-6 h-6 animate-spin text-slate-400" />
      </div>
    );
  }

  if (error && !status) {
    return (
      <div className="p-4 bg-red-900/20 border border-red-700/50 rounded-lg">
        <p className="text-sm text-red-400">Failed to load GPU compiler status: {error}</p>
        <button
          onClick={fetchStatus}
          className="mt-2 text-xs text-red-300 hover:text-red-200"
        >
          Retry
        </button>
      </div>
    );
  }

  if (!status) {
    return null;
  }

  const backendColor = BACKEND_COLORS[status.backend] || BACKEND_COLORS.none;

  if (compact) {
    return (
      <div className="space-y-2">
        <div className="flex items-center gap-2 text-sm">
          <Zap className="w-4 h-4 text-yellow-400" />
          <span className="text-slate-300">GPU Compiler</span>
          <span className={`text-xs px-1.5 py-0.5 rounded border ${backendColor}`}>
            {status.backend}
          </span>
          {status.initialized && (
            <span className="text-xs text-green-400">Ready</span>
          )}
        </div>
        {lastBenchmark && (
          <div className="flex items-center gap-2 text-xs text-slate-400">
            <Gauge className="w-3 h-3" />
            <span>{lastBenchmark.speedup.toFixed(1)}x speedup</span>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Zap className="w-5 h-5 text-yellow-400" />
          <h3 className="font-medium">GPU Compiler</h3>
          <span className={`text-xs px-2 py-0.5 rounded border ${backendColor}`}>
            {status.backend}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={runBenchmark}
            disabled={benchmarking}
            className="flex items-center gap-1 px-2 py-1 text-xs bg-yellow-600 hover:bg-yellow-500 disabled:bg-slate-700 disabled:text-slate-500 rounded transition-colors"
          >
            {benchmarking ? (
              <RefreshCw className="w-3 h-3 animate-spin" />
            ) : (
              <Play className="w-3 h-3" />
            )}
            Benchmark
          </button>
          <button
            onClick={fetchStatus}
            className="p-1.5 hover:bg-slate-700 rounded text-slate-400 hover:text-slate-300"
            title="Refresh"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Status Grid */}
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-slate-800 rounded-lg p-3 border border-slate-700">
          <div className="flex items-center gap-2 text-slate-400 text-xs mb-1">
            <Cpu className="w-3 h-3" />
            Device
          </div>
          <div className="font-medium text-sm truncate">
            {status.device_name || 'Not detected'}
          </div>
          <div className="text-xs text-slate-500">
            {status.available ? 'Available' : 'Not available'}
          </div>
        </div>

        <div className="bg-slate-800 rounded-lg p-3 border border-slate-700">
          <div className="flex items-center gap-2 text-slate-400 text-xs mb-1">
            <MemoryStick className="w-3 h-3" />
            Memory Limit
          </div>
          <div className="font-medium text-sm">
            {(status.memory_limit / (1024 * 1024 * 1024)).toFixed(1)} GB
          </div>
          <div className="text-xs text-slate-500">
            CPU fallback: {status.cpu_fallback_enabled ? 'Enabled' : 'Disabled'}
          </div>
        </div>
      </div>

      {/* Last Compilation Result */}
      {lastResult && (
        <div className="bg-slate-800 rounded-lg p-3 border border-slate-700">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2 text-slate-400 text-xs">
              <Clock className="w-3 h-3" />
              Last Compilation
            </div>
            <span className={`text-xs px-1.5 py-0.5 rounded ${
              lastResult.success
                ? 'bg-green-900/50 text-green-400 border border-green-700/50'
                : 'bg-red-900/50 text-red-400 border border-red-700/50'
            }`}>
              {lastResult.success ? 'Success' : 'Failed'}
            </span>
          </div>

          <div className="grid grid-cols-3 gap-2 text-xs">
            <div className="bg-slate-700/50 rounded p-2">
              <div className="text-slate-400">Total Time</div>
              <div className="font-mono text-lg">{lastResult.total_time_ms.toFixed(2)}ms</div>
            </div>
            <div className="bg-slate-700/50 rounded p-2">
              <div className="text-slate-400">Tokens</div>
              <div className="font-mono text-lg">{lastResult.token_count}</div>
            </div>
            <div className="bg-slate-700/50 rounded p-2">
              <div className="text-slate-400">GPU Util</div>
              <div className="font-mono text-lg">{lastResult.gpu_utilization.toFixed(0)}%</div>
            </div>
          </div>

          {/* Timing Breakdown */}
          <div className="mt-3 space-y-1">
            <div className="flex justify-between text-xs">
              <span className="text-slate-400">Parsing</span>
              <span className="font-mono">{lastResult.parsing_time_ms.toFixed(2)}ms</span>
            </div>
            <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-blue-500 rounded-full"
                style={{ width: `${(lastResult.parsing_time_ms / lastResult.total_time_ms) * 100}%` }}
              />
            </div>

            <div className="flex justify-between text-xs">
              <span className="text-slate-400">Type Check</span>
              <span className="font-mono">{lastResult.type_check_time_ms.toFixed(2)}ms</span>
            </div>
            <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-purple-500 rounded-full"
                style={{ width: `${(lastResult.type_check_time_ms / lastResult.total_time_ms) * 100}%` }}
              />
            </div>

            <div className="flex justify-between text-xs">
              <span className="text-slate-400">Code Gen</span>
              <span className="font-mono">{lastResult.codegen_time_ms.toFixed(2)}ms</span>
            </div>
            <div className="h-1.5 bg-slate-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-green-500 rounded-full"
                style={{ width: `${(lastResult.codegen_time_ms / lastResult.total_time_ms) * 100}%` }}
              />
            </div>
          </div>
        </div>
      )}

      {/* Benchmark Result */}
      {lastBenchmark && (
        <div className={`rounded-lg p-4 border ${
          lastBenchmark.gpu_accelerated
            ? 'bg-gradient-to-r from-green-900/30 to-blue-900/30 border-green-700/50'
            : 'bg-slate-800 border-slate-700'
        }`}>
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <Gauge className="w-4 h-4 text-yellow-400" />
              <span className="font-medium">Performance</span>
            </div>
            <span className={`text-2xl font-bold ${
              lastBenchmark.speedup > 5 ? 'text-green-400' :
              lastBenchmark.speedup > 2 ? 'text-yellow-400' :
              'text-slate-400'
            }`}>
              {lastBenchmark.speedup.toFixed(1)}x
            </span>
          </div>

          <div className="grid grid-cols-2 gap-3 text-sm">
            <div>
              <div className="text-slate-400 text-xs">GPU Time</div>
              <div className="font-mono">{lastBenchmark.gpu_time_ms.toFixed(2)}ms</div>
            </div>
            <div>
              <div className="text-slate-400 text-xs">Est. CPU Time</div>
              <div className="font-mono">{lastBenchmark.cpu_time_ms.toFixed(2)}ms</div>
            </div>
          </div>

          {lastBenchmark.gpu_accelerated && (
            <div className="mt-3 text-xs text-green-400 flex items-center gap-1">
              <Zap className="w-3 h-3" />
              GPU acceleration active
            </div>
          )}
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="p-3 bg-red-900/20 border border-red-700/50 rounded-lg text-sm text-red-400">
          {error}
        </div>
      )}
    </div>
  );
}

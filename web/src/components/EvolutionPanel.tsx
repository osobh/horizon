/**
 * EvolutionPanel - Displays self-improving agent evolution metrics
 *
 * Shows metrics from three evolution engines:
 * - ADAS: Automated Design of Agentic Systems
 * - DGM: Discovered Growth Mode
 * - SwarmAgentic: Population-based optimization
 */

import { useEffect, useState, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import {
  TrendingUp,
  RefreshCw,
  Dna,
  Brain,
  Users,
  Sparkles,
  Activity,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';

interface EvolutionMetrics {
  generation: number;
  total_evaluations: number;
  best_fitness: number;
  average_fitness: number;
  diversity_score: number;
  convergence_rate: number;
  elapsed_secs: number;
  custom_metrics: Record<string, number>;
}

interface AdasMetrics extends EvolutionMetrics {
  designs_explored: number;
  best_design_score: number;
  current_design: string;
  architecture_complexity: number;
}

interface DgmMetrics extends EvolutionMetrics {
  self_assessment_score: number;
  code_modifications: number;
  improvement_capability: number;
  growth_patterns: string[];
  recommendations: string[];
}

interface SwarmMetrics extends EvolutionMetrics {
  population_size: number;
  active_particles: number;
  global_best_fitness: number;
  velocity_diversity: number;
  cluster_nodes: number;
}

interface EvolutionEvent {
  timestamp: string;
  engine: 'adas' | 'dgm' | 'swarm';
  event_type: string;
  description: string;
  metrics_delta: Record<string, number> | null;
}

interface EvolutionStatus {
  adas: EngineStatus;
  dgm: EngineStatus;
  swarm: EngineStatus;
  recent_events: EvolutionEvent[];
}

interface EngineStatus {
  running: boolean;
  generation: number;
  best_fitness: number;
  improvement_pct: number;
}

const ENGINE_COLORS = {
  adas: {
    bg: 'bg-purple-900/30',
    border: 'border-purple-700/50',
    text: 'text-purple-400',
    accent: 'bg-purple-500',
  },
  dgm: {
    bg: 'bg-blue-900/30',
    border: 'border-blue-700/50',
    text: 'text-blue-400',
    accent: 'bg-blue-500',
  },
  swarm: {
    bg: 'bg-green-900/30',
    border: 'border-green-700/50',
    text: 'text-green-400',
    accent: 'bg-green-500',
  },
};

interface EvolutionPanelProps {
  compact?: boolean;
}

export default function EvolutionPanel({ compact = false }: EvolutionPanelProps) {
  const [status, setStatus] = useState<EvolutionStatus | null>(null);
  const [adasMetrics, setAdasMetrics] = useState<AdasMetrics | null>(null);
  const [dgmMetrics, setDgmMetrics] = useState<DgmMetrics | null>(null);
  const [swarmMetrics, setSwarmMetrics] = useState<SwarmMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<'adas' | 'dgm' | 'swarm' | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const [statusData, adas, dgm, swarm] = await Promise.all([
        invoke<EvolutionStatus>('get_evolution_status'),
        invoke<AdasMetrics>('get_adas_metrics'),
        invoke<DgmMetrics>('get_dgm_metrics'),
        invoke<SwarmMetrics>('get_swarm_metrics'),
      ]);
      setStatus(statusData);
      setAdasMetrics(adas);
      setDgmMetrics(dgm);
      setSwarmMetrics(swarm);
      setError(null);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  const simulateStep = async () => {
    try {
      await invoke('simulate_evolution_step');
      await fetchData();
    } catch (err) {
      console.error('Failed to simulate step:', err);
    }
  };

  useEffect(() => {
    fetchData();
    // Refresh every 10 seconds
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, [fetchData]);

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <RefreshCw className="w-6 h-6 animate-spin text-slate-400" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 bg-red-900/20 border border-red-700/50 rounded-lg">
        <p className="text-sm text-red-400">Failed to load evolution status: {error}</p>
        <button
          onClick={fetchData}
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

  if (compact) {
    return (
      <div className="space-y-2">
        <div className="flex items-center gap-2 text-sm">
          <TrendingUp className="w-4 h-4 text-green-400" />
          <span className="text-slate-300">Evolution Engines</span>
          <span className="text-xs bg-green-500/20 text-green-400 px-1.5 py-0.5 rounded">
            3 Active
          </span>
        </div>
        <div className="grid grid-cols-3 gap-2 text-xs">
          <div className={`p-2 rounded ${ENGINE_COLORS.adas.bg} ${ENGINE_COLORS.adas.border} border`}>
            <div className={ENGINE_COLORS.adas.text}>ADAS</div>
            <div className="font-mono">{(status.adas.best_fitness * 100).toFixed(1)}%</div>
          </div>
          <div className={`p-2 rounded ${ENGINE_COLORS.dgm.bg} ${ENGINE_COLORS.dgm.border} border`}>
            <div className={ENGINE_COLORS.dgm.text}>DGM</div>
            <div className="font-mono">{(status.dgm.best_fitness * 100).toFixed(1)}%</div>
          </div>
          <div className={`p-2 rounded ${ENGINE_COLORS.swarm.bg} ${ENGINE_COLORS.swarm.border} border`}>
            <div className={ENGINE_COLORS.swarm.text}>Swarm</div>
            <div className="font-mono">{(status.swarm.best_fitness * 100).toFixed(1)}%</div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <TrendingUp className="w-5 h-5 text-green-400" />
          <h3 className="font-medium">Evolution Engines</h3>
          <span className="text-xs bg-green-500/20 text-green-400 px-2 py-0.5 rounded">
            Self-improving
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={simulateStep}
            className="flex items-center gap-1 px-2 py-1 text-xs bg-green-600 hover:bg-green-500 rounded transition-colors"
          >
            <Sparkles className="w-3 h-3" />
            Evolve
          </button>
          <button
            onClick={fetchData}
            className="p-1.5 hover:bg-slate-700 rounded text-slate-400 hover:text-slate-300"
            title="Refresh"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Engine Cards */}
      <div className="space-y-3">
        {/* ADAS */}
        <EngineCard
          name="ADAS"
          description="Automated Design of Agentic Systems"
          icon={Dna}
          status={status.adas}
          colors={ENGINE_COLORS.adas}
          expanded={expanded === 'adas'}
          onToggle={() => setExpanded(expanded === 'adas' ? null : 'adas')}
        >
          {adasMetrics && (
            <div className="grid grid-cols-2 gap-3 text-xs">
              <MetricItem label="Designs Explored" value={adasMetrics.designs_explored} />
              <MetricItem label="Best Design Score" value={`${(adasMetrics.best_design_score * 100).toFixed(1)}%`} />
              <MetricItem label="Current Design" value={adasMetrics.current_design} />
              <MetricItem label="Complexity" value={`${(adasMetrics.architecture_complexity * 100).toFixed(0)}%`} />
            </div>
          )}
        </EngineCard>

        {/* DGM */}
        <EngineCard
          name="DGM"
          description="Discovered Growth Mode"
          icon={Brain}
          status={status.dgm}
          colors={ENGINE_COLORS.dgm}
          expanded={expanded === 'dgm'}
          onToggle={() => setExpanded(expanded === 'dgm' ? null : 'dgm')}
        >
          {dgmMetrics && (
            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-3 text-xs">
                <MetricItem label="Self-Assessment" value={`${(dgmMetrics.self_assessment_score * 100).toFixed(0)}%`} />
                <MetricItem label="Code Modifications" value={dgmMetrics.code_modifications} />
                <MetricItem label="Improvement Capability" value={`${(dgmMetrics.improvement_capability * 100).toFixed(0)}%`} />
                <MetricItem label="Patterns Discovered" value={dgmMetrics.growth_patterns.length} />
              </div>
              {dgmMetrics.recommendations.length > 0 && (
                <div className="bg-blue-900/20 p-2 rounded text-xs">
                  <div className="text-blue-400 mb-1">Recommendations:</div>
                  <ul className="list-disc list-inside space-y-1 text-slate-300">
                    {dgmMetrics.recommendations.map((rec, i) => (
                      <li key={i}>{rec}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </EngineCard>

        {/* SwarmAgentic */}
        <EngineCard
          name="SwarmAgentic"
          description="Population-based Optimization"
          icon={Users}
          status={status.swarm}
          colors={ENGINE_COLORS.swarm}
          expanded={expanded === 'swarm'}
          onToggle={() => setExpanded(expanded === 'swarm' ? null : 'swarm')}
        >
          {swarmMetrics && (
            <div className="grid grid-cols-2 gap-3 text-xs">
              <MetricItem label="Population Size" value={swarmMetrics.population_size} />
              <MetricItem label="Active Particles" value={swarmMetrics.active_particles} />
              <MetricItem label="Global Best" value={`${(swarmMetrics.global_best_fitness * 100).toFixed(1)}%`} />
              <MetricItem label="Velocity Diversity" value={`${(swarmMetrics.velocity_diversity * 100).toFixed(0)}%`} />
              <MetricItem label="Cluster Nodes" value={swarmMetrics.cluster_nodes} />
              <MetricItem label="Diversity Score" value={`${(swarmMetrics.diversity_score * 100).toFixed(0)}%`} />
            </div>
          )}
        </EngineCard>
      </div>

      {/* Recent Events */}
      <div className="bg-slate-800/50 rounded-lg border border-slate-700 p-3">
        <div className="flex items-center gap-2 text-slate-400 text-xs mb-2">
          <Activity className="w-3 h-3" />
          Recent Events
        </div>
        <div className="space-y-2">
          {status.recent_events.slice(0, 4).map((event, i) => (
            <div
              key={i}
              className={`flex items-start gap-3 p-2 rounded ${
                ENGINE_COLORS[event.engine].bg
              } border ${ENGINE_COLORS[event.engine].border}`}
            >
              <div className="w-12 text-xs text-slate-400 flex-shrink-0">
                {event.timestamp}
              </div>
              <div className="text-sm flex-1">{event.description}</div>
              <span className={`text-xs ${ENGINE_COLORS[event.engine].text}`}>
                {event.engine.toUpperCase()}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

interface EngineCardProps {
  name: string;
  description: string;
  icon: React.ComponentType<{ className?: string }>;
  status: EngineStatus;
  colors: typeof ENGINE_COLORS.adas;
  expanded: boolean;
  onToggle: () => void;
  children?: React.ReactNode;
}

function EngineCard({
  name,
  description,
  icon: Icon,
  status,
  colors,
  expanded,
  onToggle,
  children,
}: EngineCardProps) {
  return (
    <div className={`rounded-lg border ${colors.bg} ${colors.border}`}>
      <button
        onClick={onToggle}
        className="w-full p-3 flex items-center justify-between hover:bg-white/5 transition-colors"
      >
        <div className="flex items-center gap-3">
          <Icon className={`w-5 h-5 ${colors.text}`} />
          <div className="text-left">
            <div className="font-medium">{name}</div>
            <div className="text-xs text-slate-400">{description}</div>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className="text-right">
            <div className={`text-lg font-bold ${colors.text}`}>
              {(status.best_fitness * 100).toFixed(1)}%
            </div>
            <div className="text-xs text-slate-400">
              Gen {status.generation}
            </div>
          </div>
          <div className="text-xs text-green-400">
            +{status.improvement_pct.toFixed(1)}%
          </div>
          {expanded ? (
            <ChevronUp className="w-4 h-4 text-slate-400" />
          ) : (
            <ChevronDown className="w-4 h-4 text-slate-400" />
          )}
        </div>
      </button>
      {expanded && children && (
        <div className="px-3 pb-3 pt-1 border-t border-slate-700/50">
          {children}
        </div>
      )}
    </div>
  );
}

interface MetricItemProps {
  label: string;
  value: string | number;
}

function MetricItem({ label, value }: MetricItemProps) {
  return (
    <div className="bg-slate-800/50 rounded p-2">
      <div className="text-slate-400">{label}</div>
      <div className="font-mono truncate">{value}</div>
    </div>
  );
}

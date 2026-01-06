import { useEffect, useState } from 'react';
import {
  Dna,
  Brain,
  Zap,
  Sparkles,
  GitBranch,
  RefreshCw,
  Play,
  TrendingUp,
  Activity,
  Target,
  Users,
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, RadarChart, PolarGrid, PolarAngleAxis, Radar } from 'recharts';
import { useEvolutionStore, AdasMetrics, DgmMetrics, SwarmMetrics, EvolutionEvent } from '../../stores/evolutionStore';

export default function EvolutionView() {
  const {
    status,
    adasMetrics,
    dgmMetrics,
    swarmMetrics,
    events,
    loading,
    error,
    fetchStatus,
    fetchAdasMetrics,
    fetchDgmMetrics,
    fetchSwarmMetrics,
    fetchEvents,
    simulateStep,
  } = useEvolutionStore();

  const [selectedTab, setSelectedTab] = useState<'overview' | 'adas' | 'dgm' | 'swarm'>('overview');

  useEffect(() => {
    fetchStatus();
    fetchAdasMetrics();
    fetchDgmMetrics();
    fetchSwarmMetrics();
    fetchEvents();
    const interval = setInterval(() => {
      fetchStatus();
      fetchAdasMetrics();
      fetchDgmMetrics();
      fetchSwarmMetrics();
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="h-full flex flex-col bg-slate-900 text-white">
      {/* Header */}
      <div className="h-16 bg-slate-800 border-b border-slate-700 flex items-center px-6">
        <Dna className="w-6 h-6 text-purple-400 mr-3" />
        <h1 className="text-xl font-semibold">Evolution Engines</h1>
        <div className="flex-1" />
        <button
          onClick={simulateStep}
          className="mr-2 px-3 py-1.5 bg-purple-600 hover:bg-purple-500 rounded-lg text-sm transition-colors flex items-center gap-2"
        >
          <Play className="w-4 h-4" />
          Evolve Step
        </button>
        <button onClick={() => { fetchStatus(); fetchAdasMetrics(); fetchDgmMetrics(); fetchSwarmMetrics(); }} className="p-2 hover:bg-slate-700 rounded-lg transition-colors">
          <RefreshCw className={`w-5 h-5 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Status Bar */}
      {status && (
        <div className="bg-slate-800/50 border-b border-slate-700 px-6 py-3 flex gap-6 text-sm">
          <div className="flex items-center gap-2">
            <span className={`w-2 h-2 rounded-full ${status.active ? 'bg-green-400 animate-pulse' : 'bg-slate-500'}`} />
            <span className="text-slate-400">Status:</span>
            <span className={status.active ? 'text-green-400' : 'text-slate-500'}>
              {status.active ? 'Evolving' : 'Idle'}
            </span>
          </div>
          <div><span className="text-slate-400">Generation:</span> {status.current_generation.toLocaleString()}</div>
          <div><span className="text-slate-400">Best Fitness:</span> <span className="text-purple-400">{status.best_fitness.toFixed(4)}</span></div>
          <div><span className="text-slate-400">Population:</span> {status.population_size.toLocaleString()}</div>
        </div>
      )}

      {/* Tabs */}
      <div className="border-b border-slate-700 px-6">
        <div className="flex gap-4">
          {['overview', 'adas', 'dgm', 'swarm'].map((tab) => (
            <button
              key={tab}
              onClick={() => setSelectedTab(tab as any)}
              className={`py-3 px-4 border-b-2 transition-colors uppercase text-sm ${
                selectedTab === tab ? 'border-purple-400 text-purple-400' : 'border-transparent text-slate-400 hover:text-white'
              }`}
            >
              {tab}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto p-6">
        {error && <div className="mb-4 p-4 bg-red-500/20 border border-red-500 rounded-lg text-red-400">{error}</div>}

        {selectedTab === 'overview' && status && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Fitness Evolution */}
            <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
              <h3 className="font-medium mb-4">Fitness Evolution</h3>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={generateFitnessData()}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="generation" stroke="#94a3b8" fontSize={12} />
                  <YAxis stroke="#94a3b8" fontSize={12} domain={[0, 1]} />
                  <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none' }} />
                  <Line type="monotone" dataKey="best" stroke="#a855f7" strokeWidth={2} name="Best Fitness" />
                  <Line type="monotone" dataKey="average" stroke="#6366f1" strokeWidth={2} name="Avg Fitness" />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Engine Comparison */}
            <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
              <h3 className="font-medium mb-4">Engine Performance</h3>
              <ResponsiveContainer width="100%" height={250}>
                <RadarChart data={[
                  { metric: 'Speed', adas: 85, dgm: 70, swarm: 95 },
                  { metric: 'Quality', adas: 92, dgm: 88, swarm: 75 },
                  { metric: 'Diversity', adas: 65, dgm: 95, swarm: 85 },
                  { metric: 'Convergence', adas: 78, dgm: 72, swarm: 68 },
                  { metric: 'Stability', adas: 88, dgm: 80, swarm: 72 },
                ]}>
                  <PolarGrid stroke="#334155" />
                  <PolarAngleAxis dataKey="metric" stroke="#94a3b8" fontSize={12} />
                  <Radar name="ADAS" dataKey="adas" stroke="#f472b6" fill="#f472b6" fillOpacity={0.2} />
                  <Radar name="DGM" dataKey="dgm" stroke="#a855f7" fill="#a855f7" fillOpacity={0.2} />
                  <Radar name="Swarm" dataKey="swarm" stroke="#22d3ee" fill="#22d3ee" fillOpacity={0.2} />
                </RadarChart>
              </ResponsiveContainer>
            </div>

            {/* Recent Events */}
            <div className="lg:col-span-2 bg-slate-800 rounded-lg border border-slate-700 p-4">
              <h3 className="font-medium mb-4">Recent Evolution Events</h3>
              <div className="space-y-2 max-h-64 overflow-auto">
                {events.slice(0, 10).map((event) => (
                  <EventRow key={event.id} event={event} />
                ))}
                {events.length === 0 && (
                  <div className="text-slate-500 text-center py-4">No recent events</div>
                )}
              </div>
            </div>
          </div>
        )}

        {selectedTab === 'adas' && adasMetrics && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
              <h3 className="font-medium mb-4 flex items-center gap-2">
                <Sparkles className="w-5 h-5 text-pink-400" />
                ADAS - Automated Design Architecture Search
              </h3>
              <div className="space-y-4">
                <MetricRow label="Architectures Explored" value={adasMetrics.architectures_explored.toLocaleString()} />
                <MetricRow label="Best Architecture Score" value={adasMetrics.best_architecture_score.toFixed(4)} highlight />
                <MetricRow label="Search Space Coverage" value={`${(adasMetrics.search_space_coverage * 100).toFixed(1)}%`} />
                <MetricRow label="Convergence Rate" value={adasMetrics.convergence_rate.toFixed(4)} />
                <MetricRow label="Mutations Applied" value={adasMetrics.mutations_applied.toLocaleString()} />
              </div>
            </div>

            <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
              <h3 className="font-medium mb-4">Architecture Search Progress</h3>
              <ResponsiveContainer width="100%" height={250}>
                <AreaChart data={generateAdasProgressData()}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="iteration" stroke="#94a3b8" fontSize={12} />
                  <YAxis stroke="#94a3b8" fontSize={12} />
                  <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none' }} />
                  <Area type="monotone" dataKey="score" stroke="#f472b6" fill="#f472b6" fillOpacity={0.3} />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            {/* Top Architectures */}
            <div className="lg:col-span-2 bg-slate-800 rounded-lg border border-slate-700 p-4">
              <h3 className="font-medium mb-4">Top Architectures</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {generateTopArchitectures().map((arch, i) => (
                  <div key={arch.id} className="bg-slate-700/30 rounded-lg p-4">
                    <div className="flex items-center gap-2 mb-2">
                      <span className={`text-lg font-bold ${i === 0 ? 'text-amber-400' : i === 1 ? 'text-slate-300' : 'text-amber-600'}`}>
                        #{i + 1}
                      </span>
                      <span className="font-medium">{arch.name}</span>
                    </div>
                    <div className="text-sm text-slate-400 mb-2">{arch.layers} layers, {arch.params}M params</div>
                    <div className="text-purple-400 font-semibold">Score: {arch.score.toFixed(4)}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {selectedTab === 'dgm' && dgmMetrics && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
              <h3 className="font-medium mb-4 flex items-center gap-2">
                <Brain className="w-5 h-5 text-purple-400" />
                DGM - Dynamic Genetic Mutation
              </h3>
              <div className="space-y-4">
                <MetricRow label="Total Generations" value={dgmMetrics.total_generations.toLocaleString()} />
                <MetricRow label="Mutation Rate" value={`${(dgmMetrics.mutation_rate * 100).toFixed(1)}%`} />
                <MetricRow label="Crossover Rate" value={`${(dgmMetrics.crossover_rate * 100).toFixed(1)}%`} />
                <MetricRow label="Selection Pressure" value={dgmMetrics.selection_pressure.toFixed(2)} />
                <MetricRow label="Genetic Diversity" value={`${(dgmMetrics.diversity_index * 100).toFixed(1)}%`} highlight />
              </div>
            </div>

            <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
              <h3 className="font-medium mb-4">Genetic Diversity Over Generations</h3>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={generateDgmDiversityData()}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="generation" stroke="#94a3b8" fontSize={12} />
                  <YAxis stroke="#94a3b8" fontSize={12} domain={[0, 100]} />
                  <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none' }} />
                  <Line type="monotone" dataKey="diversity" stroke="#a855f7" strokeWidth={2} />
                  <Line type="monotone" dataKey="fitness" stroke="#10b981" strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Population Stats */}
            <div className="lg:col-span-2 grid grid-cols-4 gap-4">
              <StatCard icon={Users} label="Population Size" value={dgmMetrics.population_size.toLocaleString()} color="text-purple-400" />
              <StatCard icon={GitBranch} label="Elite Count" value={dgmMetrics.elite_count.toLocaleString()} color="text-amber-400" />
              <StatCard icon={TrendingUp} label="Avg Fitness" value={dgmMetrics.average_fitness.toFixed(4)} color="text-green-400" />
              <StatCard icon={Target} label="Best Fitness" value={dgmMetrics.best_fitness.toFixed(4)} color="text-pink-400" />
            </div>
          </div>
        )}

        {selectedTab === 'swarm' && swarmMetrics && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
              <h3 className="font-medium mb-4 flex items-center gap-2">
                <Zap className="w-5 h-5 text-cyan-400" />
                SwarmAgentic - Swarm-Based Optimization
              </h3>
              <div className="space-y-4">
                <MetricRow label="Active Agents" value={swarmMetrics.active_agents.toLocaleString()} />
                <MetricRow label="Tasks Completed" value={swarmMetrics.tasks_completed.toLocaleString()} />
                <MetricRow label="Coordination Score" value={`${(swarmMetrics.coordination_score * 100).toFixed(1)}%`} highlight />
                <MetricRow label="Avg Response Time" value={`${swarmMetrics.average_response_ms.toFixed(1)}ms`} />
                <MetricRow label="Communication Overhead" value={`${(swarmMetrics.communication_overhead * 100).toFixed(1)}%`} />
              </div>
            </div>

            <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
              <h3 className="font-medium mb-4">Swarm Activity</h3>
              <ResponsiveContainer width="100%" height={250}>
                <AreaChart data={generateSwarmActivityData()}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                  <XAxis dataKey="time" stroke="#94a3b8" fontSize={12} />
                  <YAxis stroke="#94a3b8" fontSize={12} />
                  <Tooltip contentStyle={{ backgroundColor: '#1e293b', border: 'none' }} />
                  <Area type="monotone" dataKey="agents" stroke="#22d3ee" fill="#22d3ee" fillOpacity={0.3} name="Active Agents" />
                  <Area type="monotone" dataKey="tasks" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.3} name="Tasks/min" />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            {/* Agent Distribution */}
            <div className="lg:col-span-2 bg-slate-800 rounded-lg border border-slate-700 p-4">
              <h3 className="font-medium mb-4">Agent Distribution by Role</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {generateAgentRoles().map((role) => (
                  <div key={role.name} className="bg-slate-700/30 rounded-lg p-4 text-center">
                    <div className="text-2xl font-bold text-cyan-400">{role.count}</div>
                    <div className="text-sm text-slate-400">{role.name}</div>
                    <div className="text-xs text-slate-500">{role.status}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function MetricRow({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className="flex justify-between items-center">
      <span className="text-slate-400">{label}</span>
      <span className={highlight ? 'text-purple-400 font-semibold' : 'font-medium'}>{value}</span>
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

function EventRow({ event }: { event: EvolutionEvent }) {
  const typeColors: Record<string, string> = {
    mutation: 'text-amber-400',
    crossover: 'text-green-400',
    selection: 'text-blue-400',
    improvement: 'text-purple-400',
    stagnation: 'text-red-400',
  };

  return (
    <div className="flex items-center justify-between p-2 bg-slate-700/30 rounded">
      <div className="flex items-center gap-3">
        <span className={`text-sm font-medium ${typeColors[event.event_type]}`}>
          {event.event_type.toUpperCase()}
        </span>
        <span className="text-sm">{event.description}</span>
      </div>
      <div className="text-sm text-slate-400">
        Gen {event.generation} • {new Date(event.timestamp).toLocaleTimeString()}
      </div>
    </div>
  );
}

// Helper data generators
function generateFitnessData() {
  return Array.from({ length: 20 }, (_, i) => ({
    generation: (i + 1) * 5,
    best: 0.5 + 0.4 * (1 - Math.exp(-0.1 * i)) + Math.random() * 0.02,
    average: 0.3 + 0.35 * (1 - Math.exp(-0.08 * i)) + Math.random() * 0.05,
  }));
}

function generateAdasProgressData() {
  return Array.from({ length: 15 }, (_, i) => ({
    iteration: (i + 1) * 100,
    score: 0.6 + 0.35 * (1 - Math.exp(-0.15 * i)) + Math.random() * 0.03,
  }));
}

function generateDgmDiversityData() {
  return Array.from({ length: 20 }, (_, i) => ({
    generation: (i + 1) * 10,
    diversity: 90 - 40 * (1 - Math.exp(-0.05 * i)) + Math.random() * 5,
    fitness: 30 + 60 * (1 - Math.exp(-0.08 * i)) + Math.random() * 5,
  }));
}

function generateSwarmActivityData() {
  const times = ['5m', '10m', '15m', '20m', '25m', '30m'];
  return times.map(time => ({
    time,
    agents: 40 + Math.floor(Math.random() * 20),
    tasks: 150 + Math.floor(Math.random() * 100),
  }));
}

function generateTopArchitectures() {
  return [
    { id: '1', name: 'Arch-A7X9', layers: 24, params: 125, score: 0.9423 },
    { id: '2', name: 'Arch-B3K2', layers: 18, params: 89, score: 0.9287 },
    { id: '3', name: 'Arch-C1M5', layers: 32, params: 156, score: 0.9156 },
  ];
}

function generateAgentRoles() {
  return [
    { name: 'Explorers', count: 15, status: 'Active' },
    { name: 'Exploiters', count: 12, status: 'Active' },
    { name: 'Coordinators', count: 4, status: 'Active' },
    { name: 'Validators', count: 8, status: 'Active' },
  ];
}

/**
 * Agent Grid Component
 *
 * Grid display of swarm agents with filtering capabilities.
 */

import React from 'react';
import { Filter, RefreshCw, Loader2 } from 'lucide-react';
import { useStratoSwarmStore, SwarmAgent, AgentStatus, AgentTier } from '../../stores/stratoswarmStore';
import { AgentCard } from './AgentCard';

interface AgentGridProps {
  onSelectAgent?: (agent: SwarmAgent) => void;
  compact?: boolean;
}

export function AgentGrid({ onSelectAgent, compact = false }: AgentGridProps) {
  const {
    agents,
    isLoading,
    statusFilter,
    tierFilter,
    setStatusFilter,
    setTierFilter,
    fetchAgents,
    triggerEvolution,
  } = useStratoSwarmStore();

  React.useEffect(() => {
    fetchAgents();
    const interval = setInterval(() => fetchAgents(), 10000);
    return () => clearInterval(interval);
  }, [fetchAgents]);

  const statusOptions: (AgentStatus | null)[] = [null, 'idle', 'working', 'learning', 'evolving', 'offline', 'error'];
  const tierOptions: (AgentTier | null)[] = [null, 'bronze', 'silver', 'gold', 'platinum', 'diamond'];

  const handleEvolve = async (agentId: string) => {
    try {
      await triggerEvolution(agentId);
    } catch (err) {
      console.error('Evolution failed:', err);
    }
  };

  return (
    <div className="space-y-4">
      {/* Filters */}
      <div className="bg-[#0f1729] rounded-lg border border-[#1e293b] p-3 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Filter size={14} className="text-gray-400" />
            <span className="text-xs text-gray-400">Filters:</span>
          </div>
          <select
            value={statusFilter || ''}
            onChange={(e) => setStatusFilter(e.target.value as AgentStatus | null || null)}
            className="bg-[#1e293b] text-gray-300 text-xs rounded px-2 py-1 border border-[#334155] focus:outline-none focus:ring-1 focus:ring-[#00d4ff]"
          >
            {statusOptions.map((status) => (
              <option key={status || 'all'} value={status || ''}>
                {status ? status.charAt(0).toUpperCase() + status.slice(1) : 'All Status'}
              </option>
            ))}
          </select>
          <select
            value={tierFilter || ''}
            onChange={(e) => setTierFilter(e.target.value as AgentTier | null || null)}
            className="bg-[#1e293b] text-gray-300 text-xs rounded px-2 py-1 border border-[#334155] focus:outline-none focus:ring-1 focus:ring-[#00d4ff]"
          >
            {tierOptions.map((tier) => (
              <option key={tier || 'all'} value={tier || ''}>
                {tier ? tier.charAt(0).toUpperCase() + tier.slice(1) : 'All Tiers'}
              </option>
            ))}
          </select>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-400">{agents.length} agents</span>
          <button
            onClick={() => fetchAgents()}
            disabled={isLoading}
            className="p-1.5 text-gray-400 hover:text-white hover:bg-[#1e293b] rounded transition-colors disabled:opacity-50"
          >
            <RefreshCw size={14} className={isLoading ? 'animate-spin' : ''} />
          </button>
        </div>
      </div>

      {/* Grid */}
      {isLoading && agents.length === 0 ? (
        <div className="flex items-center justify-center h-48">
          <Loader2 className="h-6 w-6 animate-spin text-[#00d4ff]" />
        </div>
      ) : agents.length === 0 ? (
        <div className="flex items-center justify-center h-48 text-gray-500">
          No agents found
        </div>
      ) : (
        <div className={`grid gap-4 ${compact ? 'grid-cols-2 lg:grid-cols-3 xl:grid-cols-4' : 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3'}`}>
          {agents.map((agent) => (
            <AgentCard
              key={agent.id}
              agent={agent}
              onSelect={onSelectAgent}
              onEvolve={handleEvolve}
              compact={compact}
            />
          ))}
        </div>
      )}
    </div>
  );
}

/**
 * Agent Card Component
 *
 * Displays individual swarm agent information with XP progress and evolution status.
 */

import React from 'react';
import {
  Bot,
  Cpu,
  HardDrive,
  Network,
  Shield,
  BarChart3,
  Brain,
  Clock,
  AlertTriangle,
  Loader2,
  WifiOff,
  Zap,
} from 'lucide-react';
import { SwarmAgent, AgentTier, AgentStatus, AgentSpecialization, getTierColor, getNextTier, getTierXpThreshold } from '../../stores/stratoswarmStore';
import { XPProgressBar } from './XPProgressBar';
import { EvolutionButton } from './EvolutionButton';

// Status icon mapping
function StatusIcon({ status }: { status: AgentStatus }) {
  const icons: Record<AgentStatus, React.ReactNode> = {
    idle: <Clock size={16} className="text-gray-400" />,
    working: <Loader2 size={16} className="text-green-400 animate-spin" />,
    learning: <Brain size={16} className="text-purple-400" />,
    evolving: <Zap size={16} className="text-yellow-400 animate-pulse" />,
    offline: <WifiOff size={16} className="text-gray-500" />,
    error: <AlertTriangle size={16} className="text-red-400" />,
  };
  return <>{icons[status]}</>;
}

// Specialization icon mapping
function SpecializationIcon({ spec }: { spec: AgentSpecialization }) {
  const icons: Record<AgentSpecialization, React.ReactNode> = {
    general: <Bot size={20} />,
    compute: <Cpu size={20} />,
    storage: <HardDrive size={20} />,
    network: <Network size={20} />,
    security: <Shield size={20} />,
    analytics: <BarChart3 size={20} />,
    machine_learning: <Brain size={20} />,
  };
  return <>{icons[spec]}</>;
}

// Tier badge component
function TierBadge({ tier }: { tier: AgentTier }) {
  const color = getTierColor(tier);
  return (
    <span
      className="px-2 py-0.5 rounded text-xs font-bold uppercase"
      style={{
        backgroundColor: `${color}20`,
        color: color,
        border: `1px solid ${color}40`,
      }}
    >
      {tier}
    </span>
  );
}

interface AgentCardProps {
  agent: SwarmAgent;
  onSelect?: (agent: SwarmAgent) => void;
  onEvolve?: (agentId: string) => Promise<void>;
  compact?: boolean;
}

export function AgentCard({ agent, onSelect, onEvolve, compact = false }: AgentCardProps) {
  const [isEvolving, setIsEvolving] = React.useState(false);

  const nextTier = getNextTier(agent.tier);
  const nextTierXp = nextTier ? getTierXpThreshold(nextTier) : agent.xp;
  const canEvolve = nextTier && agent.xp >= nextTierXp;

  const handleEvolve = async () => {
    if (!onEvolve || isEvolving) return;
    setIsEvolving(true);
    try {
      await onEvolve(agent.id);
    } finally {
      setIsEvolving(false);
    }
  };

  if (compact) {
    return (
      <div
        className="bg-[#0f1729] rounded-lg border border-[#1e293b] p-3 hover:border-[#00d4ff]/50 cursor-pointer transition-colors"
        onClick={() => onSelect?.(agent)}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div
              className="w-8 h-8 rounded-full flex items-center justify-center"
              style={{ backgroundColor: `${getTierColor(agent.tier)}20` }}
            >
              <SpecializationIcon spec={agent.specialization} />
            </div>
            <div>
              <div className="text-sm font-medium text-white">{agent.name}</div>
              <div className="flex items-center gap-1 text-xs text-gray-500">
                <StatusIcon status={agent.status} />
                {agent.status}
              </div>
            </div>
          </div>
          <TierBadge tier={agent.tier} />
        </div>
        <div className="mt-2">
          <XPProgressBar current={agent.xp} target={nextTierXp} tier={agent.tier} compact />
        </div>
      </div>
    );
  }

  return (
    <div
      className="bg-[#0f1729] rounded-lg border border-[#1e293b] overflow-hidden hover:border-[#00d4ff]/50 transition-colors"
      onClick={() => onSelect?.(agent)}
    >
      {/* Header */}
      <div
        className="p-4 border-b border-[#1e293b]"
        style={{
          background: `linear-gradient(135deg, ${getTierColor(agent.tier)}10 0%, transparent 50%)`,
        }}
      >
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            <div
              className="w-12 h-12 rounded-lg flex items-center justify-center"
              style={{
                backgroundColor: `${getTierColor(agent.tier)}20`,
                color: getTierColor(agent.tier),
              }}
            >
              <SpecializationIcon spec={agent.specialization} />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-white">{agent.name}</h3>
              <div className="flex items-center gap-2 mt-1">
                <TierBadge tier={agent.tier} />
                <span className="text-xs text-gray-400 capitalize">{agent.specialization.replace('_', ' ')}</span>
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <StatusIcon status={agent.status} />
            <span className="text-sm text-gray-300 capitalize">{agent.status}</span>
          </div>
        </div>
      </div>

      {/* XP Progress */}
      <div className="p-4 border-b border-[#1e293b]">
        <XPProgressBar
          current={agent.xp}
          target={nextTierXp}
          tier={agent.tier}
          nextTier={nextTier}
        />
        {canEvolve && onEvolve && (
          <div className="mt-3">
            <EvolutionButton
              canEvolve={canEvolve}
              isEvolving={isEvolving}
              currentTier={agent.tier}
              nextTier={nextTier}
              onClick={handleEvolve}
            />
          </div>
        )}
      </div>

      {/* Stats */}
      <div className="p-4">
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-gray-400">Tasks Completed</span>
            <div className="font-semibold text-white">{agent.tasks_completed.toLocaleString()}</div>
          </div>
          <div>
            <span className="text-gray-400">Success Rate</span>
            <div className="font-semibold text-white">{(agent.success_rate * 100).toFixed(1)}%</div>
          </div>
          <div>
            <span className="text-gray-400">Uptime</span>
            <div className="font-semibold text-white">{agent.uptime_hours.toFixed(1)}h</div>
          </div>
          <div>
            <span className="text-gray-400">Node</span>
            <div className="font-semibold text-white">{agent.node_id || '-'}</div>
          </div>
        </div>

        {/* Current Task */}
        {agent.current_task && (
          <div className="mt-4 p-3 bg-[#1e293b] rounded-lg">
            <span className="text-xs text-gray-400">Current Task</span>
            <div className="text-sm text-white font-mono">{agent.current_task}</div>
          </div>
        )}

        {/* Skills */}
        {agent.skills.length > 0 && (
          <div className="mt-4">
            <span className="text-xs text-gray-400 block mb-2">Skills</span>
            <div className="space-y-2">
              {agent.skills.map((skill) => (
                <div key={skill.name} className="flex items-center justify-between text-xs">
                  <span className="text-gray-300">{skill.name}</span>
                  <div className="flex items-center gap-2">
                    <div className="w-24 h-1.5 bg-[#1e293b] rounded-full overflow-hidden">
                      <div
                        className="h-full bg-[#00d4ff]"
                        style={{ width: `${skill.proficiency * 100}%` }}
                      />
                    </div>
                    <span className="text-gray-400 w-8 text-right">
                      {Math.round(skill.proficiency * 100)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

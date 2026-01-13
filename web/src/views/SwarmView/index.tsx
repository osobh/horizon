/**
 * Swarm View
 *
 * Main view for StratoSwarm agent visualization, XP tracking, and evolution.
 */

import { useState, useEffect } from 'react';
import {
  Users,
  RefreshCw,
  Sparkles,
  Activity,
  ArrowLeft,
} from 'lucide-react';
import { useStratoSwarmStore, SwarmAgent } from '../../stores/stratoswarmStore';
import { SwarmStats, AgentGrid, AgentCard, EvolutionTimeline } from '../../components/stratoswarm';

type ViewTab = 'overview' | 'agents' | 'evolution';

export default function SwarmView() {
  const [selectedTab, setSelectedTab] = useState<ViewTab>('overview');
  const [selectedAgent, setSelectedAgent] = useState<SwarmAgent | null>(null);
  const {
    status,
    isLoading,
    error,
    fetchStatus,
    fetchStats,
    fetchAgents,
    fetchEvolutionEvents,
    triggerEvolution,
    clearError,
  } = useStratoSwarmStore();

  useEffect(() => {
    fetchStatus();
    fetchStats();
    fetchAgents();
    fetchEvolutionEvents(10);
  }, [fetchStatus, fetchStats, fetchAgents, fetchEvolutionEvents]);

  const handleRefresh = () => {
    fetchStatus();
    fetchStats();
    fetchAgents();
    fetchEvolutionEvents(10);
  };

  const handleEvolve = async (agentId: string) => {
    try {
      await triggerEvolution(agentId);
    } catch (err) {
      console.error('Evolution failed:', err);
    }
  };

  const handleSelectAgent = (agent: SwarmAgent) => {
    setSelectedAgent(agent);
  };

  const handleBackToGrid = () => {
    setSelectedAgent(null);
  };

  return (
    <div className="h-full flex flex-col bg-[#0a0f1a] text-white">
      {/* Header */}
      <div className="h-16 bg-[#0f1729] border-b border-[#1e293b] flex items-center px-6">
        <Users className="w-6 h-6 text-purple-400 mr-3" />
        <h1 className="text-xl font-semibold">StratoSwarm</h1>
        <div className="flex-1" />
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${status.connected ? 'bg-green-400' : 'bg-red-400'}`} />
          <span className="text-sm text-gray-400">
            {status.connected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
        <button
          onClick={handleRefresh}
          disabled={isLoading}
          className="ml-4 p-2 hover:bg-[#1e293b] rounded-lg transition-colors"
        >
          <RefreshCw className={`w-5 h-5 ${isLoading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Tabs */}
      <div className="border-b border-[#1e293b] px-6">
        <div className="flex gap-4">
          {[
            { id: 'overview', label: 'Overview', icon: Activity },
            { id: 'agents', label: 'Agents', icon: Users },
            { id: 'evolution', label: 'Evolution', icon: Sparkles },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => {
                setSelectedTab(tab.id as ViewTab);
                setSelectedAgent(null);
              }}
              className={`py-3 px-4 border-b-2 transition-colors flex items-center gap-2 ${
                selectedTab === tab.id
                  ? 'border-purple-400 text-purple-400'
                  : 'border-transparent text-gray-400 hover:text-white'
              }`}
            >
              <tab.icon size={16} />
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-auto p-6">
        {error && (
          <div className="mb-4 p-4 bg-red-500/20 border border-red-500 rounded-lg text-red-400 flex items-center justify-between">
            {error}
            <button onClick={clearError} className="text-red-300 hover:text-white">
              Dismiss
            </button>
          </div>
        )}

        {selectedTab === 'overview' && (
          <div className="space-y-6">
            <SwarmStats />
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div>
                <h2 className="text-lg font-semibold text-white mb-4">Active Agents</h2>
                <AgentGrid onSelectAgent={handleSelectAgent} compact />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-white mb-4">Recent Evolutions</h2>
                <EvolutionTimeline />
              </div>
            </div>
          </div>
        )}

        {selectedTab === 'agents' && (
          <div>
            {selectedAgent ? (
              <div className="space-y-4">
                <button
                  onClick={handleBackToGrid}
                  className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors"
                >
                  <ArrowLeft size={16} />
                  Back to Agents
                </button>
                <div className="max-w-2xl">
                  <AgentCard
                    agent={selectedAgent}
                    onEvolve={handleEvolve}
                  />
                </div>
              </div>
            ) : (
              <AgentGrid onSelectAgent={handleSelectAgent} />
            )}
          </div>
        )}

        {selectedTab === 'evolution' && (
          <div className="space-y-6">
            <div className="bg-[#0f1729] rounded-lg border border-[#1e293b] p-6">
              <h2 className="text-lg font-semibold text-white mb-4">Evolution System</h2>
              <p className="text-gray-400 text-sm mb-6">
                Agents gain experience (XP) by completing tasks. When an agent accumulates enough XP,
                they become eligible for evolution to the next tier. Each tier unlocks enhanced capabilities
                and increased task efficiency.
              </p>
              <div className="grid grid-cols-5 gap-4">
                {[
                  { tier: 'Bronze', xp: '0 XP', desc: 'Starting tier', color: '#cd7f32' },
                  { tier: 'Silver', xp: '1,000 XP', desc: 'Basic proficiency', color: '#c0c0c0' },
                  { tier: 'Gold', xp: '5,000 XP', desc: 'Advanced skills', color: '#ffd700' },
                  { tier: 'Platinum', xp: '15,000 XP', desc: 'Expert level', color: '#e5e4e2' },
                  { tier: 'Diamond', xp: '50,000 XP', desc: 'Maximum tier', color: '#b9f2ff' },
                ].map((item, idx) => (
                  <div
                    key={item.tier}
                    className="text-center p-4 rounded-lg"
                    style={{ backgroundColor: `${item.color}10` }}
                  >
                    <div
                      className="w-12 h-12 rounded-full mx-auto mb-2 flex items-center justify-center text-lg font-bold"
                      style={{ backgroundColor: `${item.color}30`, color: item.color }}
                    >
                      {idx + 1}
                    </div>
                    <div className="font-medium" style={{ color: item.color }}>
                      {item.tier}
                    </div>
                    <div className="text-xs text-gray-400 mt-1">{item.xp}</div>
                    <div className="text-xs text-gray-500">{item.desc}</div>
                  </div>
                ))}
              </div>
            </div>
            <EvolutionTimeline />
          </div>
        )}
      </div>
    </div>
  );
}

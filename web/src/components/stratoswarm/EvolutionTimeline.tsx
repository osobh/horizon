/**
 * Evolution Timeline Component
 *
 * Shows recent agent evolution events.
 */

import React from 'react';
import { Sparkles, ArrowRight, Loader2 } from 'lucide-react';
import { useStratoSwarmStore, getTierColor } from '../../stores/stratoswarmStore';

export function EvolutionTimeline() {
  const { evolutionEvents, isLoading, fetchEvolutionEvents } = useStratoSwarmStore();

  React.useEffect(() => {
    fetchEvolutionEvents(10);
  }, [fetchEvolutionEvents]);

  if (isLoading && evolutionEvents.length === 0) {
    return (
      <div className="flex items-center justify-center h-32">
        <Loader2 className="h-5 w-5 animate-spin text-[#00d4ff]" />
      </div>
    );
  }

  if (evolutionEvents.length === 0) {
    return (
      <div className="bg-[#0f1729] rounded-lg border border-[#1e293b] p-6 text-center text-gray-500">
        No evolution events yet
      </div>
    );
  }

  return (
    <div className="bg-[#0f1729] rounded-lg border border-[#1e293b] overflow-hidden">
      <div className="px-4 py-3 border-b border-[#1e293b] flex items-center gap-2">
        <Sparkles size={16} className="text-yellow-400" />
        <h3 className="text-sm font-medium text-white">Recent Evolutions</h3>
      </div>
      <div className="divide-y divide-[#1e293b]">
        {evolutionEvents.map((event) => {
          const fromColor = getTierColor(event.from_tier);
          const toColor = getTierColor(event.to_tier);
          const timestamp = new Date(event.timestamp);

          return (
            <div key={event.id} className="p-4 hover:bg-[#1e293b]/50 transition-colors">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div
                    className="w-10 h-10 rounded-full flex items-center justify-center"
                    style={{
                      background: `linear-gradient(135deg, ${fromColor}40, ${toColor}40)`,
                    }}
                  >
                    <Sparkles size={18} style={{ color: toColor }} />
                  </div>
                  <div>
                    <div className="text-sm font-medium text-white">{event.agent_name}</div>
                    <div className="flex items-center gap-2 text-xs">
                      <span
                        className="uppercase font-medium"
                        style={{ color: fromColor }}
                      >
                        {event.from_tier}
                      </span>
                      <ArrowRight size={12} className="text-gray-500" />
                      <span
                        className="uppercase font-medium"
                        style={{ color: toColor }}
                      >
                        {event.to_tier}
                      </span>
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-sm text-gray-400">
                    {event.xp_at_evolution.toLocaleString()} XP
                  </div>
                  <div className="text-xs text-gray-500">
                    {timestamp.toLocaleDateString()} {timestamp.toLocaleTimeString()}
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

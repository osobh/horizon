/**
 * Evolution Button Component
 *
 * Button to trigger agent tier evolution with visual feedback.
 */

import { Sparkles, Loader2, ArrowRight } from 'lucide-react';
import { AgentTier, getTierColor } from '../../stores/stratoswarmStore';

interface EvolutionButtonProps {
  canEvolve: boolean;
  isEvolving: boolean;
  currentTier: AgentTier;
  nextTier: AgentTier | null;
  onClick: () => void;
}

export function EvolutionButton({
  canEvolve,
  isEvolving,
  currentTier,
  nextTier,
  onClick,
}: EvolutionButtonProps) {
  if (!nextTier) {
    return (
      <div className="flex items-center justify-center gap-2 py-2 px-4 bg-[#1e293b] rounded-lg text-sm text-gray-400">
        <Sparkles size={16} className="text-[#b9f2ff]" />
        Maximum Tier Achieved
      </div>
    );
  }

  const currentColor = getTierColor(currentTier);
  const nextColor = getTierColor(nextTier);

  return (
    <button
      onClick={onClick}
      disabled={!canEvolve || isEvolving}
      className={`
        w-full flex items-center justify-center gap-2 py-2 px-4 rounded-lg text-sm font-medium
        transition-all duration-300 transform
        ${canEvolve && !isEvolving
          ? 'hover:scale-[1.02] active:scale-[0.98] cursor-pointer'
          : 'cursor-not-allowed opacity-50'
        }
      `}
      style={{
        background: canEvolve
          ? `linear-gradient(135deg, ${currentColor}, ${nextColor})`
          : '#1e293b',
        color: canEvolve ? '#0f1729' : '#64748b',
        boxShadow: canEvolve ? `0 0 20px ${nextColor}40` : 'none',
      }}
    >
      {isEvolving ? (
        <>
          <Loader2 size={16} className="animate-spin" />
          Evolving...
        </>
      ) : canEvolve ? (
        <>
          <Sparkles size={16} />
          Evolve to {nextTier.charAt(0).toUpperCase() + nextTier.slice(1)}
          <ArrowRight size={16} />
        </>
      ) : (
        <>
          <Sparkles size={16} />
          Need More XP
        </>
      )}
    </button>
  );
}

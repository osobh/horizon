/**
 * XP Progress Bar Component
 *
 * Displays agent experience progress towards next tier.
 */

import { AgentTier, getTierColor } from '../../stores/stratoswarmStore';

interface XPProgressBarProps {
  current: number;
  target: number;
  tier: AgentTier;
  nextTier?: AgentTier | null;
  compact?: boolean;
}

export function XPProgressBar({ current, target, tier, nextTier, compact = false }: XPProgressBarProps) {
  const progress = Math.min((current / target) * 100, 100);
  const tierColor = getTierColor(tier);
  const nextTierColor = nextTier ? getTierColor(nextTier) : tierColor;

  if (compact) {
    return (
      <div className="flex items-center gap-2">
        <div className="flex-1 h-1 bg-[#1e293b] rounded-full overflow-hidden">
          <div
            className="h-full transition-all duration-500"
            style={{
              width: `${progress}%`,
              background: `linear-gradient(90deg, ${tierColor}, ${nextTierColor})`,
            }}
          />
        </div>
        <span className="text-xs text-gray-400 w-16 text-right">
          {current.toLocaleString()} XP
        </span>
      </div>
    );
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-sm">
        <span className="text-gray-400">Experience Points</span>
        <span className="text-white font-medium">
          {current.toLocaleString()} / {target.toLocaleString()} XP
        </span>
      </div>
      <div className="h-3 bg-[#1e293b] rounded-full overflow-hidden relative">
        <div
          className="h-full transition-all duration-500 relative"
          style={{
            width: `${progress}%`,
            background: `linear-gradient(90deg, ${tierColor}, ${nextTierColor})`,
          }}
        >
          {/* Shine effect */}
          <div
            className="absolute inset-0 opacity-30"
            style={{
              background: 'linear-gradient(90deg, transparent, white 50%, transparent)',
              animation: 'shine 2s infinite',
            }}
          />
        </div>
        {/* Milestone markers */}
        {[25, 50, 75].map((milestone) => (
          <div
            key={milestone}
            className="absolute top-0 bottom-0 w-px bg-[#334155]"
            style={{ left: `${milestone}%` }}
          />
        ))}
      </div>
      {nextTier && (
        <div className="flex items-center justify-between text-xs text-gray-500">
          <span style={{ color: tierColor }} className="uppercase font-medium">
            {tier}
          </span>
          <span className="text-gray-400">{Math.round(progress)}% to next tier</span>
          <span style={{ color: nextTierColor }} className="uppercase font-medium">
            {nextTier}
          </span>
        </div>
      )}
    </div>
  );
}

// Add shine animation to global styles or use styled-components
const styles = `
@keyframes shine {
  0% { transform: translateX(-100%); }
  100% { transform: translateX(200%); }
}
`;

// Inject styles (would be better in a CSS file)
if (typeof document !== 'undefined') {
  const styleSheet = document.createElement('style');
  styleSheet.textContent = styles;
  document.head.appendChild(styleSheet);
}

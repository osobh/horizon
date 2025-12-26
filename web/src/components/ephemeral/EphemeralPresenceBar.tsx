/**
 * EphemeralPresenceBar - Shows ephemeral collaborators in the current session
 *
 * Features:
 * - Participant avatars with activity status
 * - Connection quality indicators
 * - Session countdown timer
 * - Quick actions (invite more, end session)
 */

import { useState, useEffect, useMemo } from 'react';
import {
  Users,
  Clock,
  Wifi,
  WifiOff,
  Signal,
  SignalHigh,
  SignalMedium,
  SignalLow,
  UserPlus,
  X,
  MoreVertical,
  LogOut,
} from 'lucide-react';
import {
  useEphemeralStore,
  EphemeralParticipant,
} from '../../stores/ephemeralStore';

interface EphemeralPresenceBarProps {
  sessionId: string;
  onInvite?: () => void;
  onEndSession?: () => void;
  onLeave?: () => void;
  isHost?: boolean;
  className?: string;
}

const QUALITY_ICONS = {
  excellent: Signal,
  good: SignalHigh,
  fair: SignalMedium,
  poor: SignalLow,
};

const QUALITY_COLORS = {
  excellent: 'text-green-400',
  good: 'text-green-400',
  fair: 'text-yellow-400',
  poor: 'text-red-400',
};

function ParticipantAvatar({ participant }: { participant: EphemeralParticipant }) {
  const QualityIcon = QUALITY_ICONS[participant.connection_quality];
  const qualityColor = QUALITY_COLORS[participant.connection_quality];

  // Get initials from display name
  const initials = participant.display_name
    .split(' ')
    .map(n => n[0])
    .join('')
    .toUpperCase()
    .slice(0, 2);

  // Generate consistent color from participant ID
  const hue = participant.id.split('').reduce((acc, char) => acc + char.charCodeAt(0), 0) % 360;

  return (
    <div className="relative group">
      <div
        className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-medium text-white border-2 ${
          participant.is_active ? 'border-green-400' : 'border-gray-600'
        }`}
        style={{ backgroundColor: `hsl(${hue}, 60%, 40%)` }}
      >
        {initials}
      </div>

      {/* Activity indicator */}
      <div
        className={`absolute -bottom-0.5 -right-0.5 w-3 h-3 rounded-full border-2 border-gray-800 ${
          participant.is_active ? 'bg-green-400' : 'bg-gray-500'
        }`}
      />

      {/* Tooltip */}
      <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-2 bg-gray-900 rounded-lg shadow-lg opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none whitespace-nowrap z-10">
        <p className="text-sm text-white font-medium">{participant.display_name}</p>
        <div className="flex items-center gap-1 mt-1">
          <QualityIcon className={`w-3 h-3 ${qualityColor}`} />
          <span className={`text-xs ${qualityColor}`}>{participant.connection_quality}</span>
        </div>
        {participant.cursor_position && (
          <p className="text-xs text-gray-400 mt-1">
            {participant.cursor_position.file}:{participant.cursor_position.line}
          </p>
        )}
      </div>
    </div>
  );
}

function CountdownTimer({ expiresAt }: { expiresAt: string }) {
  const [timeLeft, setTimeLeft] = useState('');
  const [isUrgent, setIsUrgent] = useState(false);

  useEffect(() => {
    const updateTimer = () => {
      const now = new Date();
      const expiry = new Date(expiresAt);
      const diff = expiry.getTime() - now.getTime();

      if (diff <= 0) {
        setTimeLeft('Expired');
        setIsUrgent(true);
        return;
      }

      const hours = Math.floor(diff / (1000 * 60 * 60));
      const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
      const seconds = Math.floor((diff % (1000 * 60)) / 1000);

      if (hours > 0) {
        setTimeLeft(`${hours}h ${minutes}m`);
        setIsUrgent(false);
      } else if (minutes > 5) {
        setTimeLeft(`${minutes}m`);
        setIsUrgent(false);
      } else {
        setTimeLeft(`${minutes}:${seconds.toString().padStart(2, '0')}`);
        setIsUrgent(true);
      }
    };

    updateTimer();
    const interval = setInterval(updateTimer, 1000);
    return () => clearInterval(interval);
  }, [expiresAt]);

  return (
    <div
      className={`flex items-center gap-1 px-2 py-1 rounded text-xs ${
        isUrgent ? 'bg-red-900/50 text-red-400' : 'bg-gray-700 text-gray-300'
      }`}
    >
      <Clock className="w-3 h-3" />
      <span>{timeLeft}</span>
    </div>
  );
}

export default function EphemeralPresenceBar({
  sessionId,
  onInvite,
  onEndSession,
  onLeave,
  isHost = false,
  className = '',
}: EphemeralPresenceBarProps) {
  const { sessions, participants, qualityMetrics, getPresence } = useEphemeralStore();

  const [showMenu, setShowMenu] = useState(false);

  const session = sessions.get(sessionId);
  const participantList = participants.get(sessionId) || [];

  // Fetch presence on mount
  useEffect(() => {
    getPresence(sessionId);
  }, [sessionId, getPresence]);

  // Calculate connection quality summary
  const connectionSummary = useMemo(() => {
    const metrics = Array.from(qualityMetrics.values());
    if (metrics.length === 0) return 'unknown';

    const p2pCount = metrics.filter(m => m.connection_type === 'p2p').length;
    const avgLatency = metrics.reduce((sum, m) => sum + m.latency_ms, 0) / metrics.length;

    if (avgLatency < 50 && p2pCount === metrics.length) return 'excellent';
    if (avgLatency < 100) return 'good';
    if (avgLatency < 200) return 'fair';
    return 'poor';
  }, [qualityMetrics]);

  if (!session) {
    return null;
  }

  const ConnectionIcon = participantList.length > 0 ? Wifi : WifiOff;

  return (
    <div
      className={`flex items-center gap-3 bg-gray-800/80 backdrop-blur-sm border border-gray-700 rounded-lg px-3 py-2 ${className}`}
    >
      {/* Session indicator */}
      <div className="flex items-center gap-2">
        <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
        <span className="text-sm text-gray-300 font-medium">{session.name}</span>
      </div>

      <div className="w-px h-4 bg-gray-700" />

      {/* Participants */}
      <div className="flex items-center gap-1">
        <Users className="w-4 h-4 text-gray-400" />
        <div className="flex -space-x-2">
          {participantList.slice(0, 5).map(participant => (
            <ParticipantAvatar key={participant.id} participant={participant} />
          ))}
          {participantList.length > 5 && (
            <div className="w-8 h-8 rounded-full bg-gray-700 flex items-center justify-center text-xs text-gray-300 border-2 border-gray-800">
              +{participantList.length - 5}
            </div>
          )}
        </div>
        <span className="text-xs text-gray-500 ml-1">
          {participantList.length}/{session.max_participants}
        </span>
      </div>

      <div className="w-px h-4 bg-gray-700" />

      {/* Connection quality */}
      <div className="flex items-center gap-1">
        <ConnectionIcon
          className={`w-4 h-4 ${
            connectionSummary === 'excellent' || connectionSummary === 'good'
              ? 'text-green-400'
              : connectionSummary === 'fair'
              ? 'text-yellow-400'
              : 'text-red-400'
          }`}
        />
      </div>

      {/* Countdown */}
      <CountdownTimer expiresAt={session.expires_at} />

      <div className="flex-1" />

      {/* Actions */}
      <div className="flex items-center gap-1">
        {onInvite && (
          <button
            onClick={onInvite}
            className="p-1.5 hover:bg-gray-700 rounded transition-colors"
            title="Invite more"
          >
            <UserPlus className="w-4 h-4 text-gray-400" />
          </button>
        )}

        <div className="relative">
          <button
            onClick={() => setShowMenu(!showMenu)}
            className="p-1.5 hover:bg-gray-700 rounded transition-colors"
          >
            <MoreVertical className="w-4 h-4 text-gray-400" />
          </button>

          {showMenu && (
            <div className="absolute right-0 top-full mt-1 bg-gray-900 border border-gray-700 rounded-lg shadow-lg py-1 min-w-[140px] z-20">
              {onLeave && !isHost && (
                <button
                  onClick={() => {
                    setShowMenu(false);
                    onLeave();
                  }}
                  className="w-full flex items-center gap-2 px-3 py-2 text-sm text-gray-300 hover:bg-gray-800 transition-colors"
                >
                  <LogOut className="w-4 h-4" />
                  Leave Session
                </button>
              )}

              {onEndSession && isHost && (
                <button
                  onClick={() => {
                    setShowMenu(false);
                    onEndSession();
                  }}
                  className="w-full flex items-center gap-2 px-3 py-2 text-sm text-red-400 hover:bg-gray-800 transition-colors"
                >
                  <X className="w-4 h-4" />
                  End Session
                </button>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

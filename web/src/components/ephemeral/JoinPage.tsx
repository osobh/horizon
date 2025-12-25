/**
 * JoinPage - Landing page for joining ephemeral sessions via invite link
 *
 * Features:
 * - Parse invite token from URL
 * - Redemption code input with validation
 * - Display name entry
 * - Session join flow with error handling
 */

import { useState, useEffect, useCallback } from 'react';
import {
  Users,
  KeyRound,
  User,
  Loader2,
  AlertCircle,
  CheckCircle,
  Clock,
  Shield,
  ArrowRight,
} from 'lucide-react';
import { useEphemeralStore, EphemeralSession } from '../../stores/ephemeralStore';

interface JoinPageProps {
  /** Invite token from URL (if any) */
  inviteToken?: string;
  /** Callback after successful join */
  onJoined?: (session: EphemeralSession) => void;
  /** Callback on cancel */
  onCancel?: () => void;
  className?: string;
}

type JoinStatus = 'input' | 'joining' | 'success' | 'error';

export default function JoinPage({
  inviteToken: initialToken,
  onJoined,
  onCancel,
  className = '',
}: JoinPageProps) {
  const { joinSession, loading, error: storeError } = useEphemeralStore();

  const [inviteToken, setInviteToken] = useState(initialToken || '');
  const [redemptionCode, setRedemptionCode] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [status, setStatus] = useState<JoinStatus>('input');
  const [error, setError] = useState<string | null>(null);
  const [joinedSession, setJoinedSession] = useState<EphemeralSession | null>(null);

  // Parse token from URL on mount
  useEffect(() => {
    if (!initialToken) {
      const params = new URLSearchParams(window.location.search);
      const token = params.get('token');
      if (token) {
        setInviteToken(token);
      }
    }
  }, [initialToken]);

  // Format redemption code as user types (XXXX-XXXX format)
  const handleCodeChange = useCallback((value: string) => {
    // Remove non-alphanumeric characters
    const cleaned = value.replace(/[^a-zA-Z0-9]/g, '').toUpperCase();

    // Add hyphen after 4 characters
    if (cleaned.length <= 4) {
      setRedemptionCode(cleaned);
    } else {
      setRedemptionCode(`${cleaned.slice(0, 4)}-${cleaned.slice(4, 8)}`);
    }
  }, []);

  const validateInputs = useCallback(() => {
    if (!inviteToken.trim()) {
      setError('Please enter or paste the invite link');
      return false;
    }

    const codeWithoutHyphen = redemptionCode.replace('-', '');
    if (codeWithoutHyphen.length !== 8) {
      setError('Redemption code must be 8 characters');
      return false;
    }

    if (!displayName.trim()) {
      setError('Please enter your name');
      return false;
    }

    return true;
  }, [inviteToken, redemptionCode, displayName]);

  const handleJoin = async () => {
    setError(null);

    if (!validateInputs()) {
      return;
    }

    try {
      setStatus('joining');

      const session = await joinSession(inviteToken, redemptionCode);

      setJoinedSession(session);
      setStatus('success');
      onJoined?.(session);
    } catch (err) {
      setError(String(err));
      setStatus('error');
    }
  };

  const handleRetry = () => {
    setStatus('input');
    setError(null);
  };

  // Success state
  if (status === 'success' && joinedSession) {
    return (
      <div className={`max-w-md mx-auto ${className}`}>
        <div className="bg-gray-800 rounded-lg p-8 text-center">
          <div className="w-16 h-16 bg-green-900/30 rounded-full flex items-center justify-center mx-auto mb-4">
            <CheckCircle className="w-8 h-8 text-green-400" />
          </div>

          <h2 className="text-xl font-semibold text-white mb-2">
            Joined Successfully!
          </h2>

          <p className="text-gray-400 mb-6">
            You&apos;ve joined <span className="text-white">{joinedSession.name}</span>
          </p>

          <div className="bg-gray-900 rounded-lg p-4 mb-6 text-left">
            <div className="flex items-center gap-2 text-sm text-gray-400 mb-2">
              <Clock className="w-4 h-4" />
              <span>Expires: {new Date(joinedSession.expires_at).toLocaleString()}</span>
            </div>
            <div className="flex items-center gap-2 text-sm text-gray-400">
              <Users className="w-4 h-4" />
              <span>{joinedSession.participant_count}/{joinedSession.max_participants} participants</span>
            </div>
          </div>

          <div className="bg-blue-900/30 border border-blue-500/30 rounded-lg p-4 mb-6 text-left">
            <p className="text-sm text-blue-300 mb-2">Your permissions:</p>
            <div className="flex flex-wrap gap-2">
              {joinedSession.permissions.can_edit && (
                <span className="px-2 py-1 bg-blue-800/50 text-blue-300 rounded text-xs">Edit</span>
              )}
              {joinedSession.permissions.can_execute && (
                <span className="px-2 py-1 bg-green-800/50 text-green-300 rounded text-xs">Execute</span>
              )}
              {joinedSession.permissions.can_view_logs && (
                <span className="px-2 py-1 bg-purple-800/50 text-purple-300 rounded text-xs">View Logs</span>
              )}
              {joinedSession.permissions.can_chat && (
                <span className="px-2 py-1 bg-yellow-800/50 text-yellow-300 rounded text-xs">Chat</span>
              )}
            </div>
          </div>

          <button
            onClick={() => onJoined?.(joinedSession)}
            className="w-full flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-4 py-3 rounded-lg font-medium transition-colors"
          >
            Enter Session
            <ArrowRight className="w-5 h-5" />
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={`max-w-md mx-auto ${className}`}>
      <div className="bg-gray-800 rounded-lg p-8">
        <div className="text-center mb-8">
          <div className="w-16 h-16 bg-blue-900/30 rounded-full flex items-center justify-center mx-auto mb-4">
            <Users className="w-8 h-8 text-blue-400" />
          </div>

          <h2 className="text-xl font-semibold text-white mb-2">
            Join Session
          </h2>

          <p className="text-gray-400">
            Enter your invite details to join the collaboration session
          </p>
        </div>

        {(error || storeError) && (
          <div className="flex items-start gap-2 bg-red-900/30 border border-red-500 text-red-400 px-4 py-3 rounded-lg mb-6">
            <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5" />
            <div>
              <p className="font-medium">Failed to join</p>
              <p className="text-sm text-red-300">{error || storeError}</p>
              {status === 'error' && (
                <button
                  onClick={handleRetry}
                  className="text-sm text-red-300 hover:text-red-200 underline mt-1"
                >
                  Try again
                </button>
              )}
            </div>
          </div>
        )}

        <div className="space-y-5">
          {/* Invite Link */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Invite Link or Token
            </label>
            <input
              type="text"
              value={inviteToken}
              onChange={(e) => setInviteToken(e.target.value)}
              placeholder="Paste invite link here..."
              disabled={loading}
              className="w-full bg-gray-900 border border-gray-700 rounded-lg px-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 disabled:opacity-50"
            />
          </div>

          {/* Redemption Code */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              <KeyRound className="w-4 h-4 inline mr-1" />
              Redemption Code
            </label>
            <input
              type="text"
              value={redemptionCode}
              onChange={(e) => handleCodeChange(e.target.value)}
              placeholder="XXXX-XXXX"
              maxLength={9}
              disabled={loading}
              className="w-full bg-gray-900 border border-gray-700 rounded-lg px-4 py-3 text-white text-center text-2xl font-mono tracking-widest placeholder-gray-500 focus:outline-none focus:border-blue-500 disabled:opacity-50"
            />
            <p className="mt-1 text-xs text-gray-500">
              Enter the 8-character code shared with you
            </p>
          </div>

          {/* Display Name */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              <User className="w-4 h-4 inline mr-1" />
              Your Name
            </label>
            <input
              type="text"
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              placeholder="How should others see you?"
              disabled={loading}
              className="w-full bg-gray-900 border border-gray-700 rounded-lg px-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:border-blue-500 disabled:opacity-50"
            />
          </div>

          {/* Security Notice */}
          <div className="flex items-start gap-2 bg-gray-900 rounded-lg p-3">
            <Shield className="w-4 h-4 text-gray-400 mt-0.5 flex-shrink-0" />
            <p className="text-xs text-gray-400">
              Your access is time-limited and all actions are logged.
              The session host can revoke access at any time.
            </p>
          </div>

          {/* Action Buttons */}
          <div className="flex gap-3">
            {onCancel && (
              <button
                onClick={onCancel}
                disabled={loading}
                className="flex-1 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 text-white px-4 py-3 rounded-lg font-medium transition-colors"
              >
                Cancel
              </button>
            )}

            <button
              onClick={handleJoin}
              disabled={loading || !inviteToken || !redemptionCode || !displayName}
              className="flex-1 flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 disabled:cursor-not-allowed text-white px-4 py-3 rounded-lg font-medium transition-colors"
            >
              {loading || status === 'joining' ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  Joining...
                </>
              ) : (
                <>
                  Join Session
                  <ArrowRight className="w-5 h-5" />
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

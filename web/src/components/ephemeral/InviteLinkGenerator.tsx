/**
 * InviteLinkGenerator - Create and share ephemeral session invite links
 *
 * Allows users to:
 * - Select session type (pair programming, notebook, etc.)
 * - Configure permissions
 * - Generate invite URL with redemption code
 * - Copy or share the invite link
 */

import { useState, useCallback } from 'react';
import {
  Link,
  Copy,
  Check,
  Users,
  Clock,
  Shield,
  Code2,
  FileText,
  Eye,
  Play,
  MessageSquare,
  Loader2,
  QrCode,
  Share2,
} from 'lucide-react';
import {
  useEphemeralStore,
  SessionType,
  EphemeralPermissions,
  InviteLink,
} from '../../stores/ephemeralStore';

interface InviteLinkGeneratorProps {
  onSessionCreated?: (sessionId: string, invite: InviteLink) => void;
  className?: string;
}

const SESSION_TYPE_OPTIONS: { value: SessionType; label: string; icon: React.ElementType; description: string }[] = [
  {
    value: 'pair_programming',
    label: 'Pair Programming',
    icon: Code2,
    description: 'Real-time code editing and execution',
  },
  {
    value: 'notebook_collaboration',
    label: 'Notebook Collaboration',
    icon: FileText,
    description: 'Shared Jupyter notebook editing',
  },
  {
    value: 'training_observer',
    label: 'Training Observer',
    icon: Play,
    description: 'View training runs and metrics',
  },
  {
    value: 'read_only',
    label: 'Read Only',
    icon: Eye,
    description: 'View-only access to session',
  },
];

const DEFAULT_PERMISSIONS: Record<SessionType, Partial<EphemeralPermissions>> = {
  pair_programming: {
    can_edit: true,
    can_execute: true,
    can_view_logs: true,
    can_chat: true,
  },
  notebook_collaboration: {
    can_edit: true,
    can_execute: true,
    can_view_logs: true,
    can_chat: true,
  },
  training_observer: {
    can_edit: false,
    can_execute: false,
    can_view_logs: true,
    can_chat: true,
  },
  read_only: {
    can_edit: false,
    can_execute: false,
    can_view_logs: false,
    can_chat: false,
  },
};

export default function InviteLinkGenerator({
  onSessionCreated,
  className = '',
}: InviteLinkGeneratorProps) {
  const { createSession, generateInviteLink, loading, error } = useEphemeralStore();

  const [sessionType, setSessionType] = useState<SessionType>('pair_programming');
  const [sessionName, setSessionName] = useState('');
  const [ttlMinutes, setTtlMinutes] = useState(60);
  const [maxParticipants, setMaxParticipants] = useState(5);
  const [inviteExpiryMinutes, setInviteExpiryMinutes] = useState(30);
  const [maxInviteUses, setMaxInviteUses] = useState(1);

  // Permission toggles
  const [canEdit, setCanEdit] = useState(true);
  const [canExecute, setCanExecute] = useState(false);
  const [canViewLogs, setCanViewLogs] = useState(true);
  const [canChat, setCanChat] = useState(true);

  // Generated invite
  const [generatedInvite, setGeneratedInvite] = useState<InviteLink | null>(null);
  const [copied, setCopied] = useState(false);
  const [showQr, setShowQr] = useState(false);

  // Update permissions when session type changes
  const handleSessionTypeChange = useCallback((type: SessionType) => {
    setSessionType(type);
    const defaults = DEFAULT_PERMISSIONS[type];
    setCanEdit(defaults.can_edit ?? true);
    setCanExecute(defaults.can_execute ?? false);
    setCanViewLogs(defaults.can_view_logs ?? true);
    setCanChat(defaults.can_chat ?? true);
  }, []);

  const handleGenerate = async () => {
    if (!sessionName.trim()) return;

    try {
      const permissions: Partial<EphemeralPermissions> = {
        can_edit: canEdit,
        can_execute: canExecute,
        can_view_logs: canViewLogs,
        can_chat: canChat,
      };

      const session = await createSession(
        sessionType,
        sessionName,
        permissions,
        ttlMinutes,
        maxParticipants
      );

      const invite = await generateInviteLink(
        session.id,
        inviteExpiryMinutes,
        maxInviteUses
      );

      setGeneratedInvite(invite);
      onSessionCreated?.(session.id, invite);
    } catch (err) {
      console.error('Failed to generate invite:', err);
    }
  };

  const handleCopy = async () => {
    if (!generatedInvite) return;

    const inviteText = `Join my session: ${sessionName}\n\nURL: ${generatedInvite.invite_url}\nCode: ${generatedInvite.redemption_code}\n\nExpires: ${new Date(generatedInvite.expires_at).toLocaleString()}`;

    await navigator.clipboard.writeText(inviteText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleReset = () => {
    setGeneratedInvite(null);
    setSessionName('');
    setCopied(false);
  };

  if (generatedInvite) {
    return (
      <div className={`bg-gray-800 rounded-lg p-6 ${className}`}>
        <div className="flex items-center gap-2 mb-4">
          <Check className="w-5 h-5 text-green-400" />
          <h3 className="text-lg font-medium text-white">Invite Link Created</h3>
        </div>

        <div className="space-y-4">
          {/* Session Info */}
          <div className="bg-gray-900 rounded-lg p-4">
            <p className="text-sm text-gray-400 mb-1">Session</p>
            <p className="text-white font-medium">{sessionName}</p>
          </div>

          {/* Invite URL */}
          <div className="bg-gray-900 rounded-lg p-4">
            <p className="text-sm text-gray-400 mb-1">Invite URL</p>
            <p className="text-blue-400 break-all font-mono text-sm">
              {generatedInvite.invite_url}
            </p>
          </div>

          {/* Redemption Code */}
          <div className="bg-gray-900 rounded-lg p-4">
            <p className="text-sm text-gray-400 mb-1">Redemption Code</p>
            <p className="text-yellow-400 font-mono text-2xl tracking-widest">
              {generatedInvite.redemption_code}
            </p>
          </div>

          {/* Expiry Info */}
          <div className="flex items-center gap-4 text-sm text-gray-400">
            <div className="flex items-center gap-1">
              <Clock className="w-4 h-4" />
              <span>Expires: {new Date(generatedInvite.expires_at).toLocaleString()}</span>
            </div>
            <div className="flex items-center gap-1">
              <Users className="w-4 h-4" />
              <span>Uses: {generatedInvite.current_uses}/{generatedInvite.max_uses}</span>
            </div>
          </div>

          {/* Actions */}
          <div className="flex gap-2">
            <button
              onClick={handleCopy}
              className="flex-1 flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors"
            >
              {copied ? (
                <>
                  <Check className="w-4 h-4" />
                  Copied!
                </>
              ) : (
                <>
                  <Copy className="w-4 h-4" />
                  Copy Invite
                </>
              )}
            </button>

            <button
              onClick={() => setShowQr(!showQr)}
              className="flex items-center justify-center gap-2 bg-gray-700 hover:bg-gray-600 text-white px-4 py-2 rounded-lg transition-colors"
            >
              <QrCode className="w-4 h-4" />
            </button>

            <button
              onClick={handleReset}
              className="flex items-center justify-center gap-2 bg-gray-700 hover:bg-gray-600 text-white px-4 py-2 rounded-lg transition-colors"
            >
              <Share2 className="w-4 h-4" />
              New
            </button>
          </div>

          {/* QR Code Placeholder */}
          {showQr && (
            <div className="bg-white rounded-lg p-4 flex items-center justify-center">
              <div className="text-center text-gray-600">
                <QrCode className="w-32 h-32 mx-auto mb-2" />
                <p className="text-sm">QR Code for invite</p>
              </div>
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className={`bg-gray-800 rounded-lg p-6 ${className}`}>
      <div className="flex items-center gap-2 mb-6">
        <Link className="w-5 h-5 text-blue-400" />
        <h3 className="text-lg font-medium text-white">Create Invite Link</h3>
      </div>

      {error && (
        <div className="bg-red-900/30 border border-red-500 text-red-400 px-4 py-2 rounded-lg mb-4">
          {error}
        </div>
      )}

      <div className="space-y-6">
        {/* Session Type */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Session Type
          </label>
          <div className="grid grid-cols-2 gap-2">
            {SESSION_TYPE_OPTIONS.map((option) => {
              const Icon = option.icon;
              const isSelected = sessionType === option.value;
              return (
                <button
                  key={option.value}
                  onClick={() => handleSessionTypeChange(option.value)}
                  className={`p-3 rounded-lg border text-left transition-colors ${
                    isSelected
                      ? 'border-blue-500 bg-blue-900/30'
                      : 'border-gray-700 bg-gray-900 hover:border-gray-600'
                  }`}
                >
                  <div className="flex items-center gap-2 mb-1">
                    <Icon className={`w-4 h-4 ${isSelected ? 'text-blue-400' : 'text-gray-400'}`} />
                    <span className={`text-sm font-medium ${isSelected ? 'text-white' : 'text-gray-300'}`}>
                      {option.label}
                    </span>
                  </div>
                  <p className="text-xs text-gray-500">{option.description}</p>
                </button>
              );
            })}
          </div>
        </div>

        {/* Session Name */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Session Name
          </label>
          <input
            type="text"
            value={sessionName}
            onChange={(e) => setSessionName(e.target.value)}
            placeholder="My Collaboration Session"
            className="w-full bg-gray-900 border border-gray-700 rounded-lg px-4 py-2 text-white placeholder-gray-500 focus:outline-none focus:border-blue-500"
          />
        </div>

        {/* Duration & Participants */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              <Clock className="w-4 h-4 inline mr-1" />
              Session Duration
            </label>
            <select
              value={ttlMinutes}
              onChange={(e) => setTtlMinutes(Number(e.target.value))}
              className="w-full bg-gray-900 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500"
            >
              <option value={30}>30 minutes</option>
              <option value={60}>1 hour</option>
              <option value={120}>2 hours</option>
              <option value={240}>4 hours</option>
              <option value={480}>8 hours</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              <Users className="w-4 h-4 inline mr-1" />
              Max Participants
            </label>
            <select
              value={maxParticipants}
              onChange={(e) => setMaxParticipants(Number(e.target.value))}
              className="w-full bg-gray-900 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500"
            >
              <option value={2}>2 people</option>
              <option value={5}>5 people</option>
              <option value={10}>10 people</option>
              <option value={25}>25 people</option>
            </select>
          </div>
        </div>

        {/* Permissions */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            <Shield className="w-4 h-4 inline mr-1" />
            Permissions
          </label>
          <div className="grid grid-cols-2 gap-2">
            <label className={`flex items-center gap-2 p-2 rounded-lg cursor-pointer ${canEdit ? 'bg-blue-900/30' : 'bg-gray-900'}`}>
              <input
                type="checkbox"
                checked={canEdit}
                onChange={(e) => setCanEdit(e.target.checked)}
                className="rounded border-gray-600 text-blue-500 focus:ring-blue-500"
              />
              <Code2 className="w-4 h-4 text-gray-400" />
              <span className="text-sm text-gray-300">Can Edit</span>
            </label>

            <label className={`flex items-center gap-2 p-2 rounded-lg cursor-pointer ${canExecute ? 'bg-blue-900/30' : 'bg-gray-900'}`}>
              <input
                type="checkbox"
                checked={canExecute}
                onChange={(e) => setCanExecute(e.target.checked)}
                className="rounded border-gray-600 text-blue-500 focus:ring-blue-500"
              />
              <Play className="w-4 h-4 text-gray-400" />
              <span className="text-sm text-gray-300">Can Execute</span>
            </label>

            <label className={`flex items-center gap-2 p-2 rounded-lg cursor-pointer ${canViewLogs ? 'bg-blue-900/30' : 'bg-gray-900'}`}>
              <input
                type="checkbox"
                checked={canViewLogs}
                onChange={(e) => setCanViewLogs(e.target.checked)}
                className="rounded border-gray-600 text-blue-500 focus:ring-blue-500"
              />
              <FileText className="w-4 h-4 text-gray-400" />
              <span className="text-sm text-gray-300">View Logs</span>
            </label>

            <label className={`flex items-center gap-2 p-2 rounded-lg cursor-pointer ${canChat ? 'bg-blue-900/30' : 'bg-gray-900'}`}>
              <input
                type="checkbox"
                checked={canChat}
                onChange={(e) => setCanChat(e.target.checked)}
                className="rounded border-gray-600 text-blue-500 focus:ring-blue-500"
              />
              <MessageSquare className="w-4 h-4 text-gray-400" />
              <span className="text-sm text-gray-300">Can Chat</span>
            </label>
          </div>
        </div>

        {/* Invite Link Settings */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Link Expiry
            </label>
            <select
              value={inviteExpiryMinutes}
              onChange={(e) => setInviteExpiryMinutes(Number(e.target.value))}
              className="w-full bg-gray-900 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500"
            >
              <option value={15}>15 minutes</option>
              <option value={30}>30 minutes</option>
              <option value={60}>1 hour</option>
              <option value={1440}>24 hours</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Max Uses
            </label>
            <select
              value={maxInviteUses}
              onChange={(e) => setMaxInviteUses(Number(e.target.value))}
              className="w-full bg-gray-900 border border-gray-700 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500"
            >
              <option value={1}>Single use</option>
              <option value={5}>5 uses</option>
              <option value={10}>10 uses</option>
              <option value={0}>Unlimited</option>
            </select>
          </div>
        </div>

        {/* Generate Button */}
        <button
          onClick={handleGenerate}
          disabled={loading || !sessionName.trim()}
          className="w-full flex items-center justify-center gap-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 disabled:cursor-not-allowed text-white px-4 py-3 rounded-lg font-medium transition-colors"
        >
          {loading ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              Creating...
            </>
          ) : (
            <>
              <Link className="w-5 h-5" />
              Generate Invite Link
            </>
          )}
        </button>
      </div>
    </div>
  );
}

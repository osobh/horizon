import { create } from 'zustand';
import { invoke } from '@tauri-apps/api/core';
import { listen, UnlistenFn } from '@tauri-apps/api/event';

// Types matching Tauri backend
export type SessionType = 'pair_programming' | 'notebook_collaboration' | 'training_observer' | 'read_only';

export interface EphemeralPermissions {
  can_edit: boolean;
  can_execute: boolean;
  can_view_logs: boolean;
  can_chat: boolean;
  allowed_paths: string[];
  max_file_size_mb: number;
}

export interface EphemeralSession {
  id: string;
  session_type: SessionType;
  name: string;
  created_at: string;
  expires_at: string;
  creator_id: string;
  permissions: EphemeralPermissions;
  participant_count: number;
  max_participants: number;
}

export interface EphemeralParticipant {
  id: string;
  session_id: string;
  display_name: string;
  joined_at: string;
  last_activity: string;
  is_active: boolean;
  connection_quality: 'excellent' | 'good' | 'fair' | 'poor';
  cursor_position?: { file: string; line: number; column: number };
}

export interface InviteLink {
  session_id: string;
  invite_url: string;
  redemption_code: string;
  expires_at: string;
  max_uses: number;
  current_uses: number;
}

export interface QualityMetrics {
  peer_id: string;
  latency_ms: number;
  packet_loss: number;
  jitter_ms: number;
  connection_type: 'p2p' | 'relay';
}

interface EphemeralState {
  // State
  sessions: Map<string, EphemeralSession>;
  activeSessionId: string | null;
  participants: Map<string, EphemeralParticipant[]>;
  pendingInvites: Map<string, InviteLink>;
  qualityMetrics: Map<string, QualityMetrics>;
  loading: boolean;
  error: string | null;

  // Event listeners
  unlisteners: UnlistenFn[];

  // Actions
  createSession: (
    sessionType: SessionType,
    name: string,
    permissions?: Partial<EphemeralPermissions>,
    ttlMinutes?: number,
    maxParticipants?: number
  ) => Promise<EphemeralSession>;

  joinSession: (inviteToken: string, redemptionCode: string) => Promise<EphemeralSession>;

  leaveSession: (sessionId: string) => Promise<void>;

  endSession: (sessionId: string) => Promise<void>;

  generateInviteLink: (
    sessionId: string,
    expiryMinutes?: number,
    maxUses?: number
  ) => Promise<InviteLink>;

  getPresence: (sessionId: string) => Promise<EphemeralParticipant[]>;

  updateActivity: () => Promise<void>;

  setActiveSession: (sessionId: string | null) => void;

  // Subscriptions
  subscribeToPresence: (sessionId: string) => Promise<void>;
  subscribeToQuality: (sessionId: string) => Promise<void>;

  // Cleanup
  cleanup: () => void;
}

const defaultPermissions: EphemeralPermissions = {
  can_edit: true,
  can_execute: false,
  can_view_logs: true,
  can_chat: true,
  allowed_paths: [],
  max_file_size_mb: 10,
};

export const useEphemeralStore = create<EphemeralState>((set, get) => ({
  sessions: new Map(),
  activeSessionId: null,
  participants: new Map(),
  pendingInvites: new Map(),
  qualityMetrics: new Map(),
  loading: false,
  error: null,
  unlisteners: [],

  createSession: async (
    sessionType,
    name,
    permissions,
    ttlMinutes = 60,
    maxParticipants = 10
  ) => {
    try {
      set({ loading: true, error: null });

      const mergedPermissions = { ...defaultPermissions, ...permissions };

      const response = await invoke<{ session_id: string; session: EphemeralSession }>(
        'create_ephemeral_session',
        {
          sessionType,
          name,
          permissions: mergedPermissions,
          ttlMinutes,
          maxParticipants,
        }
      );

      const { sessions } = get();
      const newSessions = new Map(sessions);
      newSessions.set(response.session_id, response.session);

      set({
        sessions: newSessions,
        activeSessionId: response.session_id,
        loading: false,
      });

      return response.session;
    } catch (error) {
      set({ error: String(error), loading: false });
      throw error;
    }
  },

  joinSession: async (inviteToken, redemptionCode) => {
    try {
      set({ loading: true, error: null });

      const response = await invoke<{ session_id: string; session: EphemeralSession }>(
        'join_ephemeral_session',
        { inviteToken, redemptionCode }
      );

      const { sessions } = get();
      const newSessions = new Map(sessions);
      newSessions.set(response.session_id, response.session);

      set({
        sessions: newSessions,
        activeSessionId: response.session_id,
        loading: false,
      });

      // Start presence updates
      get().subscribeToPresence(response.session_id);

      return response.session;
    } catch (error) {
      set({ error: String(error), loading: false });
      throw error;
    }
  },

  leaveSession: async (sessionId) => {
    try {
      set({ loading: true, error: null });

      await invoke('leave_ephemeral_session', { sessionId });

      const { sessions, participants, activeSessionId } = get();
      const newSessions = new Map(sessions);
      newSessions.delete(sessionId);

      const newParticipants = new Map(participants);
      newParticipants.delete(sessionId);

      set({
        sessions: newSessions,
        participants: newParticipants,
        activeSessionId: activeSessionId === sessionId ? null : activeSessionId,
        loading: false,
      });
    } catch (error) {
      set({ error: String(error), loading: false });
      throw error;
    }
  },

  endSession: async (sessionId) => {
    try {
      set({ loading: true, error: null });

      await invoke('end_ephemeral_session', { sessionId });

      const { sessions, participants, pendingInvites, activeSessionId } = get();
      const newSessions = new Map(sessions);
      newSessions.delete(sessionId);

      const newParticipants = new Map(participants);
      newParticipants.delete(sessionId);

      const newInvites = new Map(pendingInvites);
      newInvites.delete(sessionId);

      set({
        sessions: newSessions,
        participants: newParticipants,
        pendingInvites: newInvites,
        activeSessionId: activeSessionId === sessionId ? null : activeSessionId,
        loading: false,
      });
    } catch (error) {
      set({ error: String(error), loading: false });
      throw error;
    }
  },

  generateInviteLink: async (sessionId, expiryMinutes = 30, maxUses = 1) => {
    try {
      set({ loading: true, error: null });

      const invite = await invoke<InviteLink>('generate_invite_link', {
        sessionId,
        expiryMinutes,
        maxUses,
      });

      const { pendingInvites } = get();
      const newInvites = new Map(pendingInvites);
      newInvites.set(sessionId, invite);

      set({ pendingInvites: newInvites, loading: false });

      return invite;
    } catch (error) {
      set({ error: String(error), loading: false });
      throw error;
    }
  },

  getPresence: async (sessionId) => {
    try {
      const participantList = await invoke<EphemeralParticipant[]>(
        'get_ephemeral_presence',
        { sessionId }
      );

      const { participants } = get();
      const newParticipants = new Map(participants);
      newParticipants.set(sessionId, participantList);

      set({ participants: newParticipants });

      return participantList;
    } catch (error) {
      set({ error: String(error) });
      throw error;
    }
  },

  updateActivity: async () => {
    const { activeSessionId } = get();
    if (!activeSessionId) return;

    try {
      await invoke('update_ephemeral_activity', { sessionId: activeSessionId });
    } catch (error) {
      console.error('Failed to update activity:', error);
    }
  },

  setActiveSession: (sessionId) => {
    set({ activeSessionId: sessionId });
  },

  subscribeToPresence: async (sessionId) => {
    try {
      // Listen for presence updates
      const unlistenPresence = await listen<EphemeralParticipant[]>(
        `ephemeral-presence-${sessionId}`,
        (event) => {
          const { participants } = get();
          const newParticipants = new Map(participants);
          newParticipants.set(sessionId, event.payload);
          set({ participants: newParticipants });
        }
      );

      // Listen for participant join/leave
      const unlistenJoin = await listen<EphemeralParticipant>(
        `ephemeral-participant-joined-${sessionId}`,
        (event) => {
          const { participants } = get();
          const current = participants.get(sessionId) || [];
          const newParticipants = new Map(participants);
          newParticipants.set(sessionId, [...current, event.payload]);
          set({ participants: newParticipants });
        }
      );

      const unlistenLeave = await listen<string>(
        `ephemeral-participant-left-${sessionId}`,
        (event) => {
          const { participants } = get();
          const current = participants.get(sessionId) || [];
          const newParticipants = new Map(participants);
          newParticipants.set(
            sessionId,
            current.filter(p => p.id !== event.payload)
          );
          set({ participants: newParticipants });
        }
      );

      // Store unlisteners for cleanup
      const { unlisteners } = get();
      set({ unlisteners: [...unlisteners, unlistenPresence, unlistenJoin, unlistenLeave] });

      // Fetch initial presence
      await get().getPresence(sessionId);
    } catch (error) {
      console.error('Failed to subscribe to presence:', error);
    }
  },

  subscribeToQuality: async (sessionId) => {
    try {
      const unlisten = await listen<QualityMetrics>(
        `ephemeral-quality-${sessionId}`,
        (event) => {
          const { qualityMetrics } = get();
          const newMetrics = new Map(qualityMetrics);
          newMetrics.set(event.payload.peer_id, event.payload);
          set({ qualityMetrics: newMetrics });
        }
      );

      const { unlisteners } = get();
      set({ unlisteners: [...unlisteners, unlisten] });
    } catch (error) {
      console.error('Failed to subscribe to quality metrics:', error);
    }
  },

  cleanup: () => {
    const { unlisteners } = get();
    unlisteners.forEach(unlisten => unlisten());
    set({
      unlisteners: [],
      sessions: new Map(),
      participants: new Map(),
      pendingInvites: new Map(),
      qualityMetrics: new Map(),
      activeSessionId: null,
    });
  },
}));

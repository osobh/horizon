import { useEffect, useState } from 'react';
import {
  Shield,
  Check,
  X,
  Clock,
  User,
  AlertTriangle,
} from 'lucide-react';
import { useHpcCiStore, ApprovalRequest } from '../../stores/hpcciStore';

function formatTimeAgo(ms: number): string {
  const now = Date.now();
  const diff = now - ms;
  const seconds = Math.floor(diff / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);

  if (hours > 0) return `${hours}h ago`;
  if (minutes > 0) return `${minutes}m ago`;
  return 'just now';
}

export function ApprovalPanel() {
  const { approvals, fetchApprovals, submitApproval } = useHpcCiStore();

  useEffect(() => {
    fetchApprovals();
    const interval = setInterval(fetchApprovals, 30000);
    return () => clearInterval(interval);
  }, [fetchApprovals]);

  const pendingApprovals = approvals.filter((a) => a.status === 'pending');

  if (pendingApprovals.length === 0) {
    return (
      <div className="bg-slate-800 border border-slate-700 rounded-lg p-8 text-center">
        <Shield className="w-12 h-12 text-slate-600 mx-auto mb-3" />
        <h3 className="text-lg font-medium text-slate-300 mb-1">No Pending Approvals</h3>
        <p className="text-sm text-slate-500">
          All deployment approvals have been processed
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-medium text-white flex items-center gap-2">
          <AlertTriangle className="w-5 h-5 text-yellow-400" />
          Pending Approvals
          <span className="px-2 py-0.5 text-sm bg-yellow-500/20 text-yellow-400 rounded-full">
            {pendingApprovals.length}
          </span>
        </h3>
      </div>

      <div className="space-y-3">
        {pendingApprovals.map((approval) => (
          <ApprovalCard
            key={approval.id}
            approval={approval}
            onApprove={(comment) => submitApproval(approval.id, true, comment)}
            onReject={(comment) => submitApproval(approval.id, false, comment)}
          />
        ))}
      </div>
    </div>
  );
}

interface ApprovalCardProps {
  approval: ApprovalRequest;
  onApprove: (comment?: string) => void;
  onReject: (comment?: string) => void;
}

function ApprovalCard({ approval, onApprove, onReject }: ApprovalCardProps) {
  const [comment, setComment] = useState('');
  const [showComment, setShowComment] = useState(false);

  const handleApprove = () => {
    onApprove(comment || undefined);
    setComment('');
    setShowComment(false);
  };

  const handleReject = () => {
    onReject(comment || undefined);
    setComment('');
    setShowComment(false);
  };

  return (
    <div className="bg-slate-800 border border-slate-700 rounded-lg overflow-hidden">
      <div className="p-4">
        <div className="flex items-start justify-between mb-3">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-white font-medium">Pipeline {approval.pipeline_id}</span>
              <span className="text-slate-500">-</span>
              <span className="text-orange-400 font-medium">{approval.environment}</span>
            </div>
            <div className="flex items-center gap-4 text-sm text-slate-400">
              <span className="flex items-center gap-1">
                <User className="w-3 h-3" />
                {approval.requested_by}
              </span>
              <span className="flex items-center gap-1">
                <Clock className="w-3 h-3" />
                {formatTimeAgo(approval.requested_at_ms)}
              </span>
            </div>
          </div>
          <div className="px-2 py-1 bg-yellow-500/20 text-yellow-400 text-xs rounded">
            Pending
          </div>
        </div>

        {showComment && (
          <div className="mb-3">
            <textarea
              value={comment}
              onChange={(e) => setComment(e.target.value)}
              placeholder="Add a comment (optional)..."
              className="w-full bg-slate-900 border border-slate-700 rounded px-3 py-2 text-sm text-white placeholder-slate-500 resize-none"
              rows={2}
            />
          </div>
        )}
      </div>

      <div className="flex border-t border-slate-700">
        <button
          onClick={() => setShowComment(!showComment)}
          className="flex-1 py-2.5 text-sm text-slate-400 hover:bg-slate-700/50 transition-colors"
        >
          {showComment ? 'Hide Comment' : 'Add Comment'}
        </button>
        <div className="w-px bg-slate-700" />
        <button
          onClick={handleReject}
          className="flex-1 flex items-center justify-center gap-2 py-2.5 text-sm text-red-400 hover:bg-red-500/10 transition-colors"
        >
          <X className="w-4 h-4" />
          Reject
        </button>
        <div className="w-px bg-slate-700" />
        <button
          onClick={handleApprove}
          className="flex-1 flex items-center justify-center gap-2 py-2.5 text-sm text-green-400 hover:bg-green-500/10 transition-colors"
        >
          <Check className="w-4 h-4" />
          Approve
        </button>
      </div>
    </div>
  );
}

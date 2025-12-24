import { useState } from 'react';
import {
  Home,
  BookOpen,
  Brain,
  Database,
  Server,
  Calendar,
  HardDrive,
  Network,
  Shield,
  Dna,
  DollarSign,
  TrendingUp,
  Settings,
  ChevronLeft,
  ChevronRight,
  Zap,
  Cpu,
  MoreHorizontal,
} from 'lucide-react';
import { useClusterStore } from '../../stores/clusterStore';
import { useUserRole, ROLE_DISPLAY_NAMES } from '../../contexts/UserRoleContext';
import { SidebarSection } from './SidebarSection';
import type { SidebarSectionConfig, UserRole } from './types';
import { ROLE_VISIBILITY } from './types';

// Define all sidebar sections with their items
const SIDEBAR_SECTIONS: SidebarSectionConfig[] = [
  {
    id: 'dashboard',
    label: 'Dashboard',
    icon: Home,
    items: [
      { id: 'overview', label: 'Overview', path: '/dashboard', icon: Home },
    ],
    visibility: ROLE_VISIBILITY.dashboard,
  },
  {
    id: 'notebooks',
    label: 'Notebooks',
    icon: BookOpen,
    items: [
      { id: 'notebooks', label: 'Rust Notebooks', path: '/notebook', icon: BookOpen },
      { id: 'gpu-compiler', label: 'GPU Compiler', path: '/notebook/compiler', icon: Cpu },
    ],
    visibility: ROLE_VISIBILITY.notebooks,
  },
  {
    id: 'training',
    label: 'Training',
    icon: Brain,
    items: [
      { id: 'jobs', label: 'Jobs', path: '/training', icon: Brain },
      { id: 'distributed', label: 'Distributed Config', path: '/training/distributed', icon: Network },
      { id: 'models', label: 'Model Registry', path: '/training/models', icon: Database },
    ],
    visibility: ROLE_VISIBILITY.training,
  },
  {
    id: 'dataProcessing',
    label: 'Data Processing',
    icon: Database,
    items: [
      { id: 'dataframes', label: 'DataFrames', path: '/data', icon: Database },
      { id: 'sql', label: 'SQL Editor', path: '/data/sql', icon: Database },
      { id: 'streaming', label: 'Streaming', path: '/data/streaming', icon: Zap },
    ],
    visibility: ROLE_VISIBILITY.dataProcessing,
  },
  {
    id: 'cluster',
    label: 'Cluster',
    icon: Server,
    items: [
      { id: 'nodes', label: 'Nodes', path: '/cluster', icon: Server },
      { id: 'swarm-editor', label: '.swarm Editor', path: '/cluster/editor', icon: BookOpen },
      { id: 'topology', label: 'GPU Topology', path: '/cluster/topology', icon: Network },
    ],
    visibility: ROLE_VISIBILITY.cluster,
  },
  {
    id: 'scheduler',
    label: 'Scheduler',
    icon: Calendar,
    items: [
      { id: 'job-queue', label: 'Job Queue', path: '/scheduler', icon: Calendar },
      { id: 'tenants', label: 'Tenants & Quotas', path: '/scheduler/tenants', icon: Settings },
      { id: 'backfill', label: 'Backfill Status', path: '/scheduler/backfill', icon: TrendingUp },
      { id: 'reservations', label: 'Reservations', path: '/scheduler/reservations', icon: Calendar },
      { id: 'qos', label: 'QoS Policies', path: '/scheduler/qos', icon: Shield },
      { id: 'federation', label: 'Federation', path: '/scheduler/federation', icon: Network },
    ],
    visibility: ROLE_VISIBILITY.scheduler,
  },
  {
    id: 'storage',
    label: 'Storage',
    icon: HardDrive,
    items: [
      { id: 'files', label: 'File Browser', path: '/storage', icon: HardDrive },
      { id: 'transfers', label: 'Transfers', path: '/storage/transfers', icon: Zap },
      { id: 'archives', label: 'Archives (.warp)', path: '/storage/archives', icon: Database },
      { id: 'burst-buffer', label: 'Burst Buffer', path: '/storage/burst', icon: Zap },
    ],
    visibility: ROLE_VISIBILITY.storage,
  },
  {
    id: 'network',
    label: 'Network',
    icon: Network,
    items: [
      { id: 'rdma', label: 'RDMA Topology', path: '/network', icon: Network },
      { id: 'mesh', label: 'Mesh Relay', path: '/network/mesh', icon: Server },
      { id: 'zk', label: 'ZK Proofs', path: '/network/zk', icon: Shield },
      { id: 'nat', label: 'NAT Traversal', path: '/network/nat', icon: Shield },
    ],
    visibility: ROLE_VISIBILITY.network,
  },
  {
    id: 'edgeProxy',
    label: 'Edge Proxy',
    icon: Shield,
    items: [
      { id: 'vortex', label: 'Vortex Status', path: '/edge', icon: Shield },
      { id: 'routing', label: 'Routing Rules', path: '/edge/routing', icon: Network },
      { id: 'brain', label: 'SLAI Brain', path: '/edge/brain', icon: Brain },
      { id: 'ddos', label: 'DDoS Protection', path: '/edge/ddos', icon: Shield },
    ],
    visibility: ROLE_VISIBILITY.edgeProxy,
  },
  {
    id: 'evolution',
    label: 'Evolution',
    icon: Dna,
    items: [
      { id: 'adas', label: 'ADAS Engine', path: '/evolution', icon: Dna },
      { id: 'dgm', label: 'DGM Engine', path: '/evolution/dgm', icon: Brain },
      { id: 'swarm', label: 'SwarmAgentic', path: '/evolution/swarm', icon: Zap },
      { id: 'explorer', label: 'Design Explorer', path: '/evolution/explorer', icon: TrendingUp },
    ],
    visibility: ROLE_VISIBILITY.evolution,
  },
  {
    id: 'costs',
    label: 'Costs',
    icon: DollarSign,
    items: [
      { id: 'attribution', label: 'Attribution', path: '/costs', icon: DollarSign },
      { id: 'forecast', label: 'Forecasting', path: '/costs/forecast', icon: TrendingUp },
      { id: 'chargeback', label: 'Chargeback', path: '/costs/chargeback', icon: DollarSign },
      { id: 'showback', label: 'Showback', path: '/costs/showback', icon: DollarSign },
      { id: 'alerts', label: 'Budget Alerts', path: '/costs/alerts', icon: Shield },
    ],
    visibility: ROLE_VISIBILITY.costs,
  },
  {
    id: 'intelligence',
    label: 'Intelligence',
    icon: TrendingUp,
    items: [
      { id: 'efficiency', label: 'Efficiency', path: '/intelligence', icon: TrendingUp },
      { id: 'margin', label: 'Margin Analysis', path: '/intelligence/margin', icon: DollarSign },
      { id: 'vendor', label: 'Vendor Comparison', path: '/intelligence/vendor', icon: Database },
      { id: 'initiatives', label: 'Initiatives', path: '/intelligence/initiatives', icon: Calendar },
    ],
    visibility: ROLE_VISIBILITY.intelligence,
  },
  {
    id: 'settings',
    label: 'Settings',
    icon: Settings,
    items: [
      { id: 'quotas', label: 'Quotas', path: '/settings', icon: Settings },
      { id: 'policies', label: 'Policies', path: '/settings/policies', icon: Shield },
      { id: 'api-keys', label: 'API Keys', path: '/settings/api-keys', icon: Shield },
      { id: 'preferences', label: 'Preferences', path: '/settings/preferences', icon: Settings },
    ],
    visibility: ROLE_VISIBILITY.settings,
  },
];

export function Sidebar() {
  const [collapsed, setCollapsed] = useState(false);
  const [showMore, setShowMore] = useState(false);
  const { connected, nodeCount, totalGpus } = useClusterStore();
  const { role, setRole, availableRoles } = useUserRole();

  // Filter sections by visibility
  const visibleSections = SIDEBAR_SECTIONS.filter(
    (section) => section.visibility[role] !== 'hidden'
  );
  const hiddenSections = SIDEBAR_SECTIONS.filter(
    (section) => section.visibility[role] === 'hidden'
  );

  return (
    <aside
      className={`${
        collapsed ? 'w-16' : 'w-64'
      } bg-slate-800 border-r border-slate-700 flex flex-col transition-all duration-200`}
    >
      {/* Header with Logo */}
      <div className="p-4 flex items-center justify-between border-b border-slate-700">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center flex-shrink-0">
            <Zap className="w-5 h-5 text-white" />
          </div>
          {!collapsed && (
            <span className="font-bold text-white text-lg">Horizon</span>
          )}
        </div>
        <button
          onClick={() => setCollapsed(!collapsed)}
          className="p-1 rounded text-slate-400 hover:text-white hover:bg-slate-700 transition-colors"
        >
          {collapsed ? (
            <ChevronRight className="w-4 h-4" />
          ) : (
            <ChevronLeft className="w-4 h-4" />
          )}
        </button>
      </div>

      {/* Role Selector */}
      {!collapsed && (
        <div className="px-4 py-2 border-b border-slate-700">
          <select
            value={role}
            onChange={(e) => setRole(e.target.value as UserRole)}
            className="w-full bg-slate-700 text-slate-200 text-sm rounded px-2 py-1.5 border border-slate-600 focus:outline-none focus:border-blue-500"
          >
            {availableRoles.map((r) => (
              <option key={r} value={r}>
                {ROLE_DISPLAY_NAMES[r]}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Navigation Sections */}
      <nav className="flex-1 overflow-y-auto py-2 space-y-1">
        {visibleSections.map((section) => (
          <SidebarSection
            key={section.id}
            section={section}
            visibility={section.visibility[role]}
            collapsed={collapsed}
          />
        ))}

        {/* More sections (hidden by default) */}
        {hiddenSections.length > 0 && !collapsed && (
          <div className="px-2 pt-2">
            <button
              onClick={() => setShowMore(!showMore)}
              className="w-full flex items-center gap-2 px-2 py-1.5 text-slate-500 hover:text-slate-300 transition-colors"
            >
              <MoreHorizontal className="w-4 h-4" />
              <span className="text-xs font-medium">
                {showMore ? 'Hide' : 'More'} ({hiddenSections.length})
              </span>
            </button>
            {showMore && (
              <div className="mt-1 space-y-1 opacity-60">
                {hiddenSections.map((section) => (
                  <SidebarSection
                    key={section.id}
                    section={section}
                    visibility="hidden"
                    collapsed={false}
                  />
                ))}
              </div>
            )}
          </div>
        )}
      </nav>

      {/* Footer - Connection Status */}
      <div className="p-4 border-t border-slate-700">
        <div
          className={`flex items-center gap-2 ${
            collapsed ? 'justify-center' : ''
          }`}
        >
          <div
            className={`w-2 h-2 rounded-full ${
              connected ? 'bg-green-400' : 'bg-slate-500'
            }`}
          />
          {!collapsed && (
            <span className="text-xs text-slate-400">
              {connected
                ? `${nodeCount} nodes, ${totalGpus} GPUs`
                : 'Disconnected'}
            </span>
          )}
        </div>
      </div>
    </aside>
  );
}

export default Sidebar;

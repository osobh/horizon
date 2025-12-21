import { Routes, Route, NavLink, Navigate } from 'react-router-dom';
import {
  BookOpen,
  Server,
  BarChart3,
  Settings,
  Zap,
  Cpu
} from 'lucide-react';
import { useClusterStore } from './stores/clusterStore';
import NotebookView from './views/NotebookView';
import ClusterView from './views/ClusterView';
import DashboardView from './views/DashboardView';

function App() {
  const { connected, nodeCount, totalGpus } = useClusterStore();

  return (
    <div className="flex h-screen bg-slate-900">
      {/* Sidebar */}
      <aside className="w-16 bg-slate-800 border-r border-slate-700 flex flex-col items-center py-4">
        {/* Logo */}
        <div className="mb-8">
          <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
            <Zap className="w-6 h-6 text-white" />
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 flex flex-col gap-2">
          <NavButton to="/notebook" icon={BookOpen} label="Notebook" />
          <NavButton to="/cluster" icon={Server} label="Cluster" />
          <NavButton to="/dashboard" icon={BarChart3} label="Dashboard" />
        </nav>

        {/* Status & Settings */}
        <div className="flex flex-col gap-2 items-center">
          {/* Connection Status */}
          <div
            className={`w-8 h-8 rounded-full flex items-center justify-center ${
              connected ? 'bg-green-500/20 text-green-400' : 'bg-slate-700 text-slate-500'
            }`}
            title={connected ? `${nodeCount} nodes, ${totalGpus} GPUs` : 'Disconnected'}
          >
            <Cpu className="w-4 h-4" />
          </div>

          <button
            className="w-10 h-10 rounded-lg hover:bg-slate-700 flex items-center justify-center text-slate-400 hover:text-slate-200 transition-colors"
            title="Settings"
          >
            <Settings className="w-5 h-5" />
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-hidden">
        <Routes>
          <Route path="/" element={<Navigate to="/notebook" replace />} />
          <Route path="/notebook" element={<NotebookView />} />
          <Route path="/cluster" element={<ClusterView />} />
          <Route path="/dashboard" element={<DashboardView />} />
        </Routes>
      </main>
    </div>
  );
}

interface NavButtonProps {
  to: string;
  icon: React.ComponentType<{ className?: string }>;
  label: string;
}

function NavButton({ to, icon: Icon, label }: NavButtonProps) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        `w-10 h-10 rounded-lg flex items-center justify-center transition-colors ${
          isActive
            ? 'bg-blue-600 text-white'
            : 'text-slate-400 hover:bg-slate-700 hover:text-slate-200'
        }`
      }
      title={label}
    >
      <Icon className="w-5 h-5" />
    </NavLink>
  );
}

export default App;

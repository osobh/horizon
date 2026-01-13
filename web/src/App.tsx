import { Routes, Route, Navigate } from 'react-router-dom';
import { Sidebar } from './components/Sidebar';

// Existing views
import NotebookView from './views/NotebookView';
import ClusterView from './views/ClusterView';
import DashboardView from './views/DashboardView';

// New stub views
import SchedulerView from './views/SchedulerView';
import ObservabilityView from './views/ObservabilityView';
import StorageView from './views/StorageView';
import DataProcessingView from './views/DataProcessingView';
import NetworkView from './views/NetworkView';
import EdgeProxyView from './views/EdgeProxyView';
import CostsView from './views/CostsView';
import IntelligenceView from './views/IntelligenceView';
import SettingsView from './views/SettingsView';
import TrainingView from './views/TrainingView';
import EvolutionView from './views/EvolutionView';
import PipelinesView from './views/PipelinesView';
import SwarmView from './views/SwarmView';

function App() {
  return (
    <div className="flex h-screen bg-slate-900">
      {/* Unified Sidebar */}
      <Sidebar />

      {/* Main Content */}
      <main className="flex-1 overflow-hidden">
        <Routes>
          {/* Default redirect */}
          <Route path="/" element={<Navigate to="/dashboard" replace />} />

          {/* Dashboard */}
          <Route path="/dashboard" element={<DashboardView />} />

          {/* Notebooks (ML Researchers) */}
          <Route path="/notebook" element={<NotebookView />} />
          <Route path="/notebook/compiler" element={<NotebookView />} />

          {/* Training (RustyTorch) */}
          <Route path="/training" element={<TrainingView />} />
          <Route path="/training/distributed" element={<TrainingView />} />
          <Route path="/training/models" element={<TrainingView />} />
          <Route path="/training/metrics" element={<TrainingView />} />

          {/* Data Processing (RustySpark) */}
          <Route path="/data" element={<DataProcessingView />} />
          <Route path="/data/sql" element={<DataProcessingView />} />
          <Route path="/data/streaming" element={<DataProcessingView />} />
          <Route path="/data/quality" element={<DataProcessingView />} />

          {/* Cluster (StratoSwarm) */}
          <Route path="/cluster" element={<ClusterView />} />
          <Route path="/cluster/editor" element={<ClusterView />} />
          <Route path="/cluster/topology" element={<ClusterView />} />

          {/* Scheduler (SLAI) */}
          <Route path="/scheduler" element={<SchedulerView />} />
          <Route path="/scheduler/tenants" element={<SchedulerView />} />
          <Route path="/scheduler/backfill" element={<SchedulerView />} />
          <Route path="/scheduler/reservations" element={<SchedulerView />} />
          <Route path="/scheduler/qos" element={<SchedulerView />} />
          <Route path="/scheduler/federation" element={<SchedulerView />} />

          {/* Storage (WARP) */}
          <Route path="/storage" element={<StorageView />} />
          <Route path="/storage/transfers" element={<StorageView />} />
          <Route path="/storage/archives" element={<StorageView />} />
          <Route path="/storage/burst" element={<StorageView />} />

          {/* Network (Nebula) */}
          <Route path="/network" element={<NetworkView />} />
          <Route path="/network/mesh" element={<NetworkView />} />
          <Route path="/network/zk" element={<NetworkView />} />
          <Route path="/network/nat" element={<NetworkView />} />
          <Route path="/network/ebpf" element={<NetworkView />} />

          {/* Edge Proxy (Vortex) */}
          <Route path="/edge" element={<EdgeProxyView />} />
          <Route path="/edge/routing" element={<EdgeProxyView />} />
          <Route path="/edge/brain" element={<EdgeProxyView />} />
          <Route path="/edge/transmutation" element={<EdgeProxyView />} />
          <Route path="/edge/ddos" element={<EdgeProxyView />} />

          {/* Observability (Argus) */}
          <Route path="/observability" element={<ObservabilityView />} />
          <Route path="/observability/metrics" element={<ObservabilityView />} />
          <Route path="/observability/alerts" element={<ObservabilityView />} />
          <Route path="/observability/targets" element={<ObservabilityView />} />

          {/* Evolution */}
          <Route path="/evolution" element={<EvolutionView />} />
          <Route path="/evolution/dgm" element={<EvolutionView />} />
          <Route path="/evolution/swarm" element={<EvolutionView />} />
          <Route path="/evolution/explorer" element={<EvolutionView />} />

          {/* Swarm (StratoSwarm Agents) */}
          <Route path="/swarm" element={<SwarmView />} />
          <Route path="/swarm/agents" element={<SwarmView />} />
          <Route path="/swarm/evolution" element={<SwarmView />} />

          {/* Pipelines (HPC-CI) */}
          <Route path="/pipelines" element={<PipelinesView />} />
          <Route path="/pipelines/agents" element={<PipelinesView />} />
          <Route path="/pipelines/approvals" element={<PipelinesView />} />

          {/* Costs */}
          <Route path="/costs" element={<CostsView />} />
          <Route path="/costs/forecast" element={<CostsView />} />
          <Route path="/costs/chargeback" element={<CostsView />} />
          <Route path="/costs/showback" element={<CostsView />} />
          <Route path="/costs/alerts" element={<CostsView />} />

          {/* Intelligence */}
          <Route path="/intelligence" element={<IntelligenceView />} />
          <Route path="/intelligence/margin" element={<IntelligenceView />} />
          <Route path="/intelligence/vendor" element={<IntelligenceView />} />
          <Route path="/intelligence/initiatives" element={<IntelligenceView />} />

          {/* Settings */}
          <Route path="/settings" element={<SettingsView />} />
          <Route path="/settings/policies" element={<SettingsView />} />
          <Route path="/settings/api-keys" element={<SettingsView />} />
          <Route path="/settings/preferences" element={<SettingsView />} />

          {/* Fallback */}
          <Route path="*" element={<Navigate to="/dashboard" replace />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;

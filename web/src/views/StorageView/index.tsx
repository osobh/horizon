import { HardDrive, Upload, Archive, Zap, FolderTree } from 'lucide-react';

export default function StorageView() {
  return (
    <div className="h-full flex flex-col bg-slate-900 text-white p-6">
      <header className="mb-6">
        <h1 className="text-2xl font-bold flex items-center gap-3">
          <HardDrive className="w-7 h-7 text-green-400" />
          Storage (WARP)
        </h1>
        <p className="text-slate-400 mt-1">
          GPU-accelerated data transfers, archives, and burst buffer management
        </p>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <StubCard
          icon={FolderTree}
          title="File Browser"
          description="Browse distributed filesystem with tree view"
          path="/storage"
        />
        <StubCard
          icon={Upload}
          title="Active Transfers"
          description="Monitor GPU-accelerated upload/download progress"
          path="/storage/transfers"
        />
        <StubCard
          icon={Archive}
          title="Archives (.warp)"
          description="Manage .warp compressed archives"
          path="/storage/archives"
        />
        <StubCard
          icon={Zap}
          title="Burst Buffer"
          description="NVMe staging area status"
          path="/storage/burst"
        />
      </div>

      <div className="mt-6 p-4 bg-slate-800 rounded-lg border border-slate-700">
        <h3 className="text-sm font-semibold text-slate-400 mb-2">Integration Status</h3>
        <p className="text-sm text-slate-500">
          This view will integrate with 04-warp services. Features will be migrated from 04-warp/crates/warp-dashboard/.
        </p>
      </div>
    </div>
  );
}

function StubCard({ icon: Icon, title, description, path }: {
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  description: string;
  path: string;
}) {
  return (
    <div className="p-4 bg-slate-800 rounded-lg border border-slate-700 hover:border-green-500/50 transition-colors">
      <div className="flex items-center gap-3 mb-2">
        <Icon className="w-5 h-5 text-green-400" />
        <h3 className="font-semibold">{title}</h3>
      </div>
      <p className="text-sm text-slate-400">{description}</p>
      <p className="text-xs text-slate-600 mt-2 font-mono">{path}</p>
    </div>
  );
}

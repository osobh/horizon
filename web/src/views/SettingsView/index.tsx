import { Settings, Shield, Key, UserCog, Sliders } from 'lucide-react';

export default function SettingsView() {
  return (
    <div className="h-full flex flex-col bg-slate-900 text-white p-6">
      <header className="mb-6">
        <h1 className="text-2xl font-bold flex items-center gap-3">
          <Settings className="w-7 h-7 text-slate-400" />
          Settings
        </h1>
        <p className="text-slate-400 mt-1">
          Quotas, policies, API keys, and user preferences
        </p>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <StubCard
          icon={Sliders}
          title="Quotas"
          description="Configure resource quotas (quota-manager)"
          path="/settings"
        />
        <StubCard
          icon={Shield}
          title="Policies"
          description="Manage governance policies (governor)"
          path="/settings/policies"
        />
        <StubCard
          icon={Key}
          title="API Keys"
          description="Manage API keys and tokens"
          path="/settings/api-keys"
        />
        <StubCard
          icon={UserCog}
          title="Preferences"
          description="User preferences and theme settings"
          path="/settings/preferences"
        />
      </div>

      <div className="mt-6 p-4 bg-slate-800 rounded-lg border border-slate-700">
        <h3 className="text-sm font-semibold text-slate-400 mb-2">Integration Status</h3>
        <p className="text-sm text-slate-500">
          This view will integrate with horizon-services: quota-manager and governor.
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
    <div className="p-4 bg-slate-800 rounded-lg border border-slate-700 hover:border-slate-500/50 transition-colors">
      <div className="flex items-center gap-3 mb-2">
        <Icon className="w-5 h-5 text-slate-400" />
        <h3 className="font-semibold">{title}</h3>
      </div>
      <p className="text-sm text-slate-400">{description}</p>
      <p className="text-xs text-slate-600 mt-2 font-mono">{path}</p>
    </div>
  );
}

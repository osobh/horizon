import { Database, Code, Radio, CheckCircle } from 'lucide-react';

export default function DataProcessingView() {
  return (
    <div className="h-full flex flex-col bg-slate-900 text-white p-6">
      <header className="mb-6">
        <h1 className="text-2xl font-bold flex items-center gap-3">
          <Database className="w-7 h-7 text-orange-400" />
          Data Processing (RustySpark)
        </h1>
        <p className="text-slate-400 mt-1">
          Distributed DataFrames, SQL queries, and streaming pipelines
        </p>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <StubCard
          icon={Database}
          title="DataFrames Browser"
          description="Browse and inspect distributed DataFrames"
          path="/data"
        />
        <StubCard
          icon={Code}
          title="SQL Editor"
          description="Monaco SQL editor with auto-completion"
          path="/data/sql"
        />
        <StubCard
          icon={Radio}
          title="Streaming"
          description="Kafka stream status and metrics"
          path="/data/streaming"
        />
        <StubCard
          icon={CheckCircle}
          title="Data Quality"
          description="Data quality metrics and validation"
          path="/data/quality"
        />
      </div>

      <div className="mt-6 p-4 bg-slate-800 rounded-lg border border-slate-700">
        <h3 className="text-sm font-semibold text-slate-400 mb-2">Integration Status</h3>
        <p className="text-sm text-slate-500">
          This view will integrate with 06-rustyspark services for distributed data processing.
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
    <div className="p-4 bg-slate-800 rounded-lg border border-slate-700 hover:border-orange-500/50 transition-colors">
      <div className="flex items-center gap-3 mb-2">
        <Icon className="w-5 h-5 text-orange-400" />
        <h3 className="font-semibold">{title}</h3>
      </div>
      <p className="text-sm text-slate-400">{description}</p>
      <p className="text-xs text-slate-600 mt-2 font-mono">{path}</p>
    </div>
  );
}

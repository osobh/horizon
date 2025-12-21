/**
 * SwarmConfigEditor - Monaco-based editor for .swarm DSL files
 *
 * Provides syntax highlighting, auto-completion, and validation
 * for StratoSwarm configuration files.
 */

import { useState } from 'react';
import Editor from '@monaco-editor/react';
import { Save, FileCode, Play, RotateCcw } from 'lucide-react';
import { SWARM_LANGUAGE_ID } from '../languages/swarm';

// Example .swarm configuration
const DEFAULT_SWARM_CONFIG = `// Example StratoSwarm Configuration
swarm ml_training_cluster {
  agents {
    frontend: WebAgent {
      replicas: 3,

      resources {
        cpu: 2.0,
        memory: "4Gi",
      }

      network {
        expose: 8080,
        mesh: true,
        load_balance: "round_robin",
      }
    }

    compute: ComputeAgent {
      replicas: 8,

      resources {
        cpu: 8.0,
        memory: "32Gi",
        gpu: 1.0,
      }

      personality {
        cooperation: 0.8,
        exploration: 0.6,
        efficiency_focus: 0.9,
      }

      affinity {
        requires_gpu: true,
        tier_preference: [GPU, CPU],
      }
    }

    storage: StorageAgent {
      replicas: 3,

      resources {
        cpu: 4.0,
        memory: "16Gi",
      }

      storage {
        size: "100Gi",
        type: "nvme",
        replication: 3,
      }
    }
  }

  connections {
    frontend -> compute,
    compute -> storage,
  }

  policies {
    zero_downtime_updates: true,
    canary_rollout: true,
    rollback_on: "error_rate > 0.05",
  }

  evolution {
    enabled: true,
    population_size: 50,
    mutation_rate: 0.1,
    crossover_rate: 0.7,
    selection_pressure: 0.8,
  }
}
`;

interface SwarmConfigEditorProps {
  initialConfig?: string;
  onSave?: (config: string) => void;
  onDeploy?: (config: string) => void;
}

export default function SwarmConfigEditor({
  initialConfig = DEFAULT_SWARM_CONFIG,
  onSave,
  onDeploy,
}: SwarmConfigEditorProps) {
  const [config, setConfig] = useState(initialConfig);
  const [modified, setModified] = useState(false);
  const [validationErrors, setValidationErrors] = useState<string[]>([]);

  const handleEditorChange = (value: string | undefined) => {
    if (value !== undefined) {
      setConfig(value);
      setModified(value !== initialConfig);
      // Basic validation (can be extended with actual parser)
      validateConfig(value);
    }
  };

  const validateConfig = (content: string) => {
    const errors: string[] = [];

    // Check for balanced braces
    const openBraces = (content.match(/{/g) || []).length;
    const closeBraces = (content.match(/}/g) || []).length;
    if (openBraces !== closeBraces) {
      errors.push(`Unbalanced braces: ${openBraces} open, ${closeBraces} close`);
    }

    // Check for required blocks in swarm definition
    if (content.includes('swarm ')) {
      if (!content.includes('agents {')) {
        errors.push('Missing required "agents" block');
      }
    }

    setValidationErrors(errors);
  };

  const handleSave = () => {
    if (onSave) {
      onSave(config);
      setModified(false);
    }
  };

  const handleDeploy = () => {
    if (validationErrors.length > 0) {
      return; // Don't deploy with errors
    }
    if (onDeploy) {
      onDeploy(config);
    }
  };

  const handleReset = () => {
    setConfig(initialConfig);
    setModified(false);
    validateConfig(initialConfig);
  };

  return (
    <div className="flex flex-col h-full">
      {/* Toolbar */}
      <div className="h-12 bg-slate-800 border-b border-slate-700 flex items-center px-4 gap-2">
        <FileCode className="w-4 h-4 text-slate-400" />
        <span className="text-sm font-medium">
          swarm.config
          {modified && <span className="text-yellow-400 ml-1">*</span>}
        </span>

        <div className="flex-1" />

        {/* Validation Status */}
        {validationErrors.length > 0 && (
          <div className="flex items-center gap-2 text-red-400 text-sm">
            <span className="w-2 h-2 rounded-full bg-red-500" />
            {validationErrors.length} error{validationErrors.length !== 1 ? 's' : ''}
          </div>
        )}

        <button
          onClick={handleReset}
          disabled={!modified}
          className="flex items-center gap-1 px-3 py-1.5 bg-slate-700 hover:bg-slate-600 disabled:opacity-50 rounded text-sm"
        >
          <RotateCcw className="w-4 h-4" />
          Reset
        </button>

        <button
          onClick={handleSave}
          disabled={!modified}
          className="flex items-center gap-1 px-3 py-1.5 bg-slate-700 hover:bg-slate-600 disabled:opacity-50 rounded text-sm"
        >
          <Save className="w-4 h-4" />
          Save
        </button>

        <button
          onClick={handleDeploy}
          disabled={validationErrors.length > 0}
          className="flex items-center gap-1 px-3 py-1.5 bg-green-600 hover:bg-green-700 disabled:opacity-50 rounded text-sm font-medium"
        >
          <Play className="w-4 h-4" />
          Deploy
        </button>
      </div>

      {/* Validation Errors */}
      {validationErrors.length > 0 && (
        <div className="bg-red-900/20 border-b border-red-700/50 px-4 py-2">
          {validationErrors.map((error, i) => (
            <div key={i} className="text-sm text-red-400">
              {error}
            </div>
          ))}
        </div>
      )}

      {/* Editor */}
      <div className="flex-1">
        <Editor
          height="100%"
          language={SWARM_LANGUAGE_ID}
          theme="horizon-dark"
          value={config}
          onChange={handleEditorChange}
          options={{
            minimap: { enabled: true },
            lineNumbers: 'on',
            fontSize: 14,
            fontFamily: 'JetBrains Mono, monospace',
            scrollBeyondLastLine: false,
            wordWrap: 'off',
            tabSize: 2,
            insertSpaces: true,
            automaticLayout: true,
            folding: true,
            foldingHighlight: true,
            renderLineHighlight: 'all',
            bracketPairColorization: { enabled: true },
            formatOnPaste: true,
            formatOnType: true,
          }}
        />
      </div>
    </div>
  );
}

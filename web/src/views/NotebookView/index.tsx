import { useEffect } from 'react';
import Editor from '@monaco-editor/react';
import {
  Play,
  Plus,
  RotateCcw,
  Variable,
  ChevronRight,
} from 'lucide-react';
import { useNotebookStore } from '../../stores/notebookStore';

export default function NotebookView() {
  const {
    cells,
    variables,
    selectedCellId,
    executing,
    addCell,
    updateCell,
    deleteCell,
    selectCell,
    executeCell,
    executeAll,
    restartKernel,
    fetchVariables,
  } = useNotebookStore();

  useEffect(() => {
    fetchVariables();
  }, [fetchVariables]);

  return (
    <div className="flex h-full">
      {/* Main Editor Area */}
      <div className="flex-1 flex flex-col">
        {/* Toolbar */}
        <div className="h-12 bg-slate-800 border-b border-slate-700 flex items-center px-4 gap-2">
          <button
            onClick={() => executeAll()}
            disabled={executing}
            className="flex items-center gap-1 px-3 py-1.5 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 rounded text-sm font-medium"
          >
            <Play className="w-4 h-4" />
            Run All
          </button>

          <button
            onClick={() => restartKernel()}
            className="flex items-center gap-1 px-3 py-1.5 bg-slate-700 hover:bg-slate-600 rounded text-sm"
          >
            <RotateCcw className="w-4 h-4" />
            Restart
          </button>

          <div className="flex-1" />

          <button
            onClick={() => addCell('code')}
            className="flex items-center gap-1 px-3 py-1.5 bg-slate-700 hover:bg-slate-600 rounded text-sm"
          >
            <Plus className="w-4 h-4" />
            Add Cell
          </button>
        </div>

        {/* Cells */}
        <div className="flex-1 overflow-auto p-4 space-y-4">
          {cells.map((cell) => (
            <div
              key={cell.id}
              className={`border rounded-lg overflow-hidden transition-colors ${
                selectedCellId === cell.id
                  ? 'border-blue-500'
                  : 'border-slate-700'
              }`}
              onClick={() => selectCell(cell.id)}
            >
              {/* Cell Header */}
              <div className="flex items-center gap-2 px-3 py-2 bg-slate-800 border-b border-slate-700">
                <span className="text-xs font-mono text-slate-400">
                  [{cell.executionCount ?? ' '}]
                </span>

                <span className="text-xs uppercase text-slate-500">
                  {cell.type}
                </span>

                <div className="flex-1" />

                {cell.type === 'code' && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      executeCell(cell.id);
                    }}
                    disabled={cell.executing}
                    className="p-1 hover:bg-slate-700 rounded disabled:opacity-50"
                  >
                    <Play className="w-4 h-4" />
                  </button>
                )}

                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteCell(cell.id);
                  }}
                  className="p-1 hover:bg-red-900/50 hover:text-red-400 rounded text-slate-400"
                >
                  ×
                </button>
              </div>

              {/* Cell Content */}
              <div className="bg-slate-900">
                <Editor
                  height={Math.max(80, cell.content.split('\n').length * 20)}
                  defaultLanguage="rust"
                  theme="vs-dark"
                  value={cell.content}
                  onChange={(value) => updateCell(cell.id, value ?? '')}
                  options={{
                    minimap: { enabled: false },
                    lineNumbers: 'on',
                    fontSize: 14,
                    fontFamily: 'JetBrains Mono, monospace',
                    scrollBeyondLastLine: false,
                    wordWrap: 'on',
                    padding: { top: 8, bottom: 8 },
                  }}
                />
              </div>

              {/* Cell Output */}
              {cell.outputs.length > 0 && (
                <div className="bg-slate-950 border-t border-slate-700 p-3">
                  {cell.outputs.map((output, i) => (
                    <pre
                      key={i}
                      className={`font-mono text-sm ${
                        output.output_type === 'error'
                          ? 'text-red-400'
                          : output.output_type === 'stderr'
                          ? 'text-yellow-400'
                          : 'text-green-400'
                      }`}
                    >
                      {output.content}
                    </pre>
                  ))}
                </div>
              )}

              {/* Executing indicator */}
              {cell.executing && (
                <div className="bg-blue-900/30 border-t border-blue-700/50 p-2 flex items-center gap-2">
                  <div className="w-4 h-4 border-2 border-blue-400 border-t-transparent rounded-full animate-spin" />
                  <span className="text-sm text-blue-400">Executing...</span>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Variables Panel */}
      <aside className="w-64 bg-slate-800 border-l border-slate-700 flex flex-col">
        <div className="h-12 flex items-center px-4 border-b border-slate-700">
          <Variable className="w-4 h-4 mr-2 text-slate-400" />
          <span className="font-medium">Variables</span>
        </div>

        <div className="flex-1 overflow-auto p-2">
          {variables.length === 0 ? (
            <p className="text-sm text-slate-500 p-2">
              No variables in scope
            </p>
          ) : (
            <div className="space-y-1">
              {variables.map((v) => (
                <VariableRow key={v.name} variable={v} />
              ))}
            </div>
          )}
        </div>
      </aside>
    </div>
  );
}

interface VariableRowProps {
  variable: {
    name: string;
    var_type: string;
    size_bytes: number;
    preview: string;
  };
}

function VariableRow({ variable }: VariableRowProps) {
  return (
    <div className="p-2 rounded hover:bg-slate-700/50 cursor-pointer">
      <div className="flex items-center gap-2">
        <ChevronRight className="w-3 h-3 text-slate-500" />
        <span className="font-mono text-sm text-blue-400">{variable.name}</span>
        <span className="text-xs text-slate-500">{variable.var_type}</span>
      </div>
      <div className="ml-5 text-xs font-mono text-slate-400 truncate">
        {variable.preview}
      </div>
    </div>
  );
}

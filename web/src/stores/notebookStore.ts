import { create } from 'zustand';
import { invoke } from '@tauri-apps/api/core';

interface CellOutput {
  output_type: 'stdout' | 'stderr' | 'result' | 'error' | 'display';
  content: string;
  execution_count: number;
}

interface Cell {
  id: string;
  type: 'code' | 'markdown';
  content: string;
  outputs: CellOutput[];
  executionCount: number | null;
  executing: boolean;
}

interface VariableInfo {
  name: string;
  var_type: string;
  size_bytes: number;
  preview: string;
}

interface ExecutionResult {
  success: boolean;
  outputs: CellOutput[];
  execution_count: number;
  duration_ms: number;
}

interface NotebookState {
  cells: Cell[];
  variables: VariableInfo[];
  selectedCellId: string | null;
  kernelRunning: boolean;
  executing: boolean;
  error: string | null;

  // Actions
  addCell: (type: 'code' | 'markdown', after?: string) => void;
  updateCell: (id: string, content: string) => void;
  deleteCell: (id: string) => void;
  selectCell: (id: string | null) => void;
  executeCell: (id: string) => Promise<void>;
  executeAll: () => Promise<void>;
  restartKernel: () => Promise<void>;
  fetchVariables: () => Promise<void>;
}

let cellIdCounter = 0;

function generateCellId(): string {
  return `cell-${++cellIdCounter}-${Date.now()}`;
}

export const useNotebookStore = create<NotebookState>((set, get) => ({
  cells: [
    {
      id: generateCellId(),
      type: 'code',
      content: '// Welcome to Horizon!\n// Start coding in Rust with GPU acceleration\n\nlet x = 42;\nprintln!("Hello from Horizon! x = {}", x);',
      outputs: [],
      executionCount: null,
      executing: false,
    },
  ],
  variables: [],
  selectedCellId: null,
  kernelRunning: true,
  executing: false,
  error: null,

  addCell: (type, after) => {
    const newCell: Cell = {
      id: generateCellId(),
      type,
      content: type === 'code' ? '' : '# Markdown Cell',
      outputs: [],
      executionCount: null,
      executing: false,
    };

    set((state) => {
      if (after) {
        const index = state.cells.findIndex((c) => c.id === after);
        if (index !== -1) {
          const newCells = [...state.cells];
          newCells.splice(index + 1, 0, newCell);
          return { cells: newCells, selectedCellId: newCell.id };
        }
      }
      return { cells: [...state.cells, newCell], selectedCellId: newCell.id };
    });
  },

  updateCell: (id, content) => {
    set((state) => ({
      cells: state.cells.map((c) =>
        c.id === id ? { ...c, content } : c
      ),
    }));
  },

  deleteCell: (id) => {
    set((state) => {
      if (state.cells.length <= 1) return state;
      return {
        cells: state.cells.filter((c) => c.id !== id),
        selectedCellId:
          state.selectedCellId === id ? null : state.selectedCellId,
      };
    });
  },

  selectCell: (id) => {
    set({ selectedCellId: id });
  },

  executeCell: async (id) => {
    const { cells } = get();
    const cell = cells.find((c) => c.id === id);
    if (!cell || cell.type !== 'code') return;

    set((state) => ({
      executing: true,
      cells: state.cells.map((c) =>
        c.id === id ? { ...c, executing: true, outputs: [] } : c
      ),
    }));

    try {
      const result = await invoke<ExecutionResult>('execute_cell', {
        code: cell.content,
      });

      set((state) => ({
        executing: false,
        cells: state.cells.map((c) =>
          c.id === id
            ? {
                ...c,
                executing: false,
                outputs: result.outputs,
                executionCount: result.execution_count,
              }
            : c
        ),
      }));

      // Refresh variables after execution
      await get().fetchVariables();
    } catch (error) {
      set((state) => ({
        executing: false,
        error: String(error),
        cells: state.cells.map((c) =>
          c.id === id
            ? {
                ...c,
                executing: false,
                outputs: [
                  {
                    output_type: 'error',
                    content: String(error),
                    execution_count: 0,
                  },
                ],
              }
            : c
        ),
      }));
    }
  },

  executeAll: async () => {
    const { cells, executeCell } = get();
    for (const cell of cells) {
      if (cell.type === 'code') {
        await executeCell(cell.id);
      }
    }
  },

  restartKernel: async () => {
    try {
      await invoke('restart_kernel');
      set({
        variables: [],
        cells: get().cells.map((c) => ({
          ...c,
          outputs: [],
          executionCount: null,
        })),
      });
    } catch (error) {
      set({ error: String(error) });
    }
  },

  fetchVariables: async () => {
    try {
      const variables = await invoke<VariableInfo[]>('get_variables');
      set({ variables });
    } catch (error) {
      set({ error: String(error) });
    }
  },
}));

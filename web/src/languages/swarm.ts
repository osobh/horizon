/**
 * .swarm DSL Language Configuration for Monaco Editor
 *
 * This module defines syntax highlighting, auto-completion, and other
 * language features for StratoSwarm's .swarm configuration files.
 */

import * as monaco from 'monaco-editor';

// Language ID
export const SWARM_LANGUAGE_ID = 'swarm';

/**
 * Language configuration for bracket matching, comments, etc.
 */
export const swarmLanguageConfig: monaco.languages.LanguageConfiguration = {
  comments: {
    lineComment: '//',
    blockComment: ['/*', '*/'],
  },
  brackets: [
    ['{', '}'],
    ['[', ']'],
    ['(', ')'],
  ],
  autoClosingPairs: [
    { open: '{', close: '}' },
    { open: '[', close: ']' },
    { open: '(', close: ')' },
    { open: '"', close: '"', notIn: ['string'] },
  ],
  surroundingPairs: [
    { open: '{', close: '}' },
    { open: '[', close: ']' },
    { open: '(', close: ')' },
    { open: '"', close: '"' },
  ],
  folding: {
    markers: {
      start: /^\s*\/\/#region/,
      end: /^\s*\/\/#endregion/,
    },
  },
};

/**
 * Monarch tokenizer for syntax highlighting
 */
export const swarmMonarchLanguage: monaco.languages.IMonarchLanguage = {
  defaultToken: 'invalid',
  tokenPostfix: '.swarm',

  // Keywords
  keywords: [
    'import', 'template', 'swarm', 'as', 'fn',
    'true', 'false',
  ],

  // Top-level blocks
  blocks: [
    'agents', 'connections', 'policies', 'functions',
    'evolution', 'affinity', 'resources', 'network',
    'personality', 'code', 'storage', 'tolerations',
    'node_selector',
  ],

  // Agent types
  agentTypes: [
    'WebAgent', 'ComputeAgent', 'StorageAgent',
    'NetworkAgent', 'GPUAgent',
  ],

  // Tier types (used in tier_preference)
  tierTypes: ['GPU', 'CPU', 'NVMe', 'Memory'],

  // Built-in functions
  builtinFunctions: [
    'optional', 'exponential_backoff', 'linear_backoff',
    'http_get', 'max', 'min',
  ],

  // Property names
  properties: [
    'replicas', 'memory', 'cpu', 'gpu', 'tier_preference',
    'expose', 'mesh', 'load_balance', 'protocol', 'retry',
    'circuit_breaker', 'port', 'pool_size', 'risk_tolerance',
    'cooperation', 'exploration', 'efficiency_focus',
    'stability_preference', 'strategy', 'fitness',
    'prefer_same_node', 'avoid_nodes_with', 'requires_gpu',
    'source', 'auto_evolve', 'size', 'type', 'replication',
    'zero_downtime_updates', 'canary_rollout', 'rollback_on',
    'max_surge', 'max_unavailable', 'enabled', 'population_size',
    'mutation_rate', 'crossover_rate', 'selection_pressure',
    'tier', 'region', 'key', 'operator', 'value', 'effect',
  ],

  // Type keywords
  typeKeywords: [
    'String', 'Int', 'Float', 'Bool', 'Agent', 'Metrics',
  ],

  // Operators
  operators: [
    '->', '..', ':', '=', '+', '-', '*', '/', '%',
    '&&', '||', '!', '<', '>', '<=', '>=', '==', '!=',
  ],

  // Symbol patterns
  symbols: /[=><!~?:&|+\-*\/\^%]+/,
  escapes: /\\(?:[abfnrtv\\"']|x[0-9A-Fa-f]{1,4}|u[0-9A-Fa-f]{4}|U[0-9A-Fa-f]{8})/,

  // Tokenizer rules
  tokenizer: {
    root: [
      // Whitespace
      [/[ \t\r\n]+/, 'white'],

      // Comments
      [/\/\/.*$/, 'comment'],
      [/\/\*/, 'comment', '@comment'],

      // String interpolation
      [/\$\{/, 'delimiter.bracket', '@interpolation'],

      // Strings
      [/"([^"\\]|\\.)*$/, 'string.invalid'], // non-terminated
      [/"/, 'string', '@string'],

      // Numbers
      [/\d+\.\d+/, 'number.float'],
      [/\d+/, 'number'],

      // Range operator
      [/\.\./, 'operator.range'],

      // Arrow operator (connections)
      [/->/, 'operator.arrow'],

      // Delimiters and operators
      [/[{}()\[\]]/, '@brackets'],
      [/[,;]/, 'delimiter'],
      [/:/, 'delimiter.colon'],

      // Identifiers and keywords
      [/[a-zA-Z_]\w*/, {
        cases: {
          '@keywords': 'keyword',
          '@blocks': 'keyword.block',
          '@agentTypes': 'type.agent',
          '@tierTypes': 'constant.tier',
          '@builtinFunctions': 'function.builtin',
          '@typeKeywords': 'type',
          '@properties': 'variable.property',
          '@default': 'identifier',
        },
      }],

      // Operators
      [/@symbols/, {
        cases: {
          '@operators': 'operator',
          '@default': '',
        },
      }],
    ],

    // Block comment
    comment: [
      [/[^\/*]+/, 'comment'],
      [/\/\*/, 'comment', '@push'],
      [/\*\//, 'comment', '@pop'],
      [/[\/*]/, 'comment'],
    ],

    // String
    string: [
      [/[^\\"$]+/, 'string'],
      [/\$\{/, 'delimiter.bracket', '@interpolation'],
      [/@escapes/, 'string.escape'],
      [/\\./, 'string.escape.invalid'],
      [/"/, 'string', '@pop'],
    ],

    // Interpolation
    interpolation: [
      [/[^}]+/, 'variable.interpolation'],
      [/\}/, 'delimiter.bracket', '@pop'],
    ],
  },
};

/**
 * Completion item provider for auto-completion
 */
export function createSwarmCompletionProvider(): monaco.languages.CompletionItemProvider {
  return {
    provideCompletionItems: (model, position) => {
      const word = model.getWordUntilPosition(position);
      const range: monaco.IRange = {
        startLineNumber: position.lineNumber,
        endLineNumber: position.lineNumber,
        startColumn: word.startColumn,
        endColumn: word.endColumn,
      };

      const suggestions: monaco.languages.CompletionItem[] = [
        // Keywords
        ...['import', 'template', 'swarm', 'fn', 'as'].map(kw => ({
          label: kw,
          kind: monaco.languages.CompletionItemKind.Keyword,
          insertText: kw,
          range,
        })),

        // Blocks
        ...['agents', 'connections', 'policies', 'functions', 'evolution', 'affinity'].map(block => ({
          label: block,
          kind: monaco.languages.CompletionItemKind.Module,
          insertText: `${block} {\n\t$0\n}`,
          insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
          range,
        })),

        // Agent types with snippets
        {
          label: 'WebAgent',
          kind: monaco.languages.CompletionItemKind.Class,
          insertText: [
            'WebAgent {',
            '\treplicas: ${1:3},',
            '\t',
            '\tresources {',
            '\t\tcpu: ${2:2.0},',
            '\t\tmemory: "${3:4Gi}",',
            '\t}',
            '\t',
            '\tnetwork {',
            '\t\texpose: ${4:8080},',
            '\t\tmesh: true,',
            '\t}',
            '}',
          ].join('\n'),
          insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
          documentation: 'Web serving agent with network configuration',
          range,
        },

        {
          label: 'ComputeAgent',
          kind: monaco.languages.CompletionItemKind.Class,
          insertText: [
            'ComputeAgent {',
            '\treplicas: ${1:5},',
            '\trequires_gpu: ${2:true},',
            '\t',
            '\tresources {',
            '\t\tcpu: ${3:4.0},',
            '\t\tmemory: "${4:16Gi}",',
            '\t\tgpu: ${5:1.0},',
            '\t}',
            '}',
          ].join('\n'),
          insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
          documentation: 'Compute-intensive agent with GPU support',
          range,
        },

        {
          label: 'StorageAgent',
          kind: monaco.languages.CompletionItemKind.Class,
          insertText: [
            'StorageAgent {',
            '\treplicas: ${1:3},',
            '\t',
            '\tresources {',
            '\t\tcpu: ${2:4.0},',
            '\t\tmemory: "${3:16Gi}",',
            '\t}',
            '\t',
            '\tstorage {',
            '\t\tsize: "${4:100Gi}",',
            '\t\ttype: "${5:nvme}",',
            '\t\treplication: ${6:3},',
            '\t}',
            '}',
          ].join('\n'),
          insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
          documentation: 'Storage agent with replication settings',
          range,
        },

        // Built-in functions
        {
          label: 'optional',
          kind: monaco.languages.CompletionItemKind.Function,
          insertText: 'optional(${1:value})',
          insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
          documentation: 'Marks a resource as optional',
          range,
        },
        {
          label: 'exponential_backoff',
          kind: monaco.languages.CompletionItemKind.Function,
          insertText: 'exponential_backoff(${1:1}, ${2:30})',
          insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
          documentation: 'Exponential backoff retry strategy (start, max)',
          range,
        },

        // Swarm template
        {
          label: 'swarm (full)',
          kind: monaco.languages.CompletionItemKind.Snippet,
          insertText: [
            'swarm ${1:myapp} {',
            '\tagents {',
            '\t\t${2:agent_name}: WebAgent {',
            '\t\t\treplicas: 3,',
            '\t\t\t',
            '\t\t\tresources {',
            '\t\t\t\tcpu: 2.0,',
            '\t\t\t\tmemory: "4Gi",',
            '\t\t\t}',
            '\t\t\t',
            '\t\t\tnetwork {',
            '\t\t\t\texpose: 8080,',
            '\t\t\t\tmesh: true,',
            '\t\t\t}',
            '\t\t}',
            '\t}',
            '\t',
            '\tconnections {',
            '\t\t$0',
            '\t}',
            '\t',
            '\tpolicies {',
            '\t\tzero_downtime_updates: true,',
            '\t}',
            '}',
          ].join('\n'),
          insertTextRules: monaco.languages.CompletionItemInsertTextRule.InsertAsSnippet,
          documentation: 'Complete swarm definition with agents, connections, and policies',
          range,
        },
      ];

      return { suggestions };
    },
  };
}

/**
 * Register the .swarm language with Monaco Editor
 */
export function registerSwarmLanguage() {
  // Register the language
  monaco.languages.register({
    id: SWARM_LANGUAGE_ID,
    extensions: ['.swarm'],
    aliases: ['Swarm', 'swarm', 'StratoSwarm'],
    mimetypes: ['text/x-swarm'],
  });

  // Set language configuration
  monaco.languages.setLanguageConfiguration(SWARM_LANGUAGE_ID, swarmLanguageConfig);

  // Set Monarch tokenizer
  monaco.languages.setMonarchTokensProvider(SWARM_LANGUAGE_ID, swarmMonarchLanguage);

  // Register completion provider
  monaco.languages.registerCompletionItemProvider(SWARM_LANGUAGE_ID, createSwarmCompletionProvider());

  console.log('Registered .swarm language for Monaco Editor');
}

/**
 * Custom theme tokens for .swarm files
 */
export const swarmThemeRules: monaco.editor.ITokenThemeRule[] = [
  { token: 'keyword.swarm', foreground: 'c586c0' },
  { token: 'keyword.block.swarm', foreground: '4ec9b0' },
  { token: 'type.agent.swarm', foreground: '4fc1ff' },
  { token: 'constant.tier.swarm', foreground: 'dcdcaa' },
  { token: 'function.builtin.swarm', foreground: 'dcdcaa' },
  { token: 'variable.property.swarm', foreground: '9cdcfe' },
  { token: 'variable.interpolation.swarm', foreground: 'ce9178' },
  { token: 'operator.arrow.swarm', foreground: 'd4d4d4', fontStyle: 'bold' },
  { token: 'operator.range.swarm', foreground: 'd4d4d4' },
  { token: 'string.swarm', foreground: 'ce9178' },
  { token: 'number.swarm', foreground: 'b5cea8' },
  { token: 'number.float.swarm', foreground: 'b5cea8' },
  { token: 'comment.swarm', foreground: '6a9955' },
];

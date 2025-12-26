/**
 * Query Key Factory for TanStack Query
 *
 * Provides type-safe, hierarchical query keys for all Tauri commands.
 * Following the factory pattern from TkDodo's blog.
 */

export const queryKeys = {
  // Cluster domain
  cluster: {
    all: ['cluster'] as const,
    status: () => [...queryKeys.cluster.all, 'status'] as const,
    nodes: () => [...queryKeys.cluster.all, 'nodes'] as const,
  },

  // Training domain
  training: {
    all: ['training'] as const,
    jobs: () => [...queryKeys.training.all, 'jobs'] as const,
    job: (id: string) => [...queryKeys.training.all, 'job', id] as const,
  },

  // Notebook domain
  notebook: {
    all: ['notebook'] as const,
    variables: () => [...queryKeys.notebook.all, 'variables'] as const,
  },

  // Integration domains
  integration: {
    all: ['integration'] as const,

    // Training Pipeline (#1)
    trainingPipeline: {
      all: () => [...queryKeys.integration.all, 'training-pipeline'] as const,
      memoryPressure: () => [...queryKeys.integration.trainingPipeline.all(), 'memory-pressure'] as const,
      workers: () => [...queryKeys.integration.trainingPipeline.all(), 'workers'] as const,
      events: () => [...queryKeys.integration.trainingPipeline.all(), 'events'] as const,
    },

    // Data Routing (#2)
    dataRouting: {
      all: () => [...queryKeys.integration.all, 'data-routing'] as const,
      transfers: () => [...queryKeys.integration.dataRouting.all(), 'transfers'] as const,
      affinityMaps: () => [...queryKeys.integration.dataRouting.all(), 'affinity-maps'] as const,
      events: () => [...queryKeys.integration.dataRouting.all(), 'events'] as const,
    },

    // Collaboration (#3)
    collaboration: {
      all: () => [...queryKeys.integration.all, 'collaboration'] as const,
      session: () => [...queryKeys.integration.collaboration.all(), 'session'] as const,
      collaborators: () => [...queryKeys.integration.collaboration.all(), 'collaborators'] as const,
      events: () => [...queryKeys.integration.collaboration.all(), 'events'] as const,
    },

    // Resource Management (#4)
    resources: {
      all: () => [...queryKeys.integration.all, 'resources'] as const,
      quotas: () => [...queryKeys.integration.resources.all(), 'quotas'] as const,
      policyAdjustments: () => [...queryKeys.integration.resources.all(), 'policy-adjustments'] as const,
      events: () => [...queryKeys.integration.resources.all(), 'events'] as const,
    },

    // Observability (#5)
    observability: {
      all: () => [...queryKeys.integration.all, 'observability'] as const,
      kpis: () => [...queryKeys.integration.observability.all(), 'kpis'] as const,
      alerts: () => [...queryKeys.integration.observability.all(), 'alerts'] as const,
      events: () => [...queryKeys.integration.observability.all(), 'events'] as const,
    },

    // Evolution (#6)
    evolution: {
      all: () => [...queryKeys.integration.all, 'evolution'] as const,
      configs: () => [...queryKeys.integration.evolution.all(), 'configs'] as const,
      stats: () => [...queryKeys.integration.evolution.all(), 'stats'] as const,
      events: () => [...queryKeys.integration.evolution.all(), 'events'] as const,
    },
  },
  // SLAI Scheduler domain
  slai: {
    all: ['slai'] as const,
    stats: () => [...queryKeys.slai.all, 'stats'] as const,
    gpus: () => [...queryKeys.slai.all, 'gpus'] as const,
    fairShare: () => [...queryKeys.slai.all, 'fair-share'] as const,
    tenants: () => [...queryKeys.slai.all, 'tenants'] as const,
    jobs: () => [...queryKeys.slai.all, 'jobs'] as const,
  },
} as const

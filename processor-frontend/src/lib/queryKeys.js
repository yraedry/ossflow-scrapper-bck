// Central query-key registry — every hook MUST use these factories so
// invalidation from mutations is mechanical and typo-free.
// Convention: top-level domain, then action/arg segments.

export const qk = {
  library: {
    all: ['library'],
    list: () => ['library', 'list'],
    detail: (name) => ['library', 'detail', name],
    metadata: (name) => ['library', 'metadata', name],
  },
  pipelines: {
    all: ['pipelines'],
    list: (filters) => ['pipelines', 'list', filters || {}],
    detail: (id) => ['pipelines', 'detail', id],
    eta: (args) => ['pipelines', 'eta', args || {}],
  },
  jobs: {
    all: ['jobs'],
    list: (type) => ['jobs', 'list', type || null],
    detail: (id) => ['jobs', 'detail', id],
  },
  settings: {
    all: ['settings'],
  },
  mount: {
    all: ['mount'],
  },
  oracle: {
    all: ['oracle'],
    providers: ['oracle', 'providers'],
    detail: (path) => ['oracle', 'detail', path],
    search: (query) => ['oracle', 'search', query],
  },
  logs: {
    all: ['logs'],
    list: (filters) => ['logs', filters || {}],
  },
  metrics: {
    all: ['metrics'],
  },
  cleanup: {
    all: ['cleanup'],
    job: (id) => ['cleanup', 'job', id],
  },
  duplicates: {
    all: ['duplicates'],
    job: (id) => ['duplicates', 'job', id],
  },
  voices: {
    all: ['voices'],
    list: () => ['voices', 'list'],
  },
  search: {
    all: ['search'],
    query: (q) => ['search', 'query', q],
  },
  preflight: {
    all: ['preflight'],
    detail: (path) => ['preflight', 'detail', path],
    static: ['preflight', 'static'],
  },
  subtitles: {
    all: ['subtitles'],
    validate: (srtPath) => ['subtitles', 'validate', srtPath],
  },
  telegram: {
    all: ['telegram'],
    status: ['telegram', 'status'],
    channels: ['telegram', 'channels'],
    media: (filters) => ['telegram', 'media', filters || {}],
    downloads: ['telegram', 'downloads'],
    activeSyncs: ['telegram', 'active-syncs'],
  },
}

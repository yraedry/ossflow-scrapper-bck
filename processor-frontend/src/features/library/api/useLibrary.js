// Library domain hooks.
// Endpoints (verified against processor-api/api/app.py & metadata.py & chapters.py):
//   POST   /api/scan                         → scan library (body: { path })
//   GET    /api/library                      → list instructionals
//   GET    /api/library/{name}               → instructional detail
//   POST   /api/library/{name}/poster        → upload poster (multipart)
//   GET    /api/library/{name}/metadata      → metadata
//   PUT    /api/library/{name}/metadata      → update metadata
//   PATCH  /api/chapters/rename              → rename a chapter (body old_path/new_title)
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { http } from '@/lib/httpClient'
import { qk } from '@/lib/queryKeys'

const enc = encodeURIComponent

export function useLibrary() {
  return useQuery({
    queryKey: qk.library.list(),
    queryFn: async () => {
      // refresh=1 kicks a background rescan on the backend (non-blocking)
      // and returns the current cache immediately. Returns a tuple so the
      // UI can poll faster while a refresh is in flight.
      const data = await http.get('/library?refresh=1')
      if (Array.isArray(data)) return { items: data, refreshing: false }
      return {
        items: data?.instructionals || data?.items || [],
        refreshing: Boolean(data?.refreshing),
      }
    },
    select: (d) => d?.items || [],
    staleTime: 5_000,
    // Poll fast while backend reports a refresh in progress, otherwise idle.
    refetchInterval: (q) => (q?.state?.data?.refreshing ? 2_000 : 60_000),
    refetchOnWindowFocus: true,
    refetchOnMount: true,
  })
}

export function useInstructional(name) {
  return useQuery({
    queryKey: qk.library.detail(name),
    queryFn: () => http.get(`/library/${enc(name)}`),
    enabled: !!name,
    staleTime: 10_000,
    refetchOnWindowFocus: true,
    refetchInterval: 15_000,
  })
}

export function useRefreshInstructional() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ name }) => http.post(`/library/${enc(name)}/refresh`),
    onSuccess: (_d, { name }) => {
      qc.invalidateQueries({ queryKey: qk.library.detail(name) })
      qc.invalidateQueries({ queryKey: qk.library.list() })
    },
  })
}

export function useInstructionalMetadata(name) {
  return useQuery({
    queryKey: qk.library.metadata(name),
    queryFn: () => http.get(`/library/${enc(name)}/metadata`),
    enabled: !!name,
    staleTime: 60_000,
  })
}

export function useScanLibrary() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ path }) => http.post('/scan', { path }),
    onSuccess: () => qc.invalidateQueries({ queryKey: qk.library.all }),
  })
}

export function useBurnSubs() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ path }) => http.post('/burn-subs', { path }),
    onSuccess: () => qc.invalidateQueries({ queryKey: qk.jobs.all }),
  })
}

export { useBurnSubs as useBurnSubsVideo }

export function useRenameChapter() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ oldPath, newTitle }) =>
      http.patch('/chapters/rename', { old_path: oldPath, new_title: newTitle }),
    // Optimistic update on the instructional detail: replace the chapter name
    // if we can find the matching entry by its old path.
    onMutate: async ({ instructionalName, oldPath, newTitle }) => {
      if (!instructionalName) return
      const key = qk.library.detail(instructionalName)
      await qc.cancelQueries({ queryKey: key })
      const prev = qc.getQueryData(key)
      if (prev) {
        qc.setQueryData(key, (old) => {
          if (!old) return old
          const replace = (list) =>
            (list || []).map((item) =>
              item?.path === oldPath ? { ...item, title: newTitle, _optimistic: true } : item,
            )
          // Normalise the shape: seasons/chapters nesting if present.
          if (Array.isArray(old.seasons)) {
            return {
              ...old,
              seasons: old.seasons.map((s) => ({ ...s, chapters: replace(s.chapters) })),
            }
          }
          if (Array.isArray(old.chapters)) return { ...old, chapters: replace(old.chapters) }
          return old
        })
      }
      return { prev, key }
    },
    onError: (_err, _vars, ctx) => {
      if (ctx?.prev && ctx.key) qc.setQueryData(ctx.key, ctx.prev)
    },
    onSettled: (_data, _err, vars, ctx) => {
      if (ctx?.key) qc.invalidateQueries({ queryKey: ctx.key })
      if (vars?.instructionalName) {
        qc.invalidateQueries({ queryKey: qk.library.detail(vars.instructionalName) })
      }
    },
  })
}

export function useRenameByOracle() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ seasonPath, oracle, instructionalName }) =>
      http.post('/chapters/rename-by-oracle', { season_path: seasonPath, oracle, instructional_name: instructionalName }),
    onSettled: (_data, _err, vars) => {
      if (vars?.instructionalName) {
        qc.invalidateQueries({ queryKey: qk.library.detail(vars.instructionalName) })
      }
    },
  })
}

export function useUploadPoster() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: async ({ name, file }) => {
      const fd = new FormData()
      fd.append('file', file)
      return http.post(`/library/${enc(name)}/poster`, fd)
    },
    onSuccess: (_d, { name }) => {
      qc.invalidateQueries({ queryKey: qk.library.detail(name) })
      qc.invalidateQueries({ queryKey: qk.library.list() })
    },
  })
}

export function useRedownloadPoster() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ name }) => http.post(`/library/${enc(name)}/poster/redownload`),
    onSuccess: (_d, { name }) => {
      qc.invalidateQueries({ queryKey: qk.library.detail(name) })
      qc.invalidateQueries({ queryKey: qk.library.list() })
    },
  })
}

export function useUpdateMetadata() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ name, data }) => http.put(`/library/${enc(name)}/metadata`, data),
    onSuccess: (_d, { name }) => {
      qc.invalidateQueries({ queryKey: qk.library.metadata(name) })
      qc.invalidateQueries({ queryKey: qk.library.detail(name) })
    },
  })
}

export function posterUrl(name) {
  return `/api/library/${enc(name)}/poster`
}

export function useVoiceProfiles() {
  return useQuery({
    queryKey: ['dubbing', 'voices'],
    queryFn: async () => {
      const data = await http.get('/dubbing/voices')
      return Array.isArray(data?.voices) ? data.voices : []
    },
    staleTime: 60_000,
  })
}

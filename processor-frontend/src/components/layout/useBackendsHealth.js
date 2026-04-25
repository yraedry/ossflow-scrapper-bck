import { useQuery } from '@tanstack/react-query'
import { http } from '@/lib/httpClient'

export const BACKENDS = [
  { id: 'chapter-splitter', label: 'Chapter Splitter' },
  { id: 'subtitle-generator', label: 'Subtitle Generator' },
  { id: 'dubbing-generator', label: 'Dubbing Generator' },
  { id: 'ollama', label: 'Ollama' },
]

export function useBackendsHealth() {
  const { data, isPending, isError } = useQuery({
    queryKey: ['backend-health', 'all'],
    queryFn: () => http.get('/health/backends'),
    refetchInterval: 10_000,
    retry: false,
    staleTime: 5_000,
  })

  const byId = new Map((data?.services || []).map((s) => [s.service, s]))
  return BACKENDS.map((b) => {
    if (isPending) return { ...b, status: 'loading' }
    if (isError) return { ...b, status: 'down', error: 'proxy unreachable' }
    const row = byId.get(b.id)
    if (!row) return { ...b, status: 'down', error: 'missing in proxy response' }
    return {
      ...b,
      status: row.status,
      error: row.error,
      data: { body: row.body },
    }
  })
}

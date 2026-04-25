// Logs — aggregated ring-buffer view from processor-api/api/logs_view.py
//   GET /api/logs/?service=&level=&tail=
import { useQuery } from '@tanstack/react-query'
import { http, qs } from '@/lib/httpClient'
import { qk } from '@/lib/queryKeys'

export function useLogs(filters = {}) {
  const { service, level, tail = 500 } = filters
  const key = { service: service || null, level: level || null, tail }
  return useQuery({
    queryKey: qk.logs.list(key),
    queryFn: () => http.get(`/logs/${qs({ service, level, tail })}`),
    refetchInterval: 5_000,
    staleTime: 0,
  })
}

// Dub QA hooks.
//
// * `useDubQa(videoPath)` → single chapter sidecar. Used by the inline
//   badge. 404 is swallowed so badge stays hidden when no QA exists yet.
// * `useInstructionalQa(name)` → aggregate across all chapters of an
//   instructional. Powers the "QA" tab.
import { useQuery } from '@tanstack/react-query'
import { http, HttpError } from '@/lib/httpClient'

export function useDubQa(videoPath, { enabled = true } = {}) {
  return useQuery({
    queryKey: ['dub-qa', videoPath],
    enabled: Boolean(videoPath) && enabled,
    queryFn: async () => {
      const url = `/dubbing/qa?video_path=${encodeURIComponent(videoPath)}`
      try {
        return await http.get(url)
      } catch (err) {
        if (err instanceof HttpError && err.status === 404) return null
        throw err
      }
    },
    staleTime: 60_000,
    retry: false,
  })
}

export function useInstructionalQa(name, { enabled = true } = {}) {
  return useQuery({
    queryKey: ['dub-qa', 'instructional', name],
    enabled: Boolean(name) && enabled,
    queryFn: () => http.get(`/dubbing/qa/instructional/${encodeURIComponent(name)}`),
    staleTime: 30_000,
  })
}

// Promote a dubbed video to its final multi-track form.
//
// Endpoints (see processor-api/api/promote.py):
//   POST /api/promote/chapter  → mux + cleanup for one video
//   POST /api/promote/season   → batch over a season
//
// Both invalidate the library cache on success so the UI re-renders the
// affected instructional with the new is_promoted flag.

import { useMutation, useQueryClient } from '@tanstack/react-query'
import { http } from '@/lib/httpClient'

export function useStartPromote() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ videoPath }) =>
      http.post('/promote/chapter', { video_path: videoPath }),
    onSuccess: () => {
      // Re-fetch the library so the chapter row picks up is_promoted=true
      // (and hides the Promote button) without waiting for a manual rescan.
      qc.invalidateQueries({ queryKey: ['library'] })
      qc.invalidateQueries({ queryKey: ['scan'] })
    },
  })
}

export function useStartPromoteSeason() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ seasonPath }) =>
      http.post('/promote/season', { season_path: seasonPath }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['library'] })
      qc.invalidateQueries({ queryKey: ['scan'] })
    },
  })
}

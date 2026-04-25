// ElevenLabs Dubbing Studio E2E hooks.
//
// Separate module from the XTTS pipeline (`features/pipeline`) because:
//   - It does NOT go through /api/pipeline — it has its own router.
//   - The job lifecycle is simpler (no step chaining, no SRT).
//   - We want it visually clearly separated in the UI so the user knows
//     which provider a given dub came from.
//
// Endpoints (see processor-api/api/elevenlabs_dubbing.py):
//   POST /api/elevenlabs-dubbing            → start a dubbing job
//   POST /api/elevenlabs-dubbing/batch      → queue every video in a Season
//   GET  /api/elevenlabs-dubbing            → { active, recent }
//   GET  /api/elevenlabs-dubbing/{id}       → status (also /api/jobs/{id})

import { useMutation, useQuery } from '@tanstack/react-query'
import { http } from '@/lib/httpClient'

export function useStartElevenLabsDubbing() {
  return useMutation({
    mutationFn: ({ path, sourceLang = 'en', targetLang = 'es', numSpeakers = 1, watermark = true }) =>
      http.post('/elevenlabs-dubbing', {
        path,
        source_lang: sourceLang,
        target_lang: targetLang,
        num_speakers: numSpeakers,
        watermark,
      }),
  })
}

export function useStartElevenLabsBatch() {
  return useMutation({
    mutationFn: ({ seasonPath, sourceLang = 'en', targetLang = 'es', numSpeakers = 1, watermark = true }) =>
      http.post('/elevenlabs-dubbing/batch', {
        season_path: seasonPath,
        source_lang: sourceLang,
        target_lang: targetLang,
        num_speakers: numSpeakers,
        watermark,
      }),
  })
}

export function useElevenLabsJob(jobId) {
  return useQuery({
    queryKey: ['elevenlabs-dubbing', jobId],
    queryFn: () => http.get(`/jobs/${jobId}`),
    enabled: !!jobId,
    // 5s poll — ElevenLabs dubs an episode in 3-8 min, so finer granularity
    // is wasted. The SSE stream (/api/jobs/{id}/events) is the real-time
    // source; this query is just a safety fallback.
    refetchInterval: (q) => {
      const status = q.state.data?.status
      if (status === 'completed' || status === 'failed') return false
      return 5_000
    },
    staleTime: 0,
  })
}

// Aggregated list used by the /elevenlabs page. Refreshes aggressively
// while any job is active so cards animate without SSE on the list view.
export function useElevenLabsJobs() {
  return useQuery({
    queryKey: ['elevenlabs-dubbing', 'list'],
    queryFn: () => http.get('/elevenlabs-dubbing'),
    refetchInterval: (q) => {
      const active = q.state.data?.active || []
      return active.length > 0 ? 3_000 : 15_000
    },
    staleTime: 0,
  })
}

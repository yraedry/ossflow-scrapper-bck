// Background dubbing analysis state.
// Parallel to useAudioAnalysis but targets /api/dubbing/analyze endpoint.
import { create } from 'zustand'
import { http } from '@/lib/httpClient'

const useDubbingAnalysis = create((set, get) => ({
  // { [videoPath]: { status: 'pending'|'done'|'error', data, videoPath, synthesize } }
  jobs: {},

  launch(videoPath, { synthesize = false, maxPhrases = null } = {}) {
    const { jobs } = get()
    if (jobs[videoPath]?.status === 'pending') return

    set({
      jobs: {
        ...get().jobs,
        [videoPath]: { status: 'pending', data: null, videoPath, synthesize },
      },
    })

    const body = { video_path: videoPath, synthesize }
    if (maxPhrases != null) body.max_phrases = maxPhrases

    http
      .post('/dubbing/analyze', body)
      .then((result) => {
        set({
          jobs: {
            ...get().jobs,
            [videoPath]: { status: 'done', data: result, videoPath, synthesize },
          },
        })
      })
      .catch((e) => {
        const detail = e?.response?.data?.detail || e?.message || 'Error desconocido'
        set({
          jobs: {
            ...get().jobs,
            [videoPath]: { status: 'error', data: { error: detail }, videoPath, synthesize },
          },
        })
      })
  },

  clear(videoPath) {
    const next = { ...get().jobs }
    delete next[videoPath]
    set({ jobs: next })
  },

  get(videoPath) {
    return get().jobs[videoPath] || null
  },
}))

export default useDubbingAnalysis

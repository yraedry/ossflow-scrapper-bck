// Silent, app-wide listener for ElevenLabs job transitions.
//
// Why: when the user launches a dub and navigates away (or even starts
// one from another route that doesn't have the ChaptersTab toast
// wired), we still want a loud "Job XYZ listo" / "Job XYZ falló" toast.
// The ElevenLabsPage shows live state when the user is there; this
// watcher handles everything that happens elsewhere.
//
// Implementation is deliberately polled (useElevenLabsJobs already does
// 3s / 15s refetch) rather than SSE — a single poll vs N SSE connections
// keeps the browser tab quieter and survives reloads cleanly. We track
// previously-seen terminal job IDs in a ref so the same job doesn't
// re-fire a toast every poll.
import { useEffect, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { toast } from 'sonner'
import { useElevenLabsJobs } from '../api/useElevenLabsDubbing'

export default function ElevenLabsGlobalWatcher() {
  const { data } = useElevenLabsJobs()
  const nav = useNavigate()
  const seenRef = useRef(new Set())
  const seededRef = useRef(false)

  useEffect(() => {
    if (!data) return
    const recent = data.recent || []

    // First payload after mount: treat all terminal jobs as "already
    // seen" so we don't spam toasts for history the user didn't
    // actively start in this tab.
    if (!seededRef.current) {
      for (const j of recent) seenRef.current.add(j.job_id)
      seededRef.current = true
      return
    }

    for (const job of recent) {
      if (seenRef.current.has(job.job_id)) continue
      seenRef.current.add(job.job_id)
      const name =
        job.result?.output_filename ||
        (job.video_path ? job.video_path.split(/[\\/]/).pop() : job.job_id)
      if (job.status === 'completed') {
        toast.success(`ElevenLabs listo: ${name}`, {
          duration: 10000,
          action: { label: 'Ver', onClick: () => nav('/elevenlabs') },
        })
      } else if (job.status === 'failed') {
        toast.error(`ElevenLabs falló: ${name}`, {
          description: job.message || undefined,
          duration: 15000,
          action: { label: 'Ver', onClick: () => nav('/elevenlabs') },
        })
      }
    }
  }, [data, nav])

  return null
}

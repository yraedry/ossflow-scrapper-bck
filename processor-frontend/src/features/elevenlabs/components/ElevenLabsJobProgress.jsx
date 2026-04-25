// Live progress panel for an in-flight ElevenLabs Dubbing Studio job.
//
// Why a component, not just a toast: the ElevenLabs API does NOT expose
// a numeric progress value — only a binary-ish status (`dubbing` vs
// `dubbed`). To avoid a silent spinner we read SSE events from
// /api/jobs/{id}/events and derive:
//   - a deterministic % from elapsed / estimated_total_sec (computed
//     server-side from ffprobe duration × 0.28)
//   - a named stage label (Subiendo / Transcribiendo / ...)
//   - elapsed & remaining clocks
//
// The bar caps at 95% until the real `dubbed` status arrives, so we
// never claim "done" before ElevenLabs says so.

import { useEffect, useMemo, useRef, useState } from 'react'
import { Loader2, CheckCircle2, XCircle, ExternalLink } from 'lucide-react'
import { cn } from '@/lib/utils'

function fmtSec(s) {
  if (s == null || isNaN(s) || s < 0) return '—'
  const total = Math.round(s)
  const m = Math.floor(total / 60)
  const sec = total % 60
  return m > 0 ? `${m}m ${sec.toString().padStart(2, '0')}s` : `${sec}s`
}

export default function ElevenLabsJobProgress({ jobId, filename, onClose }) {
  const [event, setEvent] = useState(null)
  const [error, setError] = useState(null)
  const esRef = useRef(null)

  useEffect(() => {
    if (!jobId) return
    // EventSource auto-reconnects; no manual retry needed.
    const es = new EventSource(`/api/jobs/${jobId}/events`)
    esRef.current = es
    es.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data)
        setEvent(data)
      } catch {
        // SSE comment / keepalive — ignore
      }
    }
    es.onerror = () => {
      // EventSource surfaces transient errors we don't need to show —
      // the server closes the stream after terminal events.
      setError(null)
    }
    return () => {
      es.close()
      esRef.current = null
    }
  }, [jobId])

  const status = event?.status
  const isDone = status === 'completed'
  const isFailed = status === 'failed'
  const progress = event?.progress ?? 0
  const stage = event?.stage || event?.message || 'Esperando...'
  const elapsed = event?.elapsed_sec
  const remaining = event?.estimated_remaining_sec
  const totalEstimated = event?.estimated_total_sec
  const result = event?.result

  const barColor = useMemo(() => {
    if (isFailed) return 'bg-rose-500'
    if (isDone) return 'bg-emerald-500'
    return 'bg-violet-500'
  }, [isDone, isFailed])

  return (
    <div className="w-full max-w-md rounded-lg border border-zinc-800 bg-zinc-900/95 p-4 shadow-xl">
      <div className="mb-2 flex items-start justify-between gap-2">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 text-xs font-medium text-violet-300">
            {isDone ? (
              <CheckCircle2 className="h-4 w-4 text-emerald-400" />
            ) : isFailed ? (
              <XCircle className="h-4 w-4 text-rose-400" />
            ) : (
              <Loader2 className="h-4 w-4 animate-spin" />
            )}
            <span>ElevenLabs Dubbing</span>
            <span className="font-mono text-[10px] text-zinc-500">#{jobId}</span>
          </div>
          <p className="mt-1 truncate text-sm text-zinc-200" title={filename}>
            {filename}
          </p>
        </div>
        {onClose && (
          <button
            type="button"
            onClick={onClose}
            className="rounded p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-300"
            aria-label="Cerrar"
          >
            ×
          </button>
        )}
      </div>

      <div
        role="progressbar"
        aria-valuenow={progress}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={`Progreso ${progress}%`}
        className="relative mt-3 h-2 w-full overflow-hidden rounded-full bg-zinc-800"
      >
        <div
          className={cn('h-full transition-all duration-500 ease-out', barColor)}
          style={{ width: `${Math.min(100, Math.max(0, progress))}%` }}
        />
      </div>

      <div className="mt-2 flex items-center justify-between text-xs text-zinc-400">
        <span className="truncate">{stage}</span>
        <span className="font-mono tabular-nums">
          {progress}%
        </span>
      </div>

      {!isDone && !isFailed && (
        <div className="mt-2 grid grid-cols-3 gap-2 text-[11px] text-zinc-500">
          <div>
            <div className="text-zinc-600">Transcurrido</div>
            <div className="font-mono tabular-nums text-zinc-300">{fmtSec(elapsed)}</div>
          </div>
          <div>
            <div className="text-zinc-600">Estimado total</div>
            <div className="font-mono tabular-nums text-zinc-300">{fmtSec(totalEstimated)}</div>
          </div>
          <div>
            <div className="text-zinc-600">Faltan</div>
            <div className="font-mono tabular-nums text-zinc-300">~{fmtSec(remaining)}</div>
          </div>
        </div>
      )}

      {isDone && result?.output_path && (
        <div className="mt-3 rounded bg-emerald-500/10 p-2 text-xs text-emerald-200">
          <div className="font-medium">Guardado en NAS</div>
          <div className="mt-1 break-all font-mono text-[10px] text-emerald-300/80">
            {result.output_path}
          </div>
          {result.total_elapsed_sec != null && (
            <div className="mt-1 text-emerald-300/70">
              Tiempo total: {fmtSec(result.total_elapsed_sec)}
            </div>
          )}
          {result.dubbing_id && (
            <a
              href={`https://elevenlabs.io/app/dubbing/${result.dubbing_id}`}
              target="_blank"
              rel="noreferrer"
              className="mt-2 inline-flex items-center gap-1 text-emerald-300 hover:text-emerald-200"
            >
              Ver en ElevenLabs
              <ExternalLink className="h-3 w-3" />
            </a>
          )}
        </div>
      )}

      {isFailed && (
        <div className="mt-3 rounded bg-rose-500/10 p-2 text-xs text-rose-200">
          <div className="font-medium">Error</div>
          <div className="mt-1 break-words text-rose-300/80">
            {event?.message || 'Unknown error'}
          </div>
        </div>
      )}
    </div>
  )
}

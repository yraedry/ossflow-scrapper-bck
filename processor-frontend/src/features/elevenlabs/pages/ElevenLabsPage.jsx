// /elevenlabs — ElevenLabs Dubbing Studio dashboard.
//
// Separate from /pipelines because the ElevenLabs flow has its own
// lifecycle: no steps, no chaining, just upload → poll → download. A
// shared view would have to fake "steps" for each job which would
// confuse both codebases. Two sections:
//
//   1. Activos       — jobs in queue or running, with live SSE progress
//   2. Recientes     — last 50 completed/failed, newest first
//
// The active cards subscribe to /api/jobs/{id}/events individually via
// ElevenLabsJobProgress, so refreshing the page or closing/reopening
// picks the SSE stream back up from whatever state the server holds.
import { useMemo } from 'react'
import { Waves, CheckCircle2, XCircle, ExternalLink, Loader2 } from 'lucide-react'
import { useElevenLabsJobs } from '../api/useElevenLabsDubbing'
import ElevenLabsJobProgress from '../components/ElevenLabsJobProgress'
import { Badge } from '@/components/ui/Badge'
import { cn } from '@/lib/utils'

function fmtSec(s) {
  if (s == null || isNaN(s) || s < 0) return '—'
  const total = Math.round(s)
  const m = Math.floor(total / 60)
  const sec = total % 60
  return m > 0 ? `${m}m ${sec.toString().padStart(2, '0')}s` : `${sec}s`
}

function fmtDate(iso) {
  if (!iso) return '—'
  try {
    const d = new Date(iso)
    return d.toLocaleString()
  } catch {
    return iso
  }
}

function shortName(path) {
  if (!path) return ''
  return path.split(/[\\/]/).pop() || path
}

function RecentRow({ job }) {
  const isDone = job.status === 'completed'
  const isFailed = job.status === 'failed'
  const result = job.result || {}
  return (
    <div className="flex items-start gap-3 rounded border border-zinc-800 bg-zinc-900/50 p-3">
      <div className="mt-0.5 shrink-0">
        {isDone ? (
          <CheckCircle2 className="h-4 w-4 text-emerald-400" />
        ) : isFailed ? (
          <XCircle className="h-4 w-4 text-rose-400" />
        ) : (
          <Loader2 className="h-4 w-4 animate-spin text-zinc-400" />
        )}
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <span className="truncate text-sm text-zinc-200" title={job.video_path}>
            {shortName(job.video_path)}
          </span>
          <Badge
            className={cn(
              'shrink-0 text-[10px]',
              isDone && 'bg-emerald-500/15 text-emerald-400',
              isFailed && 'bg-rose-500/15 text-rose-400',
            )}
          >
            {job.status}
          </Badge>
        </div>
        <div className="mt-1 flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-zinc-500">
          <span>#{job.job_id}</span>
          <span>{fmtDate(job.completed_at || job.created_at)}</span>
          {result.total_elapsed_sec != null && (
            <span>Duración: {fmtSec(result.total_elapsed_sec)}</span>
          )}
          {result.bytes != null && (
            <span>{(result.bytes / (1024 * 1024)).toFixed(1)} MB</span>
          )}
        </div>
        {isFailed && job.message && (
          <div className="mt-1 truncate text-xs text-rose-300/80" title={job.message}>
            {job.message}
          </div>
        )}
        {isDone && result.output_path && (
          <div className="mt-1 truncate font-mono text-[10px] text-emerald-300/70">
            {result.output_path}
          </div>
        )}
      </div>
      {isDone && result.dubbing_id && (
        <a
          href={`https://elevenlabs.io/app/dubbing/${result.dubbing_id}`}
          target="_blank"
          rel="noreferrer"
          className="shrink-0 text-zinc-500 hover:text-violet-300"
          title="Ver en ElevenLabs Studio"
        >
          <ExternalLink className="h-4 w-4" />
        </a>
      )}
    </div>
  )
}

export default function ElevenLabsPage() {
  const { data, isLoading } = useElevenLabsJobs()
  const active = data?.active || []
  const recent = data?.recent || []

  const credits = useMemo(() => {
    // Rough credit estimate for the user: 1 min ≈ 1000 chars (ES dub)
    // — just a hint, not authoritative.
    const running = active.length
    return { running }
  }, [active])

  return (
    <div className="mx-auto max-w-5xl space-y-6 p-4">
      <header className="flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-violet-500/10 text-violet-400">
          <Waves className="h-5 w-5" />
        </div>
        <div>
          <h1 className="text-xl font-semibold text-zinc-100">ElevenLabs Dubbing</h1>
          <p className="text-xs text-zinc-500">
            Cola serial — 1 doblaje activo a la vez. Cierra o refresca esta página sin miedo:
            los jobs siguen en el servidor.
          </p>
        </div>
        <div className="ml-auto text-right">
          <div className="text-xs text-zinc-500">Activos</div>
          <div className="text-2xl font-semibold tabular-nums text-zinc-100">
            {credits.running}
          </div>
        </div>
      </header>

      <section>
        <h2 className="mb-2 text-sm font-medium text-zinc-400">Activos</h2>
        {isLoading ? (
          <div className="rounded border border-zinc-800 bg-zinc-900/50 p-4 text-sm text-zinc-500">
            Cargando…
          </div>
        ) : active.length === 0 ? (
          <div className="rounded border border-dashed border-zinc-800 p-6 text-center text-sm text-zinc-500">
            No hay doblajes en curso.
          </div>
        ) : (
          <div className="space-y-3">
            {active.map((job) => (
              <ElevenLabsJobProgress
                key={job.job_id}
                jobId={job.job_id}
                filename={shortName(job.video_path)}
              />
            ))}
          </div>
        )}
      </section>

      <section>
        <h2 className="mb-2 text-sm font-medium text-zinc-400">Recientes</h2>
        {recent.length === 0 ? (
          <div className="rounded border border-dashed border-zinc-800 p-6 text-center text-sm text-zinc-500">
            Sin historial todavía.
          </div>
        ) : (
          <div className="space-y-2">
            {recent.map((job) => (
              <RecentRow key={job.job_id} job={job} />
            ))}
          </div>
        )}
      </section>
    </div>
  )
}

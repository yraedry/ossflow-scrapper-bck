// Validate all subtitles in a season at once — shows per-chapter summary with expandable issues.
import { useState, useEffect } from 'react'
import { Loader2, AlertCircle, AlertTriangle, Info, Check, ChevronDown, ChevronRight, ShieldCheck } from 'lucide-react'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog'
import { Badge } from '@/components/ui/Badge'
import { cn } from '@/lib/utils'
import { http } from '@/lib/httpClient'

function fmtTime(sec) {
  if (sec == null || sec < 0) return '—'
  const m = Math.floor(sec / 60)
  const s = (sec % 60).toFixed(1)
  return `${m}:${s.padStart(4, '0')}`
}

const SEV_STYLES = {
  error: 'bg-red-500/15 text-red-300 border-red-500/30',
  warn: 'bg-amber-500/15 text-amber-300 border-amber-500/30',
  info: 'bg-sky-500/15 text-sky-300 border-sky-500/30',
  ok: 'bg-emerald-500/15 text-emerald-300 border-emerald-500/30',
}
const SEV_ICON = { error: AlertCircle, warn: AlertTriangle, info: Info, ok: Check }

function srtPathFor(videoPath) {
  const dot = videoPath.lastIndexOf('.')
  return dot > 0 ? `${videoPath.slice(0, dot)}.en.srt` : `${videoPath}.en.srt`
}

function severityOf(summary) {
  if ((summary?.by_severity?.error || 0) > 0) return 'error'
  if ((summary?.by_severity?.warn || 0) > 0) return 'warn'
  if ((summary?.by_severity?.info || 0) > 0) return 'info'
  return 'ok'
}

function ChapterResult({ video, result, loading, error }) {
  const [open, setOpen] = useState(false)
  const filename = video.filename || video.path?.split('/').pop() || '—'
  const sev = error ? 'error' : (result ? severityOf(result.summary) : null)
  const Icon = sev ? (SEV_ICON[sev] || Check) : null
  const issues = result?.segments?.filter((s) => s.severity !== 'ok') || []

  return (
    <div className="border-t border-zinc-800/60">
      <button
        type="button"
        className="flex w-full items-center gap-2 px-3 py-2 text-left hover:bg-zinc-900/40 transition-colors"
        onClick={() => issues.length > 0 && setOpen((v) => !v)}
        disabled={!issues.length}
      >
        <span className="text-zinc-500 w-4 shrink-0">
          {issues.length > 0
            ? (open ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />)
            : <span className="h-3 w-3 block" />}
        </span>

        {loading && <Loader2 className="h-3 w-3 animate-spin text-zinc-500 shrink-0" />}
        {Icon && !loading && (
          <Icon className={cn('h-3.5 w-3.5 shrink-0', {
            'text-red-400': sev === 'error',
            'text-amber-400': sev === 'warn',
            'text-sky-400': sev === 'info',
            'text-emerald-400': sev === 'ok',
          })} />
        )}

        <span className="flex-1 truncate text-xs text-zinc-300 font-mono">{filename}</span>

        {error && (
          <span className="text-[10px] text-red-400 shrink-0">{error}</span>
        )}
        {result && (
          <div className="flex items-center gap-1.5 shrink-0">
            {(result.summary.by_severity.error || 0) > 0 && (
              <span className={cn('rounded border px-1 py-0.5 text-[10px]', SEV_STYLES.error)}>
                {result.summary.by_severity.error} err
              </span>
            )}
            {(result.summary.by_severity.warn || 0) > 0 && (
              <span className={cn('rounded border px-1 py-0.5 text-[10px]', SEV_STYLES.warn)}>
                {result.summary.by_severity.warn} av
              </span>
            )}
            {(result.summary.by_severity.info || 0) > 0 && (
              <span className={cn('rounded border px-1 py-0.5 text-[10px]', SEV_STYLES.info)}>
                {result.summary.by_severity.info} inf
              </span>
            )}
            {sev === 'ok' && (
              <span className={cn('rounded border px-1 py-0.5 text-[10px]', SEV_STYLES.ok)}>ok</span>
            )}
            <span className="text-[10px] text-zinc-500">
              {result.summary.coverage_percent?.toFixed(0)}%
            </span>
          </div>
        )}
      </button>

      {open && issues.length > 0 && (
        <div className="px-8 pb-2">
          <table className="w-full text-xs">
            <tbody>
              {issues.map((seg) => {
                const SIcon = SEV_ICON[seg.severity] || Info
                return (
                  <tr key={seg.idx} className="border-t border-zinc-800/40">
                    <td className="py-1 pr-2 font-mono text-zinc-500 whitespace-nowrap w-6">#{seg.idx}</td>
                    <td className="py-1 pr-2 font-mono text-zinc-500 whitespace-nowrap">
                      {fmtTime(seg.start)} → {fmtTime(seg.end)}
                    </td>
                    <td className="py-1 pr-2 w-14">
                      <span className={cn('inline-flex items-center gap-1 rounded border px-1 py-0.5 text-[10px]', SEV_STYLES[seg.severity])}>
                        <SIcon className="h-2.5 w-2.5" />
                        {seg.severity}
                      </span>
                    </td>
                    <td className="py-1 text-zinc-300">
                      <div className="truncate max-w-xs" title={seg.text}>{seg.text}</div>
                      {seg.issues?.length > 0 && (
                        <div className="text-[10px] text-zinc-500">
                          {seg.issues.map((i) => i.message).join(' · ')}
                        </div>
                      )}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

export default function SeasonValidationDialog({ open, onOpenChange, season, videos }) {
  const withSubs = (videos || []).filter((v) => v.has_subtitles_en)
  const [results, setResults] = useState({})   // path → data
  const [errors, setErrors]   = useState({})   // path → message
  const [loading, setLoading] = useState({})   // path → bool
  const [ran, setRan] = useState(false)

  useEffect(() => {
    if (!open) { setResults({}); setErrors({}); setLoading({}); setRan(false) }
  }, [open])

  const run = async () => {
    setResults({}); setErrors({}); setRan(true)
    const init = Object.fromEntries(withSubs.map((v) => [v.path, true]))
    setLoading(init)
    // Sequential — avoid hammering subtitle-generator
    for (const v of withSubs) {
      try {
        const data = await http.post('/subtitles/validate', { srt_path: srtPathFor(v.path) })
        setResults((p) => ({ ...p, [v.path]: data }))
      } catch (e) {
        setErrors((p) => ({ ...p, [v.path]: e?.message || 'error' }))
      } finally {
        setLoading((p) => ({ ...p, [v.path]: false }))
      }
    }
  }

  const totalErrors = Object.values(results).reduce((n, r) => n + (r.summary?.by_severity?.error || 0), 0)
  const totalWarns  = Object.values(results).reduce((n, r) => n + (r.summary?.by_severity?.warn  || 0), 0)
  const done = ran && Object.values(loading).every((v) => !v)

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl max-h-[85vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <ShieldCheck className="h-4 w-4 text-emerald-400" />
            Validar {season}
          </DialogTitle>
          <DialogDescription className="text-xs text-zinc-500">
            {withSubs.length} capítulos con subtítulos EN
            {videos.length - withSubs.length > 0 && ` · ${videos.length - withSubs.length} sin subs (omitidos)`}
          </DialogDescription>
        </DialogHeader>

        <div className="flex items-center gap-3 border-b border-zinc-800 pb-3">
          <button
            type="button"
            onClick={run}
            disabled={Object.values(loading).some(Boolean) || withSubs.length === 0}
            className="flex items-center gap-1.5 rounded border border-zinc-700 bg-zinc-900 px-3 py-1.5 text-xs text-zinc-200 hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-50 transition-colors"
          >
            {Object.values(loading).some(Boolean)
              ? <Loader2 className="h-3 w-3 animate-spin" />
              : <ShieldCheck className="h-3 w-3" />}
            {ran ? 'Re-validar' : 'Validar todos'}
          </button>

          {done && (
            <>
              {totalErrors > 0 && (
                <Badge className={SEV_STYLES.error}>{totalErrors} errores</Badge>
              )}
              {totalWarns > 0 && (
                <Badge className={SEV_STYLES.warn}>{totalWarns} avisos</Badge>
              )}
              {totalErrors === 0 && totalWarns === 0 && (
                <Badge className={SEV_STYLES.ok}>Todo OK</Badge>
              )}
            </>
          )}
        </div>

        <div className="min-h-0 flex-1 overflow-auto">
          {!ran && withSubs.length === 0 && (
            <p className="py-10 text-center text-sm text-zinc-500">
              Ningún capítulo tiene subtítulos EN en esta season.
            </p>
          )}
          {!ran && withSubs.length > 0 && (
            <p className="py-10 text-center text-sm text-zinc-500">
              Pulsa "Validar todos" para analizar los {withSubs.length} capítulos.
            </p>
          )}
          {ran && withSubs.map((v) => (
            <ChapterResult
              key={v.path}
              video={v}
              result={results[v.path]}
              loading={loading[v.path]}
              error={errors[v.path]}
            />
          ))}
        </div>
      </DialogContent>
    </Dialog>
  )
}

// Debug dialog: deep analysis of dubbing pipeline for a chapter.
// Uses tabs + pagination so no nested scrollbars — content expands naturally.
import { useMemo, useState } from 'react'
import {
  Loader2,
  Mic,
  AlertTriangle,
  Zap,
  ChevronLeft,
  ChevronRight,
} from 'lucide-react'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog'
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/Badge'
import { cn } from '@/lib/utils'
import { toast } from 'sonner'
import useDubbingAnalysis from '../stores/useDubbingAnalysis'

const PAGE_SIZE = 20

function fmtMs(ms) {
  if (ms == null) return '—'
  const s = ms / 1000
  const m = Math.floor(s / 60)
  const r = (s % 60).toFixed(1)
  return `${m}:${r.padStart(4, '0')}`
}

function CompressionBar({ ratio, max }) {
  const over = ratio > max
  const pct = Math.min(100, (ratio / (max * 1.5)) * 100)
  const color = over
    ? 'bg-red-500'
    : ratio > 1.2
      ? 'bg-amber-500'
      : ratio > 1.0
        ? 'bg-yellow-500'
        : 'bg-emerald-500'
  return (
    <div className="flex items-center gap-2">
      <div className="relative h-1.5 w-20 rounded-full bg-zinc-800 overflow-hidden">
        <div className={cn('h-full rounded-full', color)} style={{ width: `${pct}%` }} />
      </div>
      <span className={cn('font-mono text-[11px]', over ? 'text-red-400' : 'text-zinc-400')}>
        {ratio.toFixed(2)}x
      </span>
    </div>
  )
}

function Pager({ page, pageCount, onPage }) {
  if (pageCount <= 1) return null
  return (
    <div className="flex items-center justify-between pt-2 text-xs">
      <Button
        size="sm"
        variant="ghost"
        disabled={page === 0}
        onClick={() => onPage(Math.max(0, page - 1))}
      >
        <ChevronLeft className="h-3 w-3 mr-1" />
        Anterior
      </Button>
      <span className="text-zinc-500">
        Página {page + 1} / {pageCount}
      </span>
      <Button
        size="sm"
        variant="ghost"
        disabled={page >= pageCount - 1}
        onClick={() => onPage(Math.min(pageCount - 1, page + 1))}
      >
        Siguiente
        <ChevronRight className="h-3 w-3 ml-1" />
      </Button>
    </div>
  )
}

function SummaryCards({ summary }) {
  if (!summary) return null
  const overflowColor = summary.will_overflow_count > 0 ? 'text-amber-400' : 'text-emerald-400'
  const compColor = summary.max_compression > summary.config.max_compression_ratio ? 'text-red-400' : 'text-zinc-100'
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      <div className="rounded border border-zinc-800 bg-zinc-900/50 p-3 text-center">
        <p className="text-lg font-bold text-zinc-100">{summary.total_phrases}</p>
        <p className="text-[10px] text-zinc-500">Frases</p>
      </div>
      <div className="rounded border border-zinc-800 bg-zinc-900/50 p-3 text-center">
        <p className="text-lg font-bold text-zinc-100">{fmtMs(summary.srt_duration_ms)}</p>
        <p className="text-[10px] text-zinc-500">Duración SRT</p>
      </div>
      <div className="rounded border border-zinc-800 bg-zinc-900/50 p-3 text-center">
        <p className={cn('text-lg font-bold', overflowColor)}>{summary.will_overflow_count}</p>
        <p className="text-[10px] text-zinc-500">Frases overflow</p>
      </div>
      <div className="rounded border border-zinc-800 bg-zinc-900/50 p-3 text-center">
        <p className={cn('text-lg font-bold', compColor)}>{summary.max_compression?.toFixed?.(2)}x</p>
        <p className="text-[10px] text-zinc-500">Compresión máx</p>
      </div>
    </div>
  )
}

function ConfigRow({ config }) {
  if (!config) return null
  const entries = [
    ['TTS speed', config.tts_speed],
    ['Max comp.', `${config.max_compression_ratio}x`],
    ['Min slot', `${config.min_phrase_duration_ms}ms`],
    ['Overflow', `${config.max_overflow_ms}ms`],
    ['Pad', `${config.inter_phrase_pad_ms}ms`],
    ['Speed range', `${config.speed_min}–${config.speed_max}`],
    ['Ducking bg', config.ducking_bg_volume],
    ['Ducking voz', config.ducking_fg_volume],
  ]
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-zinc-500">
      {entries.map(([k, v]) => (
        <span key={k}>
          {k}: <b className="text-zinc-300">{v}</b>
        </span>
      ))}
    </div>
  )
}

function PlannedTable({ rows, maxComp }) {
  const [page, setPage] = useState(0)
  const pageCount = Math.max(1, Math.ceil(rows.length / PAGE_SIZE))
  const slice = rows.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)

  return (
    <div>
      <p className="mb-2 text-xs font-medium text-zinc-400">
        Frases planificadas ({rows.length})
      </p>
      <div className="rounded bg-zinc-900 p-2 text-[11px]">
        <table className="w-full">
          <thead className="text-left text-zinc-500">
            <tr>
              <th className="px-1 py-1 w-8">#</th>
              <th className="px-1 py-1 w-16">Inicio</th>
              <th className="px-1 py-1 w-14">Slot</th>
              <th className="px-1 py-1 w-12">Chars</th>
              <th className="px-1 py-1 w-28">Compresión</th>
              <th className="px-1 py-1">Texto</th>
            </tr>
          </thead>
          <tbody>
            {slice.map((r) => (
              <tr
                key={r.idx}
                className={cn('border-t border-zinc-800/50', r.will_overflow && 'bg-red-500/5')}
              >
                <td className="px-1 py-1 font-mono text-zinc-600">{r.idx}</td>
                <td className="px-1 py-1 font-mono text-zinc-500">{fmtMs(r.target_start_ms)}</td>
                <td className="px-1 py-1 font-mono text-zinc-500">{r.allocated_ms}ms</td>
                <td className="px-1 py-1 font-mono text-zinc-500">{r.chars}</td>
                <td className="px-1 py-1">
                  <CompressionBar ratio={r.compression_needed} max={maxComp} />
                </td>
                <td className="px-1 py-1 text-zinc-300">{r.text}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Pager page={page} pageCount={pageCount} onPage={setPage} />
    </div>
  )
}

function GapsList({ gaps }) {
  const [page, setPage] = useState(0)
  if (!gaps?.length) return <p className="text-xs text-emerald-400">Sin gaps entre bloques SRT</p>
  const pageCount = Math.max(1, Math.ceil(gaps.length / PAGE_SIZE))
  const slice = gaps.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)
  const totalBorrowed = gaps.reduce((a, g) => a + (g.borrowed_by_previous || 0), 0)
  return (
    <div>
      <p className="mb-1 text-xs font-medium text-zinc-400">
        Gaps SRT ({gaps.length}) — prestados al TTS: {totalBorrowed}ms
      </p>
      <div className="rounded bg-zinc-900 p-2 text-[11px] space-y-0.5">
        {slice.map((g, i) => (
          <div key={i} className="flex items-center gap-3 text-zinc-500">
            <span className="font-mono">Tras #{g.after_idx}</span>
            <Badge variant="outline" className="text-[10px]">{g.gap_ms}ms</Badge>
            {g.borrowed_by_previous > 0 && (
              <span className="text-emerald-400">→ {g.borrowed_by_previous}ms prestados</span>
            )}
          </div>
        ))}
      </div>
      <Pager page={page} pageCount={pageCount} onPage={setPage} />
    </div>
  )
}

function SynthesisResults({ synthesis, maxComp }) {
  const [page, setPage] = useState(0)
  if (!synthesis) return null
  const rows = synthesis.phrases || []
  const errors = rows.filter((r) => r.error)
  const okRows = rows.filter((r) => !r.error)
  const avgStretch = okRows.length
    ? okRows.reduce((a, r) => a + (r.stretch_ratio || 0), 0) / okRows.length
    : 0
  const pageCount = Math.max(1, Math.ceil(rows.length / PAGE_SIZE))
  const slice = rows.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)

  return (
    <div className="space-y-3 rounded border border-blue-500/30 bg-blue-500/5 p-3">
      <div className="flex items-center gap-2">
        <Zap className="h-4 w-4 text-blue-400" />
        <p className="text-sm font-medium text-blue-300">
          TTS real ejecutado ({rows.length} frases, {errors.length} errores)
        </p>
      </div>
      <div className="flex flex-wrap gap-4 text-[11px] text-zinc-400">
        <span>Stretch medio: <b className="text-zinc-200">{avgStretch.toFixed(2)}x</b></span>
        <span>Overlaps: <b className={cn(synthesis.overlaps?.length ? 'text-amber-400' : 'text-emerald-400')}>
          {synthesis.overlaps?.length || 0}
        </b></span>
      </div>
      <p className="text-[11px] text-zinc-500 break-all">
        Ref voz: <span className="font-mono">{synthesis.ref_wav}</span>
      </p>

      {synthesis.overlaps?.length > 0 && (
        <div>
          <p className="mb-1 text-xs font-medium text-amber-300">Overlaps detectados</p>
          <div className="text-[11px] space-y-0.5">
            {synthesis.overlaps.map((o, i) => (
              <div key={i} className="text-zinc-400">
                Frase #{o.between[0]} → #{o.between[1]}: solape{' '}
                <b className="text-amber-300">{o.overlap_ms}ms</b>
              </div>
            ))}
          </div>
        </div>
      )}

      <div>
        <p className="mb-1 text-xs font-medium text-zinc-400">Resultado por frase</p>
        <div className="rounded bg-zinc-950 p-2 text-[11px]">
          <table className="w-full">
            <thead className="text-left text-zinc-500">
              <tr>
                <th className="px-1 py-1 w-8">#</th>
                <th className="px-1 py-1 w-14">Slot</th>
                <th className="px-1 py-1 w-14">TTS</th>
                <th className="px-1 py-1 w-14">Fitted</th>
                <th className="px-1 py-1 w-14">Speed</th>
                <th className="px-1 py-1 w-28">Stretch</th>
                <th className="px-1 py-1 w-14">+/-</th>
                <th className="px-1 py-1">Texto</th>
              </tr>
            </thead>
            <tbody>
              {slice.map((r) => (
                <tr key={r.idx} className={cn(
                  'border-t border-zinc-800/50',
                  r.error && 'bg-red-500/10',
                  r.overflow_vs_slot_ms > 0 && !r.error && 'bg-amber-500/5',
                )}>
                  <td className="px-1 py-1 font-mono text-zinc-600">{r.idx}</td>
                  {r.error ? (
                    <td className="px-1 py-1 text-red-400" colSpan={7}>{r.error}</td>
                  ) : (
                    <>
                      <td className="px-1 py-1 font-mono text-zinc-500">{r.allocated_ms}</td>
                      <td className="px-1 py-1 font-mono text-zinc-500">{r.raw_tts_ms}</td>
                      <td className="px-1 py-1 font-mono text-zinc-400">{r.fitted_ms}</td>
                      <td className="px-1 py-1 font-mono text-zinc-400">{r.speed_used}</td>
                      <td className="px-1 py-1"><CompressionBar ratio={r.stretch_ratio} max={maxComp} /></td>
                      <td className={cn(
                        'px-1 py-1 font-mono',
                        r.overflow_vs_slot_ms > 0 ? 'text-amber-300' : 'text-emerald-400',
                      )}>
                        {r.overflow_vs_slot_ms > 0 ? '+' : ''}{r.overflow_vs_slot_ms}
                      </td>
                      <td className="px-1 py-1 text-zinc-300">{r.text}</td>
                    </>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <Pager page={page} pageCount={pageCount} onPage={setPage} />
      </div>
    </div>
  )
}

function AnalysisResults({ data }) {
  const maxComp = data?.summary?.config?.max_compression_ratio || 1.25
  const hasSynth = !!data.synthesis

  return (
    <div className="space-y-4">
      <SummaryCards summary={data.summary} />
      <ConfigRow config={data.summary?.config} />

      {data.summary?.will_overflow_count > 0 && (
        <div className="flex items-start gap-2 rounded border border-amber-500/30 bg-amber-500/5 p-3 text-xs text-amber-300">
          <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />
          <div>
            <b>{data.summary.will_overflow_count}</b> frase(s) superan el límite de compresión
            ({maxComp}x). TTS quedará acelerado. Sugerencia: aumenta{' '}
            <code className="text-amber-200">max_overflow_ms</code> o acorta la traducción.
          </div>
        </div>
      )}

      <Tabs defaultValue="planned" className="w-full">
        <TabsList className="bg-zinc-900/80">
          <TabsTrigger value="planned">Plan ({data.planned?.length || 0})</TabsTrigger>
          <TabsTrigger value="gaps">Gaps ({data.srt_gaps?.length || 0})</TabsTrigger>
          {hasSynth && (
            <TabsTrigger value="synth">
              TTS real ({data.synthesis.phrases?.length || 0})
            </TabsTrigger>
          )}
        </TabsList>
        <TabsContent value="planned" className="pt-3">
          <PlannedTable rows={data.planned || []} maxComp={maxComp} />
        </TabsContent>
        <TabsContent value="gaps" className="pt-3">
          <GapsList gaps={data.srt_gaps || []} />
        </TabsContent>
        {hasSynth && (
          <TabsContent value="synth" className="pt-3">
            <SynthesisResults synthesis={data.synthesis} maxComp={maxComp} />
          </TabsContent>
        )}
      </Tabs>
    </div>
  )
}

export default function DubbingAnalysisDialog({ open, onOpenChange, videoPath }) {
  const job = useDubbingAnalysis((s) => s.jobs[videoPath])
  const launch = useDubbingAnalysis((s) => s.launch)
  const clear = useDubbingAnalysis((s) => s.clear)
  const [synthesize, setSynthesize] = useState(false)
  const [maxPhrases, setMaxPhrases] = useState(10)

  const isPending = job?.status === 'pending'
  const data = job?.data

  const runAnalysis = (withSynth) => {
    launch(videoPath, {
      synthesize: withSynth,
      maxPhrases: withSynth ? maxPhrases : null,
    })
    toast.info(
      withSynth
        ? `TTS real de ${maxPhrases} frases lanzado (~varios minutos)`
        : 'Análisis rápido lanzado',
      { duration: 4000 },
    )
  }

  const reAnalyze = () => {
    clear(videoPath)
    runAnalysis(synthesize)
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-6xl w-[95vw] max-h-[92vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Mic className="h-5 w-5 text-purple-400" />
            Debug Doblaje
          </DialogTitle>
          <DialogDescription className="text-xs text-zinc-500 truncate">
            {videoPath}
          </DialogDescription>
        </DialogHeader>

        {!data && !isPending && (
          <div className="flex flex-col items-center gap-4 py-8">
            <p className="text-sm text-zinc-400 text-center max-w-md">
              Analiza el plan de doblaje: bloques SRT, slots, densidad por frase,
              compresión necesaria y overflow. Opcionalmente ejecuta TTS real para
              medir duraciones y detectar solapes.
            </p>
            <div className="flex flex-col gap-2 rounded border border-zinc-800 bg-zinc-900/50 p-3 text-xs">
              <label className="flex items-center gap-2 text-zinc-300 cursor-pointer">
                <input
                  type="checkbox"
                  checked={synthesize}
                  onChange={(e) => setSynthesize(e.target.checked)}
                />
                Sintetizar TTS real (carga XTTS-v2, varios minutos)
              </label>
              {synthesize && (
                <label className="flex items-center gap-2 text-zinc-400 ml-5">
                  Frases a sintetizar:
                  <input
                    type="number"
                    min={1}
                    max={200}
                    value={maxPhrases}
                    onChange={(e) => setMaxPhrases(Number(e.target.value) || 10)}
                    className="w-16 rounded bg-zinc-800 px-2 py-0.5 text-zinc-200"
                  />
                </label>
              )}
            </div>
            <Button onClick={() => runAnalysis(synthesize)}>
              <Mic className="mr-2 h-4 w-4" />
              {synthesize ? 'Analizar + sintetizar' : 'Analizar plan'}
            </Button>
          </div>
        )}

        {isPending && (
          <div className="flex flex-col items-center gap-3 py-12">
            <Loader2 className="h-8 w-8 animate-spin text-purple-400" />
            <p className="text-sm text-zinc-400">
              {job?.synthesize
                ? 'Cargando XTTS-v2 y sintetizando frases...'
                : 'Analizando SRT y plan de slots...'}
            </p>
            <p className="text-xs text-zinc-600">
              Puedes cerrar este diálogo — continúa en segundo plano
            </p>
          </div>
        )}

        {data?.error && (
          <div className="space-y-3">
            <div className="rounded border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-300 whitespace-pre-wrap break-words">
              {data.error}
            </div>
            <div className="flex justify-end">
              <Button size="sm" variant="outline" onClick={reAnalyze}>
                <Mic className="mr-1 h-3 w-3" />
                Reintentar
              </Button>
            </div>
          </div>
        )}

        {data && !data.error && (
          <>
            <AnalysisResults data={data} />
            <div className="flex justify-end gap-3 pt-3 items-center">
              <label className="flex items-center gap-2 text-xs text-zinc-400 cursor-pointer">
                <input
                  type="checkbox"
                  checked={synthesize}
                  onChange={(e) => setSynthesize(e.target.checked)}
                />
                Con TTS real
              </label>
              <Button size="sm" variant="outline" onClick={reAnalyze}>
                <Mic className="mr-1 h-3 w-3" />
                Re-analizar
              </Button>
            </div>
          </>
        )}
      </DialogContent>
    </Dialog>
  )
}

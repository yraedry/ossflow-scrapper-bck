// Debug dialog: deep analysis of a video's audio for subtitle troubleshooting.
// Shows energy map, transcription comparison (raw vs denoised), gaps, and filter stats.
// Analysis runs in background via zustand store — user can close the dialog and come back.
import { useState } from 'react'
import {
  Loader2,
  Activity,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Volume2,
  VolumeX,
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
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/Badge'
import { cn } from '@/lib/utils'
import { toast } from 'sonner'
import useAudioAnalysis from '../stores/useAudioAnalysis'

const PAGE_SIZE = 20

function Pager({ page, pageCount, onPage }) {
  if (pageCount <= 1) return null
  return (
    <div className="flex items-center justify-between pt-1 text-xs">
      <Button
        size="sm"
        variant="ghost"
        disabled={page === 0}
        onClick={() => onPage(Math.max(0, page - 1))}
      >
        <ChevronLeft className="h-3 w-3 mr-1" />
        Anterior
      </Button>
      <span className="text-zinc-500">Página {page + 1} / {pageCount}</span>
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

function fmtTime(sec) {
  if (sec == null || sec < 0) return '—'
  const m = Math.floor(sec / 60)
  const s = (sec % 60).toFixed(1)
  return `${m}:${s.padStart(4, '0')}`
}

// Compact energy timeline: one thin column per second
function EnergyTimeline({ energyMap, segments, label }) {
  if (!energyMap?.length) return null
  const maxSec = energyMap[energyMap.length - 1]?.sec || 0

  const coveredSecs = new Set()
  for (const seg of segments || []) {
    const s = Math.floor(seg.start)
    const e = Math.ceil(seg.end)
    for (let i = s; i <= e; i++) coveredSecs.add(i)
  }

  return (
    <div>
      <p className="mb-1 text-xs font-medium text-zinc-400">{label}</p>
      <div className="flex gap-px overflow-x-auto rounded bg-zinc-900 p-1" style={{ maxHeight: 48 }}>
        {energyMap.map((e) => {
          const pct = Math.max(2, Math.min(100, ((e.rms_db + 60) / 60) * 100))
          const hasSeg = coveredSecs.has(e.sec)
          const color = hasSeg ? 'bg-emerald-500' : e.rms_db > -40 ? 'bg-amber-500' : 'bg-zinc-700'
          return (
            <div
              key={e.sec}
              title={`${e.sec}s: ${e.rms_db} dB${hasSeg ? ' (cubierto)' : ''}`}
              className={cn('w-1 shrink-0 rounded-sm', color)}
              style={{ height: `${pct}%`, minHeight: 2, alignSelf: 'flex-end' }}
            />
          )
        })}
      </div>
      <div className="mt-0.5 flex justify-between text-[10px] text-zinc-600">
        <span>0s</span>
        <span>{fmtTime(maxSec)}</span>
      </div>
    </div>
  )
}

function GapRow({ gap }) {
  return (
    <div className="flex items-center gap-3 rounded border border-zinc-800 bg-zinc-900/50 px-3 py-2 text-sm">
      <span className="font-mono text-zinc-400">
        {fmtTime(gap.start)} → {fmtTime(gap.end)}
      </span>
      <Badge variant="outline" className="text-xs">
        {gap.duration?.toFixed(1)}s
      </Badge>
      {gap.energy_db != null && (
        <span className="flex items-center gap-1 text-xs text-zinc-500">
          {gap.likely_speech ? (
            <Volume2 className="h-3 w-3 text-amber-400" />
          ) : (
            <VolumeX className="h-3 w-3 text-zinc-600" />
          )}
          {gap.energy_db} dB
        </span>
      )}
      {gap.likely_speech && (
        <Badge className="bg-amber-500/15 text-amber-300 border-amber-500/30 text-[10px]">
          Posible voz perdida
        </Badge>
      )}
    </div>
  )
}

function SegmentList({ segments, title }) {
  const [page, setPage] = useState(0)
  if (!segments?.length) return <p className="text-xs text-zinc-600">Sin segmentos</p>
  const pageCount = Math.max(1, Math.ceil(segments.length / PAGE_SIZE))
  const slice = segments.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)

  return (
    <div>
      <p className="mb-1 text-xs font-medium text-zinc-400">
        {title} ({segments.length})
      </p>
      <div className="space-y-0.5 rounded bg-zinc-900 p-2 text-xs">
        {slice.map((s, i) => (
          <div key={i} className="flex gap-2">
            <span className="shrink-0 font-mono text-zinc-600 w-24">
              {fmtTime(s.start)}–{fmtTime(s.end)}
            </span>
            <span className="text-zinc-300">{s.text}</span>
          </div>
        ))}
      </div>
      <Pager page={page} pageCount={pageCount} onPage={setPage} />
    </div>
  )
}

function DroppedList({ items }) {
  const [page, setPage] = useState(0)
  if (!items?.length) return null
  const pageCount = Math.max(1, Math.ceil(items.length / PAGE_SIZE))
  const slice = items.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE)
  return (
    <div>
      <p className="mb-2 text-sm font-medium text-orange-300">
        Segmentos descartados por filtros ({items.length})
      </p>
      <div className="space-y-0.5 rounded bg-zinc-900 p-2 text-xs">
        {slice.map((s, i) => (
          <div key={i} className="flex gap-2 border-b border-zinc-800/60 pb-0.5">
            <span className="w-24 shrink-0 font-mono text-zinc-500">
              {fmtTime(s.start)}–{fmtTime(s.end)}
            </span>
            <Badge variant="outline" className="h-4 shrink-0 border-orange-500/40 px-1 text-[9px] text-orange-400">
              {s.reason}
            </Badge>
            <span className="text-zinc-400">{s.text}</span>
          </div>
        ))}
      </div>
      <Pager page={page} pageCount={pageCount} onPage={setPage} />
    </div>
  )
}

function FilterStats({ stats }) {
  if (!stats) return null
  const entries = Object.entries(stats).filter(([, v]) => v > 0)
  if (entries.length === 0)
    return <p className="text-xs text-emerald-400">Ningún segmento filtrado</p>

  return (
    <div className="flex flex-wrap gap-2">
      {entries.map(([key, val]) => (
        <Badge key={key} variant="outline" className="text-[10px] border-red-500/30 text-red-300">
          {key}: {val}
        </Badge>
      ))}
    </div>
  )
}

function AnalysisResults({ data }) {
  const improvement = data?.transcription?.improvement
  const hasGapsWithSpeech = data?.gaps?.some((g) => g.likely_speech)
  const hasSrtGapsWithSpeech = data?.gaps_srt?.some((g) => g.likely_speech)
  const hasSrtData = data?.segments_srt?.length > 0

  return (
    <div className="space-y-6">
      {/* Summary cards */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <div className="rounded border border-zinc-800 bg-zinc-900/50 p-3 text-center">
          <p className="text-lg font-bold text-zinc-100">{fmtTime(data.duration_seconds)}</p>
          <p className="text-[10px] text-zinc-500">Duración</p>
        </div>
        <div className="rounded border border-zinc-800 bg-zinc-900/50 p-3 text-center">
          <p className="text-lg font-bold text-zinc-100">{data.transcription?.raw_segments}</p>
          <p className="text-[10px] text-zinc-500">Segmentos (original)</p>
        </div>
        <div className="rounded border border-zinc-800 bg-zinc-900/50 p-3 text-center">
          <p className={cn('text-lg font-bold', improvement > 0 ? 'text-emerald-400' : 'text-zinc-100')}>
            {data.transcription?.denoised_segments}
            {improvement > 0 && <span className="ml-1 text-xs text-emerald-500">+{improvement}</span>}
          </p>
          <p className="text-[10px] text-zinc-500">Segmentos (denoised)</p>
        </div>
        <div className="rounded border border-zinc-800 bg-zinc-900/50 p-3 text-center">
          <p className={cn('text-lg font-bold', data.hallucination_filter?.dropped > 0 ? 'text-amber-400' : 'text-zinc-100')}>
            {data.hallucination_filter?.dropped || 0}
          </p>
          <p className="text-[10px] text-zinc-500">Filtrados</p>
        </div>
      </div>

      {/* VAD params */}
      <div className="flex gap-4 text-xs text-zinc-500">
        <span>VAD onset: <b className="text-zinc-300">{data.vad_params?.onset}</b></span>
        <span>VAD offset: <b className="text-zinc-300">{data.vad_params?.offset}</b></span>
        <span>Denoise: {data.has_denoise ? (
          <CheckCircle className="inline h-3 w-3 text-emerald-400" />
        ) : (
          <XCircle className="inline h-3 w-3 text-red-400" />
        )}</span>
      </div>

      {/* Energy timelines */}
      <EnergyTimeline energyMap={data.energy_map} segments={data.segments_raw} label="Energía original + cobertura transcripción" />
      {data.has_denoise && (
        <EnergyTimeline energyMap={data.energy_map_denoised} segments={data.segments_denoised} label="Energía denoised + cobertura transcripción" />
      )}
      {hasSrtData && (
        <EnergyTimeline energyMap={data.energy_map_denoised || data.energy_map} segments={data.segments_srt} label="Cobertura SRT en disco (ground truth)" />
      )}
      {!hasSrtData && data.segments_filtered?.length > 0 && (
        <EnergyTimeline energyMap={data.energy_map_denoised || data.energy_map} segments={data.segments_filtered} label="Cobertura SRT final (post-filtrado, sin SRT en disco)" />
      )}
      <div className="flex gap-3 text-[10px] text-zinc-600">
        <span className="flex items-center gap-1"><span className="inline-block h-2 w-2 rounded-sm bg-emerald-500" /> Cubierto</span>
        <span className="flex items-center gap-1"><span className="inline-block h-2 w-2 rounded-sm bg-amber-500" /> Voz sin cubrir</span>
        <span className="flex items-center gap-1"><span className="inline-block h-2 w-2 rounded-sm bg-zinc-700" /> Silencio</span>
      </div>

      {/* Gaps en SRT en disco (ground truth — más fiable) */}
      {hasSrtData && (
        <div>
          {data.gaps_srt?.length > 0 ? (
            <>
              <p className="mb-2 flex items-center gap-2 text-sm font-medium text-red-300">
                <AlertTriangle className="h-4 w-4 text-red-400" />
                Gaps en SRT en disco ({data.gaps_srt.length})
                {hasSrtGapsWithSpeech && (
                  <Badge className="bg-red-500/15 text-red-300 text-[10px]">Voz perdida</Badge>
                )}
              </p>
              <div className="space-y-1">
                {data.gaps_srt.map((g, i) => <GapRow key={i} gap={g} />)}
              </div>
            </>
          ) : (
            <p className="flex items-center gap-2 text-sm text-emerald-400">
              <CheckCircle className="h-4 w-4" />
              Sin gaps en SRT en disco ({data.segments_srt.length} bloques)
            </p>
          )}
        </div>
      )}

      {/* Gaps post-filtrado (pipeline simulado, solo si no hay SRT en disco) */}
      {!hasSrtData && data.gaps_filtered?.length > 0 && (
        <div>
          <p className="mb-2 flex items-center gap-2 text-sm font-medium text-red-300">
            <AlertTriangle className="h-4 w-4 text-red-400" />
            Gaps post-filtrado ({data.gaps_filtered.length})
            {data.gaps_filtered.some((g) => g.likely_speech) && (
              <Badge className="bg-red-500/15 text-red-300 text-[10px]">Voz perdida en SRT</Badge>
            )}
          </p>
          <div className="space-y-1">
            {data.gaps_filtered.map((g, i) => <GapRow key={i} gap={g} />)}
          </div>
        </div>
      )}

      {/* Gaps en transcripción (pre-filtrado) */}
      {data.gaps?.length > 0 && (
        <div>
          <p className="mb-2 flex items-center gap-2 text-sm font-medium text-zinc-300">
            <AlertTriangle className="h-4 w-4 text-amber-400" />
            Gaps en transcripción ({data.gaps.length})
            {hasGapsWithSpeech && (
              <Badge className="bg-amber-500/15 text-amber-300 text-[10px]">Posible voz</Badge>
            )}
          </p>
          <div className="space-y-1">
            {data.gaps.map((g, i) => <GapRow key={i} gap={g} />)}
          </div>
        </div>
      )}

      {/* Hallucination filter stats */}
      <div>
        <p className="mb-1 text-sm font-medium text-zinc-300">Filtros de alucinación</p>
        <FilterStats stats={data.hallucination_filter?.stats} />
      </div>

      {/* Segmentos descartados por filtros */}
      <DroppedList items={data.segments_dropped} />

      {/* Segment lists */}
      <div className="grid gap-4 lg:grid-cols-2">
        <SegmentList segments={data.segments_raw} title="Segmentos (audio original)" />
        <SegmentList segments={data.segments_denoised} title="Segmentos (audio denoised)" />
      </div>
      {(data.segments_aligned || data.segments_filtered) && (
        <div className="grid gap-4 lg:grid-cols-2">
          <SegmentList segments={data.segments_aligned} title="Segmentos (alineados)" />
          <SegmentList segments={data.segments_filtered} title="Segmentos (post-filtrado)" />
        </div>
      )}
      {hasSrtData && (
        <SegmentList segments={data.segments_srt} title={`Bloques SRT en disco (${data.segments_srt.length})`} />
      )}
    </div>
  )
}

export default function AudioAnalysisDialog({ open, onOpenChange, videoPath }) {
  const job = useAudioAnalysis((s) => s.jobs[videoPath])
  const launch = useAudioAnalysis((s) => s.launch)
  const clear = useAudioAnalysis((s) => s.clear)

  const isPending = job?.status === 'pending'
  const data = job?.data

  const runAnalysis = () => {
    launch(videoPath)
    toast.info('Análisis lanzado en segundo plano', {
      description: 'Puedes cerrar este diálogo y seguir navegando.',
      duration: 4000,
    })
  }

  const reAnalyze = () => {
    clear(videoPath)
    launch(videoPath)
    toast.info('Re-análisis lanzado en segundo plano', {
      description: 'Puedes cerrar este diálogo y seguir navegando.',
      duration: 4000,
    })
  }

  const handleOpenChange = (v) => {
    onOpenChange(v)
  }

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="max-w-6xl w-[95vw] max-h-[92vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Activity className="h-5 w-5 text-blue-400" />
            Análisis de Audio (Debug)
          </DialogTitle>
          <DialogDescription className="text-xs text-zinc-500 truncate">
            {videoPath}
          </DialogDescription>
        </DialogHeader>

        {!data && !isPending && (
          <div className="flex flex-col items-center gap-4 py-8">
            <p className="text-sm text-zinc-400 text-center">
              Analiza el audio para diagnosticar por qué faltan subtítulos.
              <br />
              <span className="text-zinc-600">
                Compara audio original vs denoised, detecta gaps con energía, y muestra qué filtros descartan segmentos.
              </span>
            </p>
            <Button onClick={runAnalysis}>
              <Activity className="mr-2 h-4 w-4" />
              Lanzar análisis
            </Button>
          </div>
        )}

        {isPending && (
          <div className="flex flex-col items-center gap-3 py-12">
            <Loader2 className="h-8 w-8 animate-spin text-blue-400" />
            <p className="text-sm text-zinc-400">
              Analizando audio (transcribiendo original + denoised)...
            </p>
            <p className="text-xs text-zinc-600">
              Puedes cerrar este diálogo — el análisis continúa en segundo plano
            </p>
          </div>
        )}

        {data?.error && (
          <div className="space-y-3">
            <div className="rounded border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-300">
              {data.error}
            </div>
            <div className="flex justify-end">
              <Button size="sm" variant="outline" onClick={reAnalyze}>
                <Activity className="mr-1 h-3 w-3" />
                Reintentar
              </Button>
            </div>
          </div>
        )}

        {data && !data.error && (
          <>
            <AnalysisResults data={data} />
            <div className="flex justify-end">
              <Button size="sm" variant="outline" onClick={reAnalyze}>
                <Activity className="mr-1 h-3 w-3" />
                Re-analizar
              </Button>
            </div>
          </>
        )}
      </DialogContent>
    </Dialog>
  )
}

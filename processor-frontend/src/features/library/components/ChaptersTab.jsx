// Chapters grouped by Season with inline rename + per-chapter process action.
// Reuses `useRenameChapter` (optimistic) from the library feature.
import { useMemo, useRef, useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  Film,
  Captions,
  Mic,
  Pencil,
  Check,
  X,
  Play,
  Eye,
  Loader2,
  ShieldCheck,
  Scissors,
  Languages,
  RotateCw,
} from 'lucide-react'
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/Badge'
import { cn } from '@/lib/utils'
import { toast } from 'sonner'
import { useRenameChapter, useRenameByOracle } from '../api/useLibrary'
import { useStartPipeline } from '@/features/pipeline/api/usePipeline'
import { useStartElevenLabsDubbing, useElevenLabsJob, useStartElevenLabsBatch } from '@/features/elevenlabs/api/useElevenLabsDubbing'
import { useOracleData } from '@/features/oracle/api/useOracle'
import SubtitleValidationDialog from './SubtitleValidationDialog'
import SeasonValidationDialog from './SeasonValidationDialog'
import DubQaBadge from './DubQaBadge'
import VideoReviewDialog from '@/components/media/VideoReviewDialog'

function srtPathFor(videoPath) {
  if (!videoPath) return null
  const dot = videoPath.lastIndexOf('.')
  return dot > 0 ? `${videoPath.slice(0, dot)}.en.srt` : `${videoPath}.en.srt`
}

// `{prefix} - SNNeMM - {title}{ext}` → just the title
const SNNEMM_RE = /^(.*?)\s*-\s*S\d{2}E\d{2,3}\s*-\s*(.*)\.[^.]+$/
function deriveTitle(filename) {
  const m = SNNEMM_RE.exec(filename || '')
  return m ? m[2] : filename || ''
}
function seasonEpisodeCode(filename) {
  const m = /S(\d{2})E(\d{2,3})/.exec(filename || '')
  return m ? `S${m[1]}E${m[2]}` : '—'
}
function fmtDuration(sec) {
  if (!sec || sec < 0) return '—'
  const s = Math.round(sec)
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  const r = s % 60
  if (m < 60) return r ? `${m}m ${r}s` : `${m}m`
  const h = Math.floor(m / 60)
  return `${h}h ${m % 60}m`
}

function StatusBadge({ ok, Icon, label }) {
  return (
    <span
      title={label}
      className={cn(
        'inline-flex h-6 w-6 items-center justify-center rounded',
        ok ? 'bg-emerald-500/15 text-emerald-300' : 'bg-zinc-800/50 text-zinc-600',
      )}
    >
      <Icon className="h-3 w-3" />
    </span>
  )
}

function ChapterRow({ video, instructionalName, onNext, hasOracle }) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState('')
  const [err, setErr] = useState(null)
  const [validateOpen, setValidateOpen] = useState(false)
  const [playerOpen, setPlayerOpen] = useState(false)
  const inputRef = useRef(null)
  const nav = useNavigate()
  const rename = useRenameChapter()
  const startSplit = useStartPipeline()
  const startElevenLabs = useStartElevenLabsDubbing()
  const [elevenLabsJobId, setElevenLabsJobId] = useState(null)
  const elevenLabsJob = useElevenLabsJob(elevenLabsJobId)

  const handleElevenLabs = async () => {
    if (startElevenLabs.isPending || elevenLabsJobId) return
    const confirmMsg =
      `Enviar "${video.filename}" a ElevenLabs Dubbing Studio?\n\n` +
      `• Consume créditos de tu plan\n` +
      `• Con marca de agua (33% menos créditos)\n` +
      `• Resultado en <Season>/elevenlabs/ con el mismo nombre\n` +
      `• Sigue el progreso en la sección ElevenLabs de la barra lateral`
    if (!window.confirm(confirmMsg)) return
    const toastId = toast.loading('Enviando a ElevenLabs…')
    try {
      const resp = await startElevenLabs.mutateAsync({ path: video.path })
      const jid = resp?.job_id
      setElevenLabsJobId(jid)
      toast.success(`Job ${jid} iniciado`, {
        id: toastId,
        duration: 5000,
        action: {
          label: 'Ver progreso',
          onClick: () => nav('/elevenlabs'),
        },
      })
    } catch (e) {
      const msg = e?.body?.detail || e?.message || 'Error'
      toast.error(`ElevenLabs falló: ${msg}`, { id: toastId })
    }
  }

  // Terminal transitions: fire a loud toast and clear the inline badge.
  // The persistent view lives at /elevenlabs, so we don't need to keep
  // the pill visible after completion.
  useEffect(() => {
    if (!elevenLabsJob.data) return
    const { status, message, result } = elevenLabsJob.data
    if (status === 'completed') {
      const out = result?.output_filename || result?.output_path?.split(/[\\\/]/).pop() || 'ok'
      toast.success(`ElevenLabs listo: ${out}`, { duration: 10000 })
      setElevenLabsJobId(null)
    } else if (status === 'failed') {
      toast.error(`ElevenLabs falló: ${message || 'error'}`, { duration: 15000 })
      setElevenLabsJobId(null)
    }
  }, [elevenLabsJob.data?.status, nav])

  useEffect(() => {
    if (editing) {
      setDraft(deriveTitle(video.filename))
      setErr(null)
      setTimeout(() => inputRef.current?.focus(), 0)
    }
  }, [editing, video.filename])

  const hasSubs = Boolean(video.has_subtitles_en)
  const hasSubsEs = Boolean(video.has_subtitles_es)
  const hasDub = Boolean(video.has_dubbing || video.has_dubbed)
  // "Ya troceado" = pertenece a una Season válida o tiene código SNNEMM en el nombre.
  const isChaptered =
    Boolean(video.season && video.season !== 'Sin temporada') ||
    /S\d{2}E\d{2,3}/i.test(video.filename || '')

  const missing = []
  if (!hasSubs) missing.push('subtitles')
  if (!hasSubsEs) missing.push('translate')
  if (!hasDub) missing.push('dubbing')

  const commit = async () => {
    const newTitle = draft.trim()
    if (!newTitle) {
      setErr('Vacío')
      return
    }
    try {
      await rename.mutateAsync({
        oldPath: video.path,
        newTitle,
        instructionalName,
      })
      toast.success('Capítulo renombrado')
      setEditing(false)
    } catch (e) {
      setErr(e.message || 'Error')
    }
  }

  const onKeyDown = (e) => {
    if (e.key === 'Enter') {
      e.preventDefault()
      commit()
    } else if (e.key === 'Escape') {
      e.preventDefault()
      setEditing(false)
    } else if (e.key === 'Tab') {
      e.preventDefault()
      commit().finally(() => onNext?.())
    }
  }

  const [processOpen, setProcessOpen] = useState(false)
  const [selectedSteps, setSelectedSteps] = useState([])
  const [forceRegen, setForceRegen] = useState(false)

  const openProcessMenu = () => {
    setSelectedSteps([...missing])
    setForceRegen(false)
    setProcessOpen(true)
  }

  const toggleStep = (step) => {
    setSelectedSteps((prev) =>
      prev.includes(step) ? prev.filter((s) => s !== step) : [...prev, step],
    )
  }

  const launchProcess = async () => {
    setProcessOpen(false)
    const order = ['chapters', 'subtitles', 'translate', 'dubbing']
    const sorted = order.filter((s) => selectedSteps.includes(s))
    if (!sorted.length) return

    const launchingId = toast.loading('Lanzando pipeline…')
    try {
      const opts = forceRegen ? { force: true } : {}
      if (hasOracle && sorted.includes('chapters')) opts.mode = 'oracle'
      const resp = await startSplit.mutateAsync({ path: video.path, steps: sorted, options: opts })
      const id = resp?.pipeline_id || resp?.id
      toast.success('Pipeline lanzado', { id: launchingId })
      if (id) nav(`/pipelines/${id}`)
    } catch (e) {
      toast.error(`Error: ${e?.message || 'desconocido'}`, { id: launchingId })
    }
  }

  const handleSplit = async () => {
    try {
      const opts = hasOracle ? { mode: 'oracle' } : {}
      const resp = await startSplit.mutateAsync({
        path: video.path,
        steps: ['chapters'],
        options: opts,
      })
      const id = resp?.pipeline_id || resp?.id
      toast.success('Troceado lanzado')
      if (id) nav(`/pipelines/${id}`)
    } catch (e) {
      toast.error(`Error: ${e?.message || 'desconocido'}`)
    }
  }

  return (
    <tr className="border-t border-zinc-800/60 hover:bg-zinc-900/40">
      <td className="px-3 py-2 font-mono text-xs text-zinc-500 shrink-0 whitespace-nowrap">
        {seasonEpisodeCode(video.filename)}
      </td>
      <td className="px-3 py-2 max-w-xs">
        {editing ? (
          <div className="flex items-center gap-2">
            <Input
              ref={inputRef}
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={onKeyDown}
              disabled={rename.isPending}
              className="h-8 text-sm"
            />
            <button
              type="button"
              aria-label="Confirmar"
              onClick={commit}
              disabled={rename.isPending}
              className="text-emerald-400 hover:text-emerald-300"
            >
              {rename.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Check className="h-4 w-4" />
              )}
            </button>
            <button
              type="button"
              aria-label="Cancelar"
              onClick={() => setEditing(false)}
              disabled={rename.isPending}
              className="text-zinc-500 hover:text-zinc-300"
            >
              <X className="h-4 w-4" />
            </button>
            {err && <span className="text-[11px] text-red-400">{err}</span>}
          </div>
        ) : (
          <div className="group flex min-w-0 items-center gap-2">
            <span
              className={cn(
                'truncate text-sm text-zinc-100',
                video._optimistic && 'opacity-70',
              )}
              title={video.filename}
            >
              {video.filename}
            </span>
            <button
              type="button"
              onClick={() => setEditing(true)}
              aria-label="Renombrar"
              className="opacity-0 group-hover:opacity-100 text-zinc-500 hover:text-amber-400"
            >
              <Pencil className="h-3.5 w-3.5" />
            </button>
          </div>
        )}
      </td>
      <td className="px-3 py-2 text-xs tabular-nums text-zinc-500 shrink-0 whitespace-nowrap">
        {fmtDuration(video.duration)}
      </td>
      <td className="px-3 py-2">
        <div className="flex items-center gap-1.5">
          {isChaptered ? (
            <StatusBadge ok Icon={Scissors} label="Ya troceado" />
          ) : (
            <button
              type="button"
              onClick={handleSplit}
              disabled={startSplit.isPending}
              title="Trocear en capítulos"
              className={cn(
                'inline-flex h-6 w-6 items-center justify-center rounded',
                'bg-amber-500/15 text-amber-400 hover:bg-amber-500/25 disabled:opacity-50',
              )}
            >
              {startSplit.isPending ? (
                <Loader2 className="h-3 w-3 animate-spin" />
              ) : (
                <Scissors className="h-3 w-3" />
              )}
            </button>
          )}
          <StatusBadge ok={hasSubs} Icon={Captions} label="Subs EN" />
          <StatusBadge ok={hasSubsEs} Icon={Languages} label="Subs ES" />
          <StatusBadge ok={hasDub} Icon={Mic} label="Doblaje ES" />
          <DubQaBadge videoPath={video.path} enabled={hasDub} />
        </div>
      </td>
      <td className="px-3 py-2 text-right shrink-0 whitespace-nowrap">
        <div className="flex items-center justify-end gap-1">
          <Button
            size="sm"
            variant="ghost"
            onClick={() => setPlayerOpen(true)}
            title="Ver capítulo"
          >
            <Eye className="mr-1 h-3 w-3" />
            Ver
          </Button>
          <Button
            size="sm"
            variant="ghost"
            onClick={() => setValidateOpen(true)}
            disabled={!hasSubs}
            title={hasSubs ? 'Validar calidad de los subtítulos (detectar alucinaciones)' : 'Sin subtítulos EN que validar'}
          >
            <ShieldCheck className="mr-1 h-3 w-3" />
            Validar
          </Button>
          <Button
            size="sm"
            variant="ghost"
            type="button"
            onClick={elevenLabsJobId ? () => nav('/elevenlabs') : handleElevenLabs}
            disabled={startElevenLabs.isPending}
            title={
              elevenLabsJobId
                ? 'Job en curso — click para ir a la página de ElevenLabs'
                : 'Doblar con ElevenLabs Dubbing Studio (consume créditos del plan)'
            }
            className="text-violet-400 hover:text-violet-300"
          >
            {startElevenLabs.isPending || elevenLabsJobId ? (
              <Loader2 className="mr-1 h-3 w-3 animate-spin" />
            ) : (
              <Mic className="mr-1 h-3 w-3" />
            )}
            {elevenLabsJobId
              ? `ElevenLabs ${elevenLabsJob.data?.progress ?? 0}%`
              : 'ElevenLabs'}
          </Button>
          <DropdownMenu open={processOpen} onOpenChange={setProcessOpen}>
            <DropdownMenuTrigger asChild>
              <Button size="sm" variant="ghost" onClick={openProcessMenu}>
                <Play className="mr-1 h-3 w-3" />
                Procesar
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-52 p-2">
              <p className="mb-2 text-xs font-medium text-zinc-400">Pasos a ejecutar</p>
              {[
                { key: 'chapters', label: 'Trocear capítulos', Icon: Scissors, done: isChaptered },
                { key: 'subtitles', label: 'Subtítulos EN', Icon: Captions, done: hasSubs },
                { key: 'translate', label: 'Traducir a ES', Icon: Languages, done: hasSubsEs },
                { key: 'dubbing', label: 'Doblaje', Icon: Mic, done: hasDub },
              ].map(({ key, label, Icon: StepIcon, done }) => {
                const locked = done && !forceRegen
                return (
                <button
                  key={key}
                  type="button"
                  disabled={locked}
                  onClick={(e) => { e.preventDefault(); toggleStep(key) }}
                  className={cn(
                    'flex w-full items-center gap-2 rounded px-2 py-1.5 text-sm transition-colors',
                    locked
                      ? 'cursor-not-allowed text-zinc-600 line-through'
                      : 'cursor-pointer hover:bg-zinc-800 text-zinc-200',
                  )}
                >
                  <span
                    className={cn(
                      'flex h-4 w-4 shrink-0 items-center justify-center rounded border',
                      locked
                        ? 'border-zinc-700 bg-zinc-800'
                        : selectedSteps.includes(key)
                          ? 'border-emerald-500 bg-emerald-500/20'
                          : 'border-zinc-600',
                    )}
                  >
                    {(locked || selectedSteps.includes(key)) && (
                      <Check className="h-3 w-3 text-emerald-400" />
                    )}
                  </span>
                  <StepIcon className="h-3.5 w-3.5" />
                  {label}
                </button>
                )
              })}
              <div className="mt-2 border-t border-zinc-800 pt-2 space-y-0.5">
                <button
                  type="button"
                  onClick={(e) => { e.preventDefault(); setForceRegen((v) => !v) }}
                  className="flex w-full items-center gap-2 rounded px-2 py-1.5 text-sm cursor-pointer hover:bg-zinc-800 text-zinc-400"
                >
                  <span
                    className={cn(
                      'flex h-4 w-4 shrink-0 items-center justify-center rounded border',
                      forceRegen ? 'border-amber-500 bg-amber-500/20' : 'border-zinc-600',
                    )}
                  >
                    {forceRegen && <Check className="h-3 w-3 text-amber-400" />}
                  </span>
                  <RotateCw className="h-3.5 w-3.5" />
                  Forzar regeneración
                </button>
              </div>
              <Button
                size="sm"
                className="mt-2 w-full"
                disabled={selectedSteps.length === 0 || startSplit.isPending}
                onClick={launchProcess}
              >
                {startSplit.isPending ? (
                  <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                ) : (
                  <Play className="mr-1 h-3 w-3" />
                )}
                Lanzar {selectedSteps.length ? `(${selectedSteps.length})` : ''}
              </Button>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
        {hasSubs && validateOpen && (
          <SubtitleValidationDialog
            open={validateOpen}
            onOpenChange={setValidateOpen}
            srtPath={srtPathFor(video.path)}
            videoPath={video.path}
          />
        )}
        {playerOpen && (
          <VideoReviewDialog
            open={playerOpen}
            onOpenChange={setPlayerOpen}
            videoPath={video.path}
            title={video.filename}
            hasSubsEn={hasSubs}
            // Fix: antes pasábamos `hasDub` como `hasSubsEs`, lo que hacía que el
            // player intentara cargar pistas .es.srt cuando había dubbing en lugar
            // de cuando realmente había subtítulos ES → track 404 en consola.
            hasSubsEs={hasSubsEs}
          />
        )}
      </td>
    </tr>
  )
}

function deriveSeasonPath(list) {
  const first = list?.[0]?.path
  if (!first) return null
  const sep = first.includes('\\') ? '\\' : '/'
  const idx = first.lastIndexOf(sep)
  return idx > 0 ? first.slice(0, idx) : null
}

function SeasonPipelineButton({ seasonPath, steps, label, Icon, title, hasOracle, extraOptions }) {
  const nav = useNavigate()
  const start = useStartPipeline()
  const onClick = async (e) => {
    e.stopPropagation()
    if (!seasonPath) {
      toast.error('No se pudo inferir la ruta de la Season')
      return
    }
    const tid = toast.loading(`Lanzando ${label}…`)
    try {
      const opts = { ...(extraOptions || {}) }
      if (hasOracle && steps.includes('chapters')) opts.mode = 'oracle'
      const resp = await start.mutateAsync({ path: seasonPath, steps, options: opts })
      const id = resp?.pipeline_id || resp?.id
      toast.success('Pipeline lanzado', { id: tid })
      if (id) nav(`/pipelines/${id}`)
    } catch (err) {
      toast.error(`Error: ${err?.message || 'desconocido'}`, { id: tid })
    }
  }
  return (
    <Button
      size="sm"
      variant="outline"
      disabled={start.isPending || !seasonPath}
      onClick={onClick}
      title={title}
    >
      {start.isPending ? (
        <Loader2 className="mr-1 h-3 w-3 animate-spin" />
      ) : (
        <Icon className="mr-1 h-3 w-3" />
      )}
      {label}
    </Button>
  )
}

function SeasonValidateButton({ season, list }) {
  const [open, setOpen] = useState(false)
  const hasSubs = list.some((v) => v.has_subtitles_en)
  return (
    <>
      <Button
        size="sm"
        variant="outline"
        disabled={!hasSubs}
        onClick={(e) => { e.stopPropagation(); setOpen(true) }}
        title={hasSubs ? 'Validar subtítulos EN de toda la Season' : 'Sin subtítulos EN que validar'}
      >
        <ShieldCheck className="mr-1 h-3 w-3" />
        Validar
      </Button>
      <SeasonValidationDialog
        open={open}
        onOpenChange={setOpen}
        season={season}
        videos={list}
      />
    </>
  )
}

function SeasonRenameOracleButton({ seasonPath, oracle, instructionalName }) {
  const rename = useRenameByOracle()
  const onClick = async (e) => {
    e.stopPropagation()
    if (!seasonPath) {
      toast.error('No se pudo inferir la ruta de la Season')
      return
    }
    try {
      const result = await rename.mutateAsync({ seasonPath, oracle, instructionalName })
      const n = result?.renamed?.length ?? 0
      const sk = result?.skipped?.length ?? 0
      toast.success(`Renombrados ${n} capítulo(s)${sk ? ` · ${sk} sin coincidencia` : ''}`)
    } catch (err) {
      toast.error(`Renombrado falló: ${err?.message || 'desconocido'}`)
    }
  }
  return (
    <Button
      size="sm"
      variant="outline"
      disabled={rename.isPending || !seasonPath}
      onClick={onClick}
      title="Renombrar capítulos usando los títulos del oráculo"
    >
      {rename.isPending ? (
        <Loader2 className="mr-1 h-3 w-3 animate-spin" />
      ) : (
        <Pencil className="mr-1 h-3 w-3" />
      )}
      Renombrar por Oracle
    </Button>
  )
}

function SeasonElevenLabsButton({ seasonPath, list }) {
  const nav = useNavigate()
  const batch = useStartElevenLabsBatch()
  // Rough filter so the confirm dialog shows a meaningful count. The
  // server still recomputes against the actual folder so this doesn't
  // need to be exact.
  const candidates = (list || []).filter(
    (v) => !v.filename?.toLowerCase().endsWith('_doblado.mp4')
           && !v.filename?.toLowerCase().endsWith('_doblado.mkv'),
  )
  const onClick = async (e) => {
    e.stopPropagation()
    if (!seasonPath) {
      toast.error('No se pudo inferir la ruta de la Season')
      return
    }
    const msg =
      `Enviar ${candidates.length} capítulo(s) de esta Season a ElevenLabs?\n\n` +
      `• Se procesan en serie (1 a la vez)\n` +
      `• Capítulos ya presentes en <Season>/elevenlabs/ se omiten\n` +
      `• Sigue el progreso en el apartado ElevenLabs`
    if (!window.confirm(msg)) return
    try {
      const resp = await batch.mutateAsync({ seasonPath })
      const q = resp?.queued_count ?? 0
      const s = resp?.skipped_count ?? 0
      toast.success(
        `Encolados ${q}${s ? ` · ${s} omitidos` : ''}`,
        {
          action: { label: 'Ver progreso', onClick: () => nav('/elevenlabs') },
          duration: 6000,
        },
      )
    } catch (err) {
      toast.error(`Batch falló: ${err?.body?.detail || err?.message || 'error'}`)
    }
  }
  return (
    <Button
      size="sm"
      variant="outline"
      disabled={batch.isPending || !seasonPath || candidates.length === 0}
      onClick={onClick}
      title="Doblar toda la Season con ElevenLabs (serial, omite ya doblados)"
      className="text-violet-400 hover:text-violet-300"
    >
      {batch.isPending ? (
        <Loader2 className="mr-1 h-3 w-3 animate-spin" />
      ) : (
        <Mic className="mr-1 h-3 w-3" />
      )}
      ElevenLabs Season
    </Button>
  )
}

function SeasonProcessButton({ seasonPath, list, hasOracle, oracleData }) {
  const nav = useNavigate()
  const start = useStartPipeline()
  const [open, setOpen] = useState(false)
  const [selectedSteps, setSelectedSteps] = useState([])
  const [forceRegen, setForceRegen] = useState(false)

  const isChaptered = list.every(
    (v) =>
      Boolean(v.season && v.season !== 'Sin temporada') ||
      /S\d{2}E\d{2,3}/i.test(v.filename || ''),
  )
  const allHaveSubs = list.every((v) => Boolean(v.has_subtitles_en))
  const allHaveSubsEs = list.every((v) => Boolean(v.has_subtitles_es))
  const allHaveDub = list.every((v) => Boolean(v.has_dubbing || v.has_dubbed))

  const openMenu = () => {
    const missing = []
    if (!isChaptered) missing.push('chapters')
    if (!allHaveSubs) missing.push('subtitles')
    if (!allHaveSubsEs) missing.push('translate')
    if (!allHaveDub) missing.push('dubbing')
    setSelectedSteps(missing.length ? missing : ['subtitles', 'translate', 'dubbing'])
    setForceRegen(false)
    setOpen(true)
  }

  const toggleStep = (key) => {
    setSelectedSteps((prev) =>
      prev.includes(key) ? prev.filter((s) => s !== key) : [...prev, key],
    )
  }

  const launch = async () => {
    if (!seasonPath) {
      toast.error('No se pudo inferir la ruta de la Season')
      return
    }
    const order = ['chapters', 'subtitles', 'translate', 'dubbing']
    const steps = order.filter((k) => selectedSteps.includes(k))
    if (!steps.length) return
    const tid = toast.loading('Lanzando Season…')
    try {
      const opts = {}
      if (hasOracle && steps.includes('chapters')) opts.mode = 'oracle'
      if (forceRegen) opts.force = true
      const resp = await start.mutateAsync({ path: seasonPath, steps, options: opts })
      const id = resp?.pipeline_id || resp?.id
      toast.success('Pipeline Season lanzado', { id: tid })
      if (id) nav(`/pipelines/${id}`)
      setOpen(false)
    } catch (err) {
      toast.error(`Error: ${err?.message || 'desconocido'}`, { id: tid })
    }
  }

  return (
    <DropdownMenu open={open} onOpenChange={setOpen}>
      <DropdownMenuTrigger asChild>
        <Button
          size="sm"
          variant="outline"
          onClick={(e) => { e.stopPropagation(); openMenu() }}
          disabled={!seasonPath || start.isPending}
          title="Procesar toda la Season (elige los pasos)"
        >
          {start.isPending ? (
            <Loader2 className="mr-1 h-3 w-3 animate-spin" />
          ) : (
            <Play className="mr-1 h-3 w-3" />
          )}
          Procesar Season
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-56 p-2" onClick={(e) => e.stopPropagation()}>
        <p className="mb-2 text-xs font-medium text-zinc-400">Pasos a ejecutar</p>
        {[
          { key: 'chapters', label: 'Trocear capítulos', Icon: Scissors, done: isChaptered },
          { key: 'subtitles', label: 'Subtítulos EN', Icon: Captions, done: allHaveSubs },
          { key: 'translate', label: 'Traducir a ES', Icon: Languages, done: allHaveSubsEs },
          { key: 'dubbing', label: 'Doblaje', Icon: Mic, done: allHaveDub },
        ].map(({ key, label, Icon: StepIcon, done }) => {
          const locked = done && !forceRegen
          return (
            <button
              key={key}
              type="button"
              disabled={locked}
              onClick={(e) => { e.preventDefault(); toggleStep(key) }}
              className={cn(
                'flex w-full items-center gap-2 rounded px-2 py-1.5 text-sm transition-colors',
                locked
                  ? 'cursor-not-allowed text-zinc-600 line-through'
                  : 'cursor-pointer hover:bg-zinc-800 text-zinc-200',
              )}
            >
              <span
                className={cn(
                  'flex h-4 w-4 shrink-0 items-center justify-center rounded border',
                  locked
                    ? 'border-zinc-700 bg-zinc-800'
                    : selectedSteps.includes(key)
                      ? 'border-emerald-500 bg-emerald-500/20'
                      : 'border-zinc-600',
                )}
              >
                {(locked || selectedSteps.includes(key)) && (
                  <Check className="h-3 w-3 text-emerald-400" />
                )}
              </span>
              <StepIcon className="h-3.5 w-3.5" />
              {label}
            </button>
          )
        })}
        <div className="mt-2 border-t border-zinc-800 pt-2 space-y-0.5">
          <button
            type="button"
            onClick={(e) => { e.preventDefault(); setForceRegen((v) => !v) }}
            className="flex w-full items-center gap-2 rounded px-2 py-1.5 text-sm cursor-pointer hover:bg-zinc-800 text-zinc-400"
          >
            <span
              className={cn(
                'flex h-4 w-4 shrink-0 items-center justify-center rounded border',
                forceRegen ? 'border-amber-500 bg-amber-500/20' : 'border-zinc-600',
              )}
            >
              {forceRegen && <Check className="h-3 w-3 text-amber-400" />}
            </span>
            <RotateCw className="h-3.5 w-3.5" />
            Forzar regeneración
          </button>
        </div>
        <Button
          size="sm"
          className="mt-2 w-full"
          disabled={selectedSteps.length === 0 || start.isPending}
          onClick={launch}
        >
          {start.isPending ? (
            <Loader2 className="mr-1 h-3 w-3 animate-spin" />
          ) : (
            <Play className="mr-1 h-3 w-3" />
          )}
          Lanzar {selectedSteps.length ? `(${selectedSteps.length})` : ''}
        </Button>
      </DropdownMenuContent>
    </DropdownMenu>
  )
}

export default function ChaptersTab({ instructional }) {
  const videos = instructional?.videos || []
  const name = instructional?.name
  const { data: oracleData } = useOracleData(name)
  const hasOracle = !!oracleData && Array.isArray(oracleData.volumes)

  const seasons = useMemo(() => {
    const map = new Map()
    for (const v of videos) {
      const key = v.season || 'Sin temporada'
      if (!map.has(key)) map.set(key, [])
      map.get(key).push(v)
    }
    return [...map.entries()].sort(([a], [b]) => {
      if (a === 'Sin temporada') return 1
      if (b === 'Sin temporada') return -1
      return String(a).localeCompare(String(b), undefined, { numeric: true })
    })
  }, [videos])

  if (seasons.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-zinc-800 p-12 text-center">
        <Film className="h-10 w-10 text-zinc-700" />
        <p className="text-sm text-zinc-500">Aún no hay capítulos detectados.</p>
      </div>
    )
  }

  return (
    <Accordion
      type="multiple"
      defaultValue={seasons.map(([s]) => String(s))}
      className="space-y-2"
    >
      {seasons.map(([season, list]) => {
        const seasonPath = deriveSeasonPath(list)
        return (
        <AccordionItem
          key={season}
          value={String(season)}
          className="overflow-hidden rounded-lg border border-zinc-800/80 bg-zinc-950/50"
        >
          <AccordionTrigger className="px-4 hover:no-underline">
            <div className="flex w-full items-center gap-3">
              <span className="font-semibold text-zinc-100">{season}</span>
              <Badge variant="secondary" className="font-mono">
                {list.length} caps
              </Badge>
              <div
                className="ml-auto mr-2 flex items-center gap-2"
                onClick={(e) => e.stopPropagation()}
              >
                {season === 'Sin temporada' ? (
                  <SeasonPipelineButton
                    seasonPath={seasonPath}
                    steps={['chapters']}
                    label={hasOracle ? 'Trocear (oráculo)' : 'Trocear'}
                    Icon={Scissors}
                    title="Detectar y fragmentar capítulos en estos videos"
                    hasOracle={hasOracle}
                  />
                ) : (
                  <>
                    {hasOracle && (
                      <SeasonRenameOracleButton
                        seasonPath={seasonPath}
                        oracle={oracleData}
                        instructionalName={name}
                      />
                    )}
                    <SeasonPipelineButton
                      seasonPath={seasonPath}
                      steps={['subtitles']}
                      label="Subs EN"
                      Icon={Captions}
                      title="Generar subtítulos EN en toda la Season"
                    />
                    <SeasonValidateButton season={season} list={list} />
                    <SeasonPipelineButton
                      seasonPath={seasonPath}
                      steps={['translate']}
                      label="Subs ES"
                      Icon={Captions}
                      title="Traducir subtítulos EN → ES en toda la Season (requiere subs EN previos)"
                    />
                    <SeasonElevenLabsButton seasonPath={seasonPath} list={list} />
                    <SeasonProcessButton
                      seasonPath={seasonPath}
                      list={list}
                      hasOracle={hasOracle}
                      oracleData={oracleData}
                    />
                  </>
                )}
              </div>
            </div>
          </AccordionTrigger>
          <AccordionContent className="px-0 pb-0">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-zinc-900/40 text-xs uppercase tracking-wide text-zinc-500">
                  <tr>
                    <th className="px-3 py-2 text-left font-medium w-20">Código</th>
                    <th className="px-3 py-2 text-left font-medium">Título</th>
                    <th className="px-3 py-2 text-left font-medium w-20 whitespace-nowrap">Duración</th>
                    <th className="px-3 py-2 text-left font-medium w-28">Estado</th>
                    <th className="px-3 py-2 text-right font-medium w-auto whitespace-nowrap">Acción</th>
                  </tr>
                </thead>
                <tbody>
                  {list.map((v) => (
                    <ChapterRow
                      key={v.path}
                      video={v}
                      instructionalName={name}
                      hasOracle={hasOracle}
                    />
                  ))}
                </tbody>
              </table>
            </div>
          </AccordionContent>
        </AccordionItem>
        )
      })}
    </Accordion>
  )
}

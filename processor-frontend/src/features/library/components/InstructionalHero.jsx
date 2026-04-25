// Netflix/Plex-style hero for an instructional: poster side by side with
// title, stats, status pills and action CTAs. Background uses a blurred
// duplicate of the poster with an amber/zinc gradient overlay.
import { useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  Film,
  Upload,
  Loader2,
  Play,
  Sparkles,
  Pencil,
  Scissors,
  Captions,
  Mic,
  ChevronDown,
  Languages,
  Download,
  RefreshCw,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/Badge'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { cn } from '@/lib/utils'
import { toast } from 'sonner'
import { posterUrl, useUploadPoster, useRedownloadPoster } from '../api/useLibrary'

const PIPELINE_STEPS = [
  { id: 'chapters',   label: 'Capítulos',    Icon: Scissors,  desc: 'Detectar y trocear capítulos' },
  { id: 'subtitles',  label: 'Subtítulos',   Icon: Captions,  desc: 'Transcripción EN con WhisperX' },
  { id: 'translate',  label: 'Traducción',   Icon: Languages, desc: 'Traducir subtítulos EN → ES' },
  { id: 'dubbing',    label: 'Doblaje',      Icon: Mic,       desc: 'Genera guion .dub.es.srt + síntesis ES' },
]

const STEPS_KEY = 'process_all_steps_v2'
const FORCE_KEY = 'process_all_force_v1'
const CONTINUE_ON_FAIL_KEY = 'process_all_continue_on_fail_v1'

function loadSteps() {
  try {
    const raw = localStorage.getItem(STEPS_KEY)
    if (raw) {
      const parsed = JSON.parse(raw)
      if (Array.isArray(parsed) && parsed.length > 0) return parsed
    }
  } catch { /* noop */ }
  return ['chapters', 'subtitles', 'translate', 'dubbing']
}

function loadForce() {
  try {
    return localStorage.getItem(FORCE_KEY) === '1'
  } catch { return false }
}

function loadContinueOnFail() {
  try {
    const raw = localStorage.getItem(CONTINUE_ON_FAIL_KEY)
    // Default: true — un fallo puntual en S01 no debería detener S02-SXX.
    return raw === null ? true : raw === '1'
  } catch { return true }
}

function StatusPill({ ok, label, Icon }) {
  return (
    <Badge
      variant="outline"
      className={cn(
        'gap-1.5 border-zinc-700/70',
        ok
          ? 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300'
          : 'bg-zinc-800/50 text-zinc-400',
      )}
    >
      <Icon className="h-3 w-3" />
      {label}
    </Badge>
  )
}

export default function InstructionalHero({
  instructional,
  onProcessAll,
  onEditMetadata,
  processingAll = false,
}) {
  const nav = useNavigate()
  const [cacheBust, setCacheBust] = useState(() => instructional?.poster_mtime || 0)
  const [posterErrored, setPosterErrored] = useState(false)
  const [selectedSteps, setSelectedSteps] = useState(loadSteps)
  const [forceRegen, setForceRegen] = useState(loadForce)
  const [continueOnFail, setContinueOnFail] = useState(loadContinueOnFail)
  const fileInputRef = useRef(null)

  const toggleStep = (id) => {
    setSelectedSteps((prev) => {
      const next = prev.includes(id) ? prev.filter((s) => s !== id) : [...prev, id]
      try { localStorage.setItem(STEPS_KEY, JSON.stringify(next)) } catch { /* noop */ }
      return next
    })
  }

  const toggleForce = () => {
    setForceRegen((prev) => {
      const next = !prev
      try { localStorage.setItem(FORCE_KEY, next ? '1' : '0') } catch { /* noop */ }
      return next
    })
  }

  const toggleContinueOnFail = () => {
    setContinueOnFail((prev) => {
      const next = !prev
      try { localStorage.setItem(CONTINUE_ON_FAIL_KEY, next ? '1' : '0') } catch { /* noop */ }
      return next
    })
  }

  const handleProcessAll = () => {
    if (selectedSteps.length === 0) {
      toast.error('Selecciona al menos un paso')
      return
    }
    // Preserve pipeline order
    const ordered = PIPELINE_STEPS.map((s) => s.id).filter((id) => selectedSteps.includes(id))
    onProcessAll(ordered, { force: forceRegen, continueOnFail })
  }

  const name = instructional?.name || ''
  const author =
    instructional?.author || instructional?.metadata?.author || instructional?.instructor || ''

  const videos = instructional?.videos || []
  const seasons = useMemo(() => {
    const set = new Set()
    for (const v of videos) set.add(v.season || 'Sin temporada')
    return set.size
  }, [videos])

  const hasChapters = videos.length > 0 && videos.every((v) => v.has_chapters || v.is_chapter)
  const hasSubs = videos.length > 0 && videos.every((v) => v.has_subtitles_en)
  const hasDub = videos.length > 0 && videos.every((v) => v.has_dubbing || v.has_dubbed)

  const poster = instructional?.has_poster || instructional?.poster_filename
  const src = poster && !posterErrored ? `${posterUrl(name)}?v=${cacheBust}` : null

  const upload = useUploadPoster()
  const redownload = useRedownloadPoster()

  const handleRedownload = async () => {
    try {
      await redownload.mutateAsync({ name })
      setPosterErrored(false)
      setCacheBust((n) => n + 1)
      toast.success('Póster re-descargado desde Oracle')
    } catch (err) {
      const status = err?.status
      if (status === 404) {
        toast.error('Este instructional no tiene oracle.poster_url. Ejecuta Oracle primero.')
      } else {
        toast.error(`Error re-descargando póster: ${err?.message || 'desconocido'}`)
      }
    }
  }

  const handleUpload = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    try {
      await upload.mutateAsync({ name, file })
      setPosterErrored(false)
      setCacheBust((n) => n + 1)
      toast.success('Póster actualizado')
    } catch (err) {
      toast.error(`Error subiendo póster: ${err.message || 'desconocido'}`)
    } finally {
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  return (
    <motion.header
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
      className="relative overflow-hidden rounded-xl border border-zinc-800/80 bg-zinc-950"
    >
      {/* Blurred poster background */}
      {src && (
        <div
          aria-hidden
          className="absolute inset-0 scale-110 opacity-30"
          style={{
            backgroundImage: `url(${src})`,
            backgroundSize: 'cover',
            backgroundPosition: 'center',
            filter: 'blur(40px) saturate(1.2)',
          }}
        />
      )}
      {/* Gradient overlay */}
      <div
        aria-hidden
        className="absolute inset-0 bg-gradient-to-br from-amber-950/20 via-zinc-950/80 to-zinc-950"
      />

      <div className="relative flex flex-col gap-6 p-6 md:flex-row md:p-8">
        {/* Poster */}
        <div className="flex-shrink-0">
          {/* Sin aspect-ratio fijo: el ancho es 200/280 px y la altura la
              dicta la imagen. BJJFanatics mezcla ratios 0.70-0.80 entre
              instruccionales y forzar uno común recortaba texto o dejaba
              bandas negras según el caso. */}
          <div className="relative w-[200px] overflow-hidden rounded-lg border border-zinc-800 bg-black shadow-2xl md:w-[280px]">
            {src ? (
              <img
                src={src}
                alt={name}
                onError={() => setPosterErrored(true)}
                className="block h-auto w-full"
                loading="lazy"
                decoding="async"
              />
            ) : (
              <div className="flex aspect-[3/4] w-full items-center justify-center">
                <Film className="h-16 w-16 text-zinc-700" />
              </div>
            )}
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/jpeg,image/png,image/webp"
            onChange={handleUpload}
            className="hidden"
          />
          <Button
            variant="outline"
            size="sm"
            className="mt-3 w-full"
            onClick={() => fileInputRef.current?.click()}
            disabled={upload.isPending}
          >
            {upload.isPending ? (
              <Loader2 className="mr-2 h-3.5 w-3.5 animate-spin" />
            ) : (
              <Upload className="mr-2 h-3.5 w-3.5" />
            )}
            {upload.isPending ? 'Subiendo…' : 'Cambiar póster'}
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="mt-2 w-full text-zinc-400 hover:text-zinc-100"
            onClick={handleRedownload}
            disabled={redownload.isPending}
            title="Re-descargar póster desde la URL de Oracle"
          >
            {redownload.isPending ? (
              <Loader2 className="mr-2 h-3.5 w-3.5 animate-spin" />
            ) : (
              <Download className="mr-2 h-3.5 w-3.5" />
            )}
            {redownload.isPending ? 'Descargando…' : 'Re-descargar de Oracle'}
          </Button>
        </div>

        {/* Info */}
        <div className="min-w-0 flex-1">
          <h1 className="truncate text-3xl font-bold tracking-tight text-zinc-100 md:text-4xl">
            {name}
          </h1>
          {author && (
            <p className="mt-1 truncate text-sm text-zinc-400">{author}</p>
          )}
          <p className="mt-1 truncate font-mono text-[11px] text-zinc-600">
            {instructional?.path}
          </p>

          <div className="mt-4 flex flex-wrap items-center gap-3 text-sm text-zinc-400">
            <span>{videos.length} vídeos</span>
            <span className="text-zinc-700">·</span>
            <span>
              {seasons} {seasons === 1 ? 'temporada' : 'temporadas'}
            </span>
          </div>

          <div className="mt-4 flex flex-wrap gap-2">
            <StatusPill ok={hasChapters} label="Capítulos" Icon={Scissors} />
            <StatusPill ok={hasSubs} label="Subtítulos" Icon={Captions} />
            <StatusPill ok={hasDub} label="Doblaje" Icon={Mic} />
          </div>

          <div className="mt-6 flex flex-wrap gap-2">
            {/* Split button: main action + step selector */}
            <div className="flex items-stretch">
              <Button
                onClick={handleProcessAll}
                disabled={processingAll || selectedSteps.length === 0}
                className="rounded-r-none border-r border-r-primary-foreground/20"
              >
                {processingAll
                  ? <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  : <Play className="mr-2 h-4 w-4" />}
                {processingAll ? 'Iniciando…' : 'Procesar'}
                {!processingAll && selectedSteps.length < PIPELINE_STEPS.length && (
                  <span className="ml-1.5 rounded-full bg-primary-foreground/20 px-1.5 py-0.5 text-[10px] font-semibold">
                    {selectedSteps.length}
                  </span>
                )}
              </Button>
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button
                    disabled={processingAll}
                    className="rounded-l-none px-2"
                    aria-label="Elegir pasos del pipeline"
                  >
                    <ChevronDown className="h-4 w-4" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="start" className="w-64 p-1" onCloseAutoFocus={(e) => e.preventDefault()}>
                  <DropdownMenuLabel className="text-xs uppercase tracking-wide text-muted-foreground">
                    Pasos del pipeline
                  </DropdownMenuLabel>
                  <DropdownMenuSeparator />
                  {PIPELINE_STEPS.map(({ id, label, Icon, desc }) => {
                    const checked = selectedSteps.includes(id)
                    return (
                      <button
                        key={id}
                        type="button"
                        onClick={(e) => { e.preventDefault(); toggleStep(id) }}
                        className={cn(
                          'flex w-full items-center gap-3 rounded-sm px-2 py-2 text-left text-sm transition-colors hover:bg-accent',
                          checked ? 'text-foreground' : 'text-muted-foreground',
                        )}
                      >
                        <div className={cn(
                          'flex h-5 w-5 shrink-0 items-center justify-center rounded border',
                          checked ? 'border-primary bg-primary text-primary-foreground' : 'border-border',
                        )}>
                          {checked && <span className="text-[10px] font-bold">✓</span>}
                        </div>
                        <Icon className="h-4 w-4 shrink-0" />
                        <div className="min-w-0">
                          <div className="font-medium">{label}</div>
                          <div className="truncate text-[11px] text-muted-foreground">{desc}</div>
                        </div>
                      </button>
                    )
                  })}
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
            <Button
              variant="outline"
              onClick={() => nav(`/library/${encodeURIComponent(name)}/oracle`)}
            >
              <Sparkles className="mr-2 h-4 w-4 text-amber-400" /> Oracle
            </Button>
            <Button variant="outline" onClick={onEditMetadata}>
              <Pencil className="mr-2 h-4 w-4" /> Editar metadatos
            </Button>
          </div>

          <div className="mt-3 flex flex-wrap gap-2">
            <button
              type="button"
              onClick={toggleForce}
              disabled={processingAll}
              title="Sobrescribe subtítulos EN, traducciones ES y doblajes existentes. Los capítulos ya detectados no se regeneran."
              className={cn(
                'inline-flex items-center gap-2 rounded-md border px-2.5 py-1.5 text-xs font-medium transition-colors',
                forceRegen
                  ? 'border-amber-500/50 bg-amber-500/10 text-amber-300 hover:bg-amber-500/20'
                  : 'border-zinc-700 bg-zinc-900/50 text-zinc-400 hover:bg-zinc-800/60 hover:text-zinc-200',
                processingAll && 'opacity-50 cursor-not-allowed',
              )}
            >
              <div className={cn(
                'flex h-4 w-4 shrink-0 items-center justify-center rounded border',
                forceRegen ? 'border-amber-400 bg-amber-500 text-zinc-950' : 'border-zinc-600',
              )}>
                {forceRegen && <span className="text-[9px] font-bold leading-none">✓</span>}
              </div>
              <RefreshCw className="h-3.5 w-3.5" />
              Forzar regeneración
            </button>

            <button
              type="button"
              onClick={toggleContinueOnFail}
              disabled={processingAll}
              title="Si una season falla, sigue con las siguientes en vez de detener todo el instruccional."
              className={cn(
                'inline-flex items-center gap-2 rounded-md border px-2.5 py-1.5 text-xs font-medium transition-colors',
                continueOnFail
                  ? 'border-emerald-500/50 bg-emerald-500/10 text-emerald-300 hover:bg-emerald-500/20'
                  : 'border-zinc-700 bg-zinc-900/50 text-zinc-400 hover:bg-zinc-800/60 hover:text-zinc-200',
                processingAll && 'opacity-50 cursor-not-allowed',
              )}
            >
              <div className={cn(
                'flex h-4 w-4 shrink-0 items-center justify-center rounded border',
                continueOnFail ? 'border-emerald-400 bg-emerald-500 text-zinc-950' : 'border-zinc-600',
              )}>
                {continueOnFail && <span className="text-[9px] font-bold leading-none">✓</span>}
              </div>
              Continuar si una season falla
            </button>
          </div>
        </div>
      </div>
    </motion.header>
  )
}

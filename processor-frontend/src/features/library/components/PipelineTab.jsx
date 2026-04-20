// Pipeline tab: preflight banner + step selector + config + last-pipeline diff.
// Reuses `usePreflight`, `useStartPipeline`, `usePipelineEta`, `usePipelines`,
// `useVoices` and the shared `StepDiff` component.
import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  Play,
  Loader2,
  ShieldCheck,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Scissors,
  Captions,
  Languages,
  Mic,
} from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Checkbox } from '@/components/ui/checkbox'
import { Badge } from '@/components/ui/Badge'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { toast } from 'sonner'
import { cn } from '@/lib/utils'
import { usePreflight } from '@/features/preflight/api/usePreflight'
import {
  useStartPipeline,
  usePipelineEta,
  usePipelines,
} from '@/features/pipeline/api/usePipeline'
import { useVoices } from '@/features/voices/api/useVoices'
import StepDiff from '@/features/pipeline/components/StepDiff'

const STEPS = [
  { id: 'chapters', label: 'Capítulos', Icon: Scissors },
  { id: 'subtitles', label: 'Subtítulos EN', Icon: Captions },
  { id: 'translate', label: 'Traducción ES', Icon: Languages },
  { id: 'dubbing', label: 'Doblaje ES', Icon: Mic },
]

function fmtDuration(sec) {
  if (!sec || sec <= 0) return null
  const s = Math.round(sec)
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  return `~${m}m`
}

function PreflightBanner({ path }) {
  const { data, isLoading, isError, error } = usePreflight(path)
  if (isLoading) {
    return (
      <div className="flex items-center gap-2 rounded-lg border border-zinc-800 bg-zinc-950/60 p-3 text-sm text-zinc-400">
        <Loader2 className="h-4 w-4 animate-spin" /> Ejecutando comprobaciones…
      </div>
    )
  }
  if (isError) {
    return (
      <div
        role="alert"
        className="rounded-lg border border-red-900/60 bg-red-950/30 p-3"
      >
        <div className="flex items-center gap-2 text-sm text-red-300">
          <XCircle className="h-4 w-4" /> Pre-flight no disponible
        </div>
        <p className="mt-1 text-xs text-zinc-500">{error?.message || '—'}</p>
      </div>
    )
  }
  const allOk = !!data?.all_ok
  const checks = data?.checks || []
  return (
    <div
      className={cn(
        'rounded-lg border p-3',
        allOk
          ? 'border-emerald-500/30 bg-emerald-500/5'
          : 'border-amber-500/40 bg-amber-500/5',
      )}
    >
      <div className="mb-2 flex items-center gap-2">
        {allOk ? (
          <ShieldCheck className="h-4 w-4 text-emerald-400" />
        ) : (
          <AlertTriangle className="h-4 w-4 text-amber-400" />
        )}
        <span className="text-sm font-semibold text-zinc-100">
          {allOk ? 'Pre-flight OK' : 'Pre-flight con incidencias'}
        </span>
      </div>
      <ul className="space-y-1">
        {checks.map((c) => (
          <li key={c.name} className="flex items-start gap-2 text-xs">
            {c.ok ? (
              <CheckCircle2 className="mt-0.5 h-3 w-3 shrink-0 text-emerald-400" />
            ) : (
              <XCircle className="mt-0.5 h-3 w-3 shrink-0 text-red-400" />
            )}
            <span className="min-w-[90px] font-medium text-zinc-300">
              {c.name}
            </span>
            <span className="text-zinc-500">{c.message}</span>
          </li>
        ))}
      </ul>
    </div>
  )
}

export default function PipelineTab({ instructional }) {
  const nav = useNavigate()
  const path = instructional?.path
  const [searchParams] = useSearchParams()

  // Initial steps from URL ?steps=chapters,subtitles
  const initialSteps = useMemo(() => {
    const raw = searchParams.get('steps')
    const known = new Set(STEPS.map((s) => s.id))
    if (raw) {
      const picked = raw
        .split(',')
        .map((s) => s.trim())
        .filter((s) => known.has(s))
      if (picked.length) return new Set(picked)
    }
    return new Set(['chapters', 'subtitles', 'translate', 'dubbing'])
  }, [searchParams])

  const [selected, setSelected] = useState(initialSteps)
  const [voiceProfile, setVoiceProfile] = useState('')
  const [outputDir, setOutputDir] = useState('')
  const [dubbingScript, setDubbingScript] = useState(true)

  useEffect(() => {
    setSelected(initialSteps)
  }, [initialSteps])

  const toggleStep = (id) => {
    setSelected((s) => {
      const next = new Set(s)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const stepsList = STEPS.filter((s) => selected.has(s.id)).map((s) => s.id)
  const { data: eta } = usePipelineEta(
    stepsList.length ? { steps: stepsList.join(','), path } : undefined,
  )
  const { data: voices } = useVoices()
  const start = useStartPipeline()

  // Find last pipeline for this instructional path
  const { data: pipelines } = usePipelines(path ? { path } : undefined)
  const lastPipeline = useMemo(() => {
    const list = Array.isArray(pipelines) ? pipelines : []
    return list
      .slice()
      .sort(
        (a, b) =>
          Date.parse(b.created_at || b.started_at || 0) -
          Date.parse(a.created_at || a.started_at || 0),
      )[0]
  }, [pipelines])

  const handleRun = async () => {
    if (!path) return
    if (stepsList.length === 0) {
      toast.error('Selecciona al menos un paso')
      return
    }
    try {
      const options = {}
      if (voiceProfile) options.voice_profile = voiceProfile
      if (outputDir.trim()) options.output_dir = outputDir.trim()
      // Iso-synchronous dubbing script (.dub.es.srt) in addition to the literal
      // subtitle file. Only matters when the translate step runs.
      options.dubbing_mode = Boolean(dubbingScript)
      const resp = await start.mutateAsync({
        path,
        steps: stepsList,
        options,
      })
      const id = resp?.pipeline_id || resp?.id
      if (id) {
        toast.success('Pipeline lanzado')
        nav(`/pipelines/${id}`)
      } else {
        toast.success('Pipeline lanzado')
      }
    } catch (e) {
      toast.error(`Error: ${e.message || 'desconocido'}`)
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-4"
    >
      <PreflightBanner path={path} />

      <Card className="border-zinc-800 bg-zinc-950/60">
        <CardHeader className="pb-3">
          <CardTitle className="text-sm">Pasos del pipeline</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
            {STEPS.map(({ id, label, Icon }) => {
              const checked = selected.has(id)
              const eSec = eta?.per_step?.[id]
              return (
                <label
                  key={id}
                  className={cn(
                    'flex items-center gap-3 rounded-lg border p-3 cursor-pointer transition-colors',
                    checked
                      ? 'border-amber-500/40 bg-amber-500/5'
                      : 'border-zinc-800 bg-zinc-950 hover:border-zinc-700',
                  )}
                >
                  <Checkbox
                    checked={checked}
                    onCheckedChange={() => toggleStep(id)}
                  />
                  <Icon
                    className={cn(
                      'h-4 w-4',
                      checked ? 'text-amber-400' : 'text-zinc-500',
                    )}
                  />
                  <span className="flex-1 text-sm text-zinc-100">{label}</span>
                  {eSec != null && (
                    <Badge variant="outline" className="text-[10px] tabular-nums">
                      {fmtDuration(eSec) || '—'}
                    </Badge>
                  )}
                </label>
              )
            })}
          </div>

          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            <div>
              <Label htmlFor="voice-profile" className="text-xs">
                Perfil de voz (doblaje)
              </Label>
              <Select value={voiceProfile} onValueChange={setVoiceProfile}>
                <SelectTrigger id="voice-profile" className="mt-1 h-9">
                  <SelectValue placeholder="Por defecto" />
                </SelectTrigger>
                <SelectContent>
                  {(voices || []).map((v) => {
                    const val = v.slug || v.id || v.name
                    return (
                      <SelectItem key={val} value={String(val)}>
                        {v.name || val}
                      </SelectItem>
                    )
                  })}
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label htmlFor="output-dir" className="text-xs">
                Output dir (opcional)
              </Label>
              <Input
                id="output-dir"
                value={outputDir}
                onChange={(e) => setOutputDir(e.target.value)}
                placeholder="/media/out/…"
                className="mt-1 h-9 font-mono text-xs"
              />
            </div>
          </div>

          <label
            className={cn(
              'flex items-start gap-3 rounded-lg border p-3 cursor-pointer transition-colors',
              dubbingScript
                ? 'border-amber-500/40 bg-amber-500/5'
                : 'border-zinc-800 bg-zinc-950 hover:border-zinc-700',
            )}
          >
            <Checkbox
              checked={dubbingScript}
              onCheckedChange={(v) => setDubbingScript(Boolean(v))}
              className="mt-0.5"
            />
            <div className="min-w-0 flex-1">
              <div className="text-sm text-zinc-100">
                Generar guion de doblaje (<code className="font-mono text-xs">.dub.es.srt</code>)
              </div>
              <p className="mt-0.5 text-xs text-zinc-500">
                Versión compactada del ES para que el TTS cuadre con los
                tiempos del video sin acelerar. El <code className="font-mono">.es.srt</code>{' '}
                literal (subtítulos) no se toca.
              </p>
            </div>
          </label>

          <div className="flex items-center justify-between pt-2">
            <p className="text-xs text-zinc-500">
              {stepsList.length} paso(s) seleccionado(s)
            </p>
            <Button onClick={handleRun} disabled={start.isPending}>
              {start.isPending ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <Play className="mr-2 h-4 w-4" />
              )}
              Ejecutar pipeline
            </Button>
          </div>
        </CardContent>
      </Card>

      {lastPipeline && (
        <Card className="border-zinc-800 bg-zinc-950/60">
          <CardHeader className="pb-2">
            <CardTitle className="flex items-center gap-2 text-sm">
              Último pipeline
              <Badge
                variant="outline"
                className="text-[10px] uppercase"
                onClick={() =>
                  nav(`/pipelines/${lastPipeline.pipeline_id || lastPipeline.id}`)
                }
              >
                {lastPipeline.status}
              </Badge>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {(lastPipeline.steps || []).map((s, i) => (
              <div key={`${s.name}-${i}`}>
                <div className="flex items-center gap-2 text-xs text-zinc-400">
                  <span className="font-medium capitalize text-zinc-200">
                    {s.name}
                  </span>
                  <span className="text-zinc-600">{s.status}</span>
                </div>
                <StepDiff diff={s.diff} />
              </div>
            ))}
            <Button
              variant="outline"
              size="sm"
              className="mt-2"
              onClick={() =>
                nav(`/pipelines/${lastPipeline.pipeline_id || lastPipeline.id}`)
              }
            >
              Ver detalle
            </Button>
          </CardContent>
        </Card>
      )}
    </motion.div>
  )
}

// WIRE_ROUTE_SETTINGS: /settings → src/features/settings/pages/SettingsPage.jsx
import { useState, useMemo, useEffect } from 'react'
import {
  FolderOpen,
  Folder,
  ChevronRight,
  ChevronLeft,
  Cog,
  Sparkles,
  Palette,
  Shield,
  Send,
  Eye,
  EyeOff,
  Loader2,
  Save,
  CheckCircle2,
  XCircle,
  Download,
  Upload,
  RotateCcw,
  HardDrive,
  Plug,
  Wrench,
  Trash2,
  Copy,
  ScanSearch,
  Ban,
  Clock,
  Route,
  AlertTriangle,
  RefreshCw,
  Mic2,
} from 'lucide-react'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import { toast } from 'sonner'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/Card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Separator } from '@/components/ui/separator'
import { Badge } from '@/components/ui/Badge'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { useTheme } from '@/components/theme-provider'
import { useBackendsHealth } from '@/components/layout/useBackendsHealth'
import { useSettings, useUpdateSettings } from '@/features/settings/api/useSettings'
import { useMount, useMountStatus } from '@/features/settings/api/useMount'
import { useProviders } from '@/features/oracle/api/useOracle'
import { http } from '@/lib/httpClient'
import { formatBytes, formatDuration } from '@/lib/format'
import {
  useCleanupScan,
  useCleanupJob,
  useCleanupApply,
} from '@/features/cleanup/api/useCleanup'
import { useDuplicatesScan, useDuplicatesJob } from '@/features/duplicates/api/useDuplicates'
import { Checkbox } from '@/components/ui/checkbox'
import { Progress } from '@/components/ui/Progress'
import {
  Accordion,
  AccordionItem,
  AccordionTrigger,
  AccordionContent,
} from '@/components/ui/accordion'
import {
  AlertDialog,
  AlertDialogTrigger,
  AlertDialogContent,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogAction,
  AlertDialogCancel,
} from '@/components/ui/alert-dialog'

// --- Zod schemas per section ---
const librarySchema = z.object({
  library_path: z.string().trim(),
})

const mountSchema = z.object({
  share: z.string().trim().min(1, 'Requerido'),
  username: z.string().trim().optional(),
  password: z.string().optional(),
})

const processingSchema = z.object({
  output_dir: z.string().optional(),
  source_lang: z.string().min(2).max(8).default('en'),
  target_lang: z.string().min(2).max(8).default('es'),
})

const ttsSchema = z.object({
  tts_engine: z.enum(['s2pro', 'elevenlabs', 'piper', 'kokoro']).default('elevenlabs'),
  elevenlabs_voice_id: z.string().optional().nullable(),
  elevenlabs_model_id: z.string().optional().nullable(),
  piper_model_path: z.string().optional().nullable(),
  kokoro_voice: z.enum(['em_alex', 'em_santa']).optional().nullable(),
  s2_voice_profile: z.string().optional().nullable(),
  s2_ref_text: z.string().optional().nullable(),
  s2_temperature: z.coerce.number().min(0.1).max(1.5).optional(),
  s2_top_p: z.coerce.number().min(0.1).max(1.0).optional(),
  s2_top_k: z.coerce.number().int().min(1).max(200).optional(),
  s2_max_tokens: z.coerce.number().int().min(128).max(2048).optional(),
})

const oracleSchema = z.object({
  provider_default: z.string().optional(),
  timeout_seconds: z.coerce.number().int().min(5).max(300).default(30),
})

const telegramSchema = z.object({
  telegram_api_id: z
    .union([z.string().length(0), z.coerce.number().int().positive()])
    .optional(),
  telegram_api_hash: z
    .string()
    .optional()
    .refine((v) => !v || /^[a-f0-9]{32}$/.test(v), {
      message: 'Debe ser 32 caracteres hex (a-f, 0-9)',
    }),
})

const translationSchema = z.object({
  translation_provider: z.enum(['ollama', 'openai']).default('ollama'),
  translation_model: z.string().optional(),
  translation_fallback_provider: z.enum(['', 'ollama', 'openai']).optional(),
  openai_api_key: z.string().optional(),
})

const SECTIONS = [
  { id: 'library', label: 'Biblioteca', icon: FolderOpen },
  { id: 'processing', label: 'Procesamiento', icon: Cog },
  { id: 'tts', label: 'TTS / Doblaje', icon: Mic2 },
  { id: 'oracle', label: 'Oracle', icon: Sparkles },
  { id: 'telegram', label: 'Telegram', icon: Send },
  { id: 'translation', label: 'Traducción', icon: Plug },
  { id: 'authors', label: 'Alias autores', icon: Sparkles },
  { id: 'appearance', label: 'Apariencia', icon: Palette },
  { id: 'maintenance', label: 'Mantenimiento', icon: Wrench },
  { id: 'advanced', label: 'Avanzado', icon: Shield },
]

export default function SettingsPage() {
  const [active, setActive] = useState('library')
  const { data: settings, isLoading } = useSettings()

  return (
    <div>
      <header className="mb-6">
        <h1 className="text-2xl font-bold">Configuración</h1>
        <p className="text-sm text-muted-foreground">
          Ajustes globales de la plataforma de procesamiento.
        </p>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-[220px_1fr] gap-6">
        {/* Vertical nav */}
        <nav aria-label="Secciones de configuración" className="md:sticky md:top-4 self-start">
          <ul className="flex md:flex-col gap-1 overflow-x-auto md:overflow-visible">
            {SECTIONS.map((s) => {
              const Icon = s.icon
              const isActive = active === s.id
              return (
                <li key={s.id}>
                  <button
                    type="button"
                    onClick={() => setActive(s.id)}
                    aria-current={isActive ? 'page' : undefined}
                    className={`w-full flex items-center gap-2 px-3 py-2 rounded-md text-sm transition-colors ${
                      isActive
                        ? 'bg-primary/10 text-primary border border-primary/30'
                        : 'text-muted-foreground hover:bg-muted hover:text-foreground'
                    }`}
                  >
                    <Icon size={14} />
                    {s.label}
                  </button>
                </li>
              )
            })}
          </ul>
        </nav>

        {/* Content */}
        <div className="min-w-0 space-y-4">
          {isLoading && !settings ? (
            <Card>
              <CardContent className="py-10 text-center text-muted-foreground text-sm">
                <Loader2 className="inline-block animate-spin mr-2" size={14} />
                Cargando…
              </CardContent>
            </Card>
          ) : (
            <>
              {active === 'library' && <LibrarySection settings={settings} />}
              {active === 'processing' && <ProcessingSection settings={settings} />}
              {active === 'tts' && <TtsSection settings={settings} />}
              {active === 'oracle' && <OracleSection settings={settings} />}
              {active === 'telegram' && <TelegramSection settings={settings} />}
              {active === 'translation' && <TranslationSection settings={settings} />}
              {active === 'authors' && <AuthorAliasesSection settings={settings} />}
              {active === 'appearance' && <AppearanceSection />}
              {active === 'maintenance' && <MaintenanceSection />}
              {active === 'advanced' && <AdvancedSection settings={settings} />}
            </>
          )}
        </div>
      </div>
    </div>
  )
}

// --- Sections ---

function LibrarySection({ settings }) {
  const updateMut = useUpdateSettings()
  const [selected, setSelected] = useState(settings?.library_path || '')
  const initial = settings?.library_path || ''
  const isDirty = selected !== initial

  const onSave = async () => {
    try {
      await updateMut.mutateAsync({ library_path: selected })
      toast.success('Biblioteca guardada')
    } catch (e) {
      toast.error(e?.message || 'Error al guardar')
    }
  }

  return (
    <div className="space-y-4">
      <NasMountCard onMounted={() => setSelected('/media')} />
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FolderOpen size={16} className="text-primary" />
            Biblioteca
            {isDirty && (
              <Badge variant="outline" className="ml-2 border-amber-500/40 text-amber-500">
                sin guardar
              </Badge>
            )}
          </CardTitle>
          <CardDescription>
            Elige la carpeta de instruccionales dentro del volumen montado.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <Label>Carpeta seleccionada</Label>
            <div className="mt-1.5 flex items-center gap-2 px-3 py-2 rounded-md border bg-muted/30 font-mono text-sm">
              {selected ? (
                <>
                  <CheckCircle2 size={14} className="text-emerald-500 shrink-0" />
                  <span className="truncate">{selected}</span>
                </>
              ) : (
                <>
                  <XCircle size={14} className="text-muted-foreground shrink-0" />
                  <span className="text-muted-foreground">Sin seleccionar</span>
                </>
              )}
            </div>
          </div>
          <LibraryPicker value={selected} onChange={setSelected} />
        </CardContent>
        <Separator />
        <div className="flex justify-end gap-2 p-4">
          <Button type="button" onClick={onSave} disabled={!isDirty || !selected || updateMut.isPending}>
            {updateMut.isPending ? <Loader2 size={14} className="animate-spin" /> : <Save size={14} />}
            Guardar
          </Button>
        </div>
      </Card>
    </div>
  )
}

function LibraryPicker({ value, onChange }) {
  const [cwd, setCwd] = useState('')
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const load = async (path) => {
    setLoading(true)
    setError(null)
    try {
      const qs = path ? `?path=${encodeURIComponent(path)}` : ''
      const res = await http.get(`/fs/browse${qs}`)
      setData(res)
      setCwd(res.path)
    } catch (e) {
      setError(e?.message || 'Error listando carpetas')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load('') }, []) // initial load

  return (
    <div>
      <Label>Explorador</Label>
      <div className="mt-1.5 rounded-md border overflow-hidden">
        <div className="flex items-center gap-2 px-3 py-2 bg-muted/40 border-b">
          <Button
            type="button"
            size="sm"
            variant="ghost"
            disabled={!data?.parent || loading}
            onClick={() => load(data.parent)}
          >
            <ChevronLeft size={14} />
          </Button>
          <code className="text-xs truncate flex-1">{cwd || '…'}</code>
          <Button
            type="button"
            size="sm"
            variant="secondary"
            disabled={!cwd || loading || value === cwd}
            onClick={() => onChange(cwd)}
          >
            Usar esta carpeta
          </Button>
        </div>
        <div className="max-h-64 overflow-y-auto">
          {loading ? (
            <div className="py-6 text-center text-sm text-muted-foreground">
              <Loader2 size={14} className="inline animate-spin mr-2" />
              Cargando…
            </div>
          ) : error ? (
            <div className="py-6 text-center text-sm text-destructive">{error}</div>
          ) : data?.entries?.length ? (
            <ul className="divide-y">
              {data.entries.map((e) => (
                <li key={e.path}>
                  <button
                    type="button"
                    onClick={() => load(e.path)}
                    className="w-full flex items-center gap-2 px-3 py-2 text-sm hover:bg-muted text-left"
                  >
                    <Folder size={14} className="text-primary shrink-0" />
                    <span className="truncate flex-1">{e.name}</span>
                    <ChevronRight size={14} className="text-muted-foreground" />
                  </button>
                </li>
              ))}
            </ul>
          ) : (
            <div className="py-6 text-center text-sm text-muted-foreground">
              Carpeta vacía
            </div>
          )}
        </div>
      </div>
      <p className="text-xs text-muted-foreground mt-2">
        Navega y pulsa <strong>Usar esta carpeta</strong>. Solo se muestra el contenido del volumen
        montado en el contenedor.
      </p>
    </div>
  )
}

function NasMountCard({ onMounted }) {
  const { data: status, isLoading } = useMountStatus()
  const mountMut = useMount()
  const [showPass, setShowPass] = useState(false)
  const form = useForm({
    resolver: zodResolver(mountSchema),
    defaultValues: { share: '', username: '', password: '' },
  })

  const mounted = status?.mounted
  const onSubmit = async (values) => {
    try {
      const payload = {
        share: values.share.trim(),
        username: values.username?.trim() || 'guest',
        password: values.password || '',
      }
      const res = await mountMut.mutateAsync(payload)
      if (res?.error) {
        toast.error(res.error)
        return
      }
      toast.success('NAS montado correctamente')
      onMounted?.()
    } catch (e) {
      toast.error(e?.message || 'Error al montar NAS')
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <HardDrive size={16} className="text-primary" />
          Conectar NAS / Unidad de red
          {mounted && (
            <Badge variant="outline" className="ml-2 border-emerald-500/40 text-emerald-500">
              montado
            </Badge>
          )}
        </CardTitle>
        <CardDescription>
          Monta una carpeta compartida SMB/CIFS en <code>/media</code> para acceder a los instruccionales.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        {mounted ? (
          <div className="space-y-2">
            <p className="text-sm text-muted-foreground">
              <CheckCircle2 size={14} className="inline -mt-0.5 mr-1 text-emerald-500" />
              NAS montado — {status?.directories ?? 0} carpetas detectadas.
            </p>
            {Array.isArray(status?.items) && status.items.length > 0 && (
              <div className="flex flex-wrap gap-1.5">
                {status.items.slice(0, 8).map((name) => (
                  <Badge key={name} variant="secondary" className="text-[10px]">{name}</Badge>
                ))}
                {status.items.length > 8 && (
                  <Badge variant="outline" className="text-[10px]">
                    +{status.items.length - 8} más
                  </Badge>
                )}
              </div>
            )}
          </div>
        ) : (
          <form id="nas-mount-form" onSubmit={form.handleSubmit(onSubmit)} className="space-y-3">
            <div>
              <Label htmlFor="nas-share">Ruta del share</Label>
              <Input
                id="nas-share"
                placeholder="//10.10.100.6/multimedia/instruccionales"
                {...form.register('share')}
              />
              {form.formState.errors.share && (
                <p className="text-xs text-destructive mt-1">{form.formState.errors.share.message}</p>
              )}
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div>
                <Label htmlFor="nas-user">Usuario</Label>
                <Input
                  id="nas-user"
                  placeholder="guest"
                  autoComplete="username"
                  {...form.register('username')}
                />
              </div>
              <div>
                <Label htmlFor="nas-pass">Contraseña</Label>
                <div className="relative">
                  <Input
                    id="nas-pass"
                    type={showPass ? 'text' : 'password'}
                    autoComplete="current-password"
                    {...form.register('password')}
                  />
                  <button
                    type="button"
                    onClick={() => setShowPass((v) => !v)}
                    className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                    aria-label={showPass ? 'Ocultar' : 'Mostrar'}
                  >
                    {showPass ? <EyeOff size={14} /> : <Eye size={14} />}
                  </button>
                </div>
              </div>
            </div>
          </form>
        )}
      </CardContent>
      <Separator />
      <div className="flex justify-end gap-2 p-4">
        {!mounted && (
          <Button
            type="submit"
            form="nas-mount-form"
            disabled={mountMut.isPending || isLoading}
          >
            {mountMut.isPending ? <Loader2 size={14} className="animate-spin" /> : <Plug size={14} />}
            {mountMut.isPending ? 'Conectando...' : 'Conectar NAS'}
          </Button>
        )}
      </div>
    </Card>
  )
}

function ProcessingSection({ settings }) {
  const updateMut = useUpdateSettings()
  const defaults = settings?.processing_defaults || {}
  const form = useForm({
    resolver: zodResolver(processingSchema),
    defaultValues: {
      output_dir: defaults.output_dir || '',
      source_lang: defaults.source_lang || 'en',
      target_lang: defaults.target_lang || 'es',
    },
  })

  const onSubmit = async (values) => {
    try {
      const payload = {
        processing_defaults: {
          ...defaults,
          output_dir: values.output_dir || undefined,
          source_lang: values.source_lang,
          target_lang: values.target_lang,
        },
      }
      await updateMut.mutateAsync(payload)
      toast.success('Procesamiento guardado')
      form.reset(values)
    } catch (e) {
      toast.error(e?.message || 'Error al guardar')
    }
  }

  const { isDirty } = form.formState

  return (
    <form onSubmit={form.handleSubmit(onSubmit)}>
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Cog size={16} className="text-primary" />
            Procesamiento
            {isDirty && <Badge variant="outline" className="ml-2 border-amber-500/40 text-amber-500">sin guardar</Badge>}
          </CardTitle>
          <CardDescription>Valores por defecto para nuevos pipelines.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <Label htmlFor="output_dir">Directorio de salida (opcional)</Label>
            <Input
              id="output_dir"
              placeholder="Dejar vacío para procesar in-place"
              {...form.register('output_dir')}
              className="mt-1.5"
            />
            <p className="text-xs text-muted-foreground mt-1">
              Cuando está vacío, los artefactos se guardan junto a los capítulos.
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div>
              <Label htmlFor="source_lang">Idioma origen</Label>
              <Input
                id="source_lang"
                placeholder="en"
                {...form.register('source_lang')}
                className="mt-1.5 font-mono"
              />
            </div>
            <div>
              <Label htmlFor="target_lang">Idioma destino</Label>
              <Input
                id="target_lang"
                placeholder="es"
                {...form.register('target_lang')}
                className="mt-1.5 font-mono"
              />
            </div>
          </div>
        </CardContent>
        <Separator />
        <div className="flex justify-end gap-2 p-4">
          <Button type="submit" disabled={!isDirty || updateMut.isPending}>
            {updateMut.isPending ? <Loader2 size={14} className="animate-spin" /> : <Save size={14} />}
            Guardar
          </Button>
        </div>
      </Card>
    </form>
  )
}

function TtsSection({ settings }) {
  const updateMut = useUpdateSettings()
  const form = useForm({
    resolver: zodResolver(ttsSchema),
    defaultValues: {
      tts_engine: settings?.tts_engine || 'elevenlabs',
      elevenlabs_voice_id: settings?.elevenlabs_voice_id || '',
      elevenlabs_model_id: settings?.elevenlabs_model_id || 'eleven_multilingual_v2',
      piper_model_path: settings?.piper_model_path || '/models/piper/es_ES-sharvard-medium.onnx',
      kokoro_voice: settings?.kokoro_voice || 'em_alex',
      s2_voice_profile: settings?.s2_voice_profile || 'voice_martin_osborne_24k.wav',
      s2_ref_text:
        settings?.s2_ref_text ||
        'nunca te olvidé, nunca, el último beso que me diste todavía está grabado en mi corazón, por el día todo es más fácil. pero, todavía sueño contigo.',
      s2_temperature: settings?.s2_temperature ?? 0.8,
      s2_top_p: settings?.s2_top_p ?? 0.8,
      s2_top_k: settings?.s2_top_k ?? 30,
      s2_max_tokens: settings?.s2_max_tokens ?? 1024,
    },
  })

  const engine = form.watch('tts_engine')
  const kokoroVoice = form.watch('kokoro_voice')
  const currentS2Voice = form.watch('s2_voice_profile')

  // Load available voice WAVs from dubbing-generator. Each entry carries
  // the transcript sidecar so changing the dropdown can pre-fill the
  // reference text — no more retyping per voice.
  const [voices, setVoices] = useState([])
  const [voicesLoading, setVoicesLoading] = useState(false)
  const [savingTranscript, setSavingTranscript] = useState(false)
  const reloadVoices = async () => {
    setVoicesLoading(true)
    try {
      const r = await http.get('/dubbing/voices')
      setVoices(Array.isArray(r?.voices) ? r.voices : [])
    } catch (e) {
      // Backend may be down at first paint — silent fail; user can retry.
    } finally {
      setVoicesLoading(false)
    }
  }
  useEffect(() => {
    if (engine === 's2pro') reloadVoices()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [engine])

  // First-load convenience: if the saved settings have an empty
  // s2_ref_text but the selected voice has a sidecar transcript, fill
  // it in. We only do this when the field is genuinely empty so we
  // never overwrite text the user typed in but hasn't saved.
  useEffect(() => {
    if (engine !== 's2pro') return
    const cur = form.getValues('s2_ref_text')
    if (cur && cur.trim().length > 0) return
    const v = voices.find((x) => x.id === currentS2Voice)
    if (v && v.transcript) {
      form.setValue('s2_ref_text', v.transcript, { shouldDirty: false })
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [voices, currentS2Voice, engine])

  // When the user switches voices, auto-fill the reference transcript
  // from the sidecar (if any) so they don't have to retype it.
  const onVoiceChange = (filename) => {
    form.setValue('s2_voice_profile', filename, { shouldDirty: true })
    const v = voices.find((x) => x.id === filename)
    if (v && typeof v.transcript === 'string' && v.transcript.length > 0) {
      form.setValue('s2_ref_text', v.transcript, { shouldDirty: true })
    }
  }

  const saveTranscriptForCurrentVoice = async () => {
    const filename = form.getValues('s2_voice_profile')
    const transcript = form.getValues('s2_ref_text') || ''
    if (!filename) {
      toast.error('Selecciona primero un perfil de voz')
      return
    }
    setSavingTranscript(true)
    try {
      await http.put(
        `/dubbing/voices/${encodeURIComponent(filename)}/transcript`,
        { transcript },
      )
      toast.success('Transcripción guardada junto al WAV')
      // Refresh local cache so future selections see the new value.
      await reloadVoices()
    } catch (e) {
      toast.error('No se pudo guardar la transcripción', {
        description: e?.message || 'Error',
      })
    } finally {
      setSavingTranscript(false)
    }
  }

  const onSubmit = async (values) => {
    try {
      const payload = {
        tts_engine: values.tts_engine,
        elevenlabs_voice_id: values.elevenlabs_voice_id || null,
        elevenlabs_model_id: values.elevenlabs_model_id || null,
        piper_model_path: values.piper_model_path || null,
        kokoro_voice: values.kokoro_voice || null,
        s2_voice_profile: values.s2_voice_profile || null,
        s2_ref_text: values.s2_ref_text || null,
        s2_temperature: values.s2_temperature,
        s2_top_p: values.s2_top_p,
        s2_top_k: values.s2_top_k,
        s2_max_tokens: values.s2_max_tokens,
      }
      await updateMut.mutateAsync(payload)
      toast.success('TTS guardado')
      form.reset(values)
    } catch (e) {
      toast.error(e?.message || 'Error al guardar')
    }
  }

  const { isDirty } = form.formState

  return (
    <form onSubmit={form.handleSubmit(onSubmit)}>
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Mic2 size={16} className="text-primary" />
            TTS / Doblaje
            {isDirty && <Badge variant="outline" className="ml-2 border-amber-500/40 text-amber-500">sin guardar</Badge>}
          </CardTitle>
          <CardDescription>
            Motor de síntesis para el doblaje. ElevenLabs (cloud, voice cloning, de pago) o Piper (local, voz preset ES, gratis).
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <Label htmlFor="tts_engine">Motor TTS</Label>
            <Select
              value={engine}
              onValueChange={(v) => form.setValue('tts_engine', v, { shouldDirty: true })}
            >
              <SelectTrigger id="tts_engine" className="mt-1.5">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="s2pro">S2-Pro (local, clonación de voz, Vulkan)</SelectItem>
                <SelectItem value="elevenlabs">ElevenLabs (cloud, clonación de voz, de pago)</SelectItem>
                <SelectItem value="piper">Piper (local, voz preset ES, gratis, rápido)</SelectItem>
                <SelectItem value="kokoro">Kokoro-82M (local, voz preset ES, GPU, mejor prosodia)</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {engine === 'elevenlabs' && (
            <>
              <div>
                <Label htmlFor="elevenlabs_voice_id">Voice ID</Label>
                <Input
                  id="elevenlabs_voice_id"
                  placeholder="ej: LlZr3QuzbW4WrPjgATHG"
                  {...form.register('elevenlabs_voice_id')}
                  className="mt-1.5 font-mono"
                />
                <p className="text-xs text-muted-foreground mt-1">
                  Voice ID pre-registrado en el dashboard de ElevenLabs (PVC o IVC).
                </p>
              </div>
              <div>
                <Label htmlFor="elevenlabs_model_id">Modelo</Label>
                <Input
                  id="elevenlabs_model_id"
                  placeholder="eleven_multilingual_v2"
                  {...form.register('elevenlabs_model_id')}
                  className="mt-1.5 font-mono"
                />
              </div>
            </>
          )}

          {engine === 'piper' && (
            <div>
              <Label htmlFor="piper_model_path">Path del modelo Piper</Label>
              <Input
                id="piper_model_path"
                placeholder="/models/piper/es_ES-sharvard-medium.onnx"
                {...form.register('piper_model_path')}
                className="mt-1.5 font-mono"
              />
              <p className="text-xs text-muted-foreground mt-1">
                Path absoluto al .onnx dentro del contenedor dubbing-generator. La imagen incluye <code>es_ES-sharvard-medium</code> por defecto.
              </p>
            </div>
          )}

          {engine === 's2pro' && (
            <>
              <div>
                <div className="flex items-end justify-between gap-2">
                  <Label htmlFor="s2_voice_profile">Perfil de voz (archivo en /voices)</Label>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={reloadVoices}
                    disabled={voicesLoading}
                    className="h-7 gap-1 text-xs"
                  >
                    {voicesLoading
                      ? <Loader2 size={12} className="animate-spin" />
                      : <RefreshCw size={12} />}
                    Recargar
                  </Button>
                </div>
                <Select
                  value={currentS2Voice}
                  onValueChange={onVoiceChange}
                >
                  <SelectTrigger id="s2_voice_profile" className="mt-1.5 font-mono">
                    <SelectValue placeholder={
                      voicesLoading
                        ? 'Cargando voces…'
                        : (voices.length === 0 ? 'No se encontraron voces en /voices' : 'Selecciona un perfil')
                    } />
                  </SelectTrigger>
                  <SelectContent>
                    {voices.map((v) => (
                      <SelectItem key={v.id} value={v.id} className="font-mono">
                        {v.id}
                        {v.transcript ? '' : '  (sin transcripción)'}
                      </SelectItem>
                    ))}
                    {/* Fallback: show the current value even if it's not on disk
                        (e.g. user typed a path manually on a previous version).
                        Otherwise the Select would show the placeholder and look
                        like the setting got lost. */}
                    {currentS2Voice && !voices.some((v) => v.id === currentS2Voice) && (
                      <SelectItem value={currentS2Voice} className="font-mono">
                        {currentS2Voice}  (no presente en /voices)
                      </SelectItem>
                    )}
                  </SelectContent>
                </Select>
                <p className="text-xs text-muted-foreground mt-1">
                  WAV (16-24 kHz mono, 5-30 s) en el directorio bind-mounted <code>/voices</code>.
                  Cada voz puede llevar un sidecar <code>.txt</code> con su transcripción.
                </p>
              </div>
              <div>
                <div className="flex items-end justify-between gap-2">
                  <Label htmlFor="s2_ref_text">Transcripción de referencia</Label>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onClick={saveTranscriptForCurrentVoice}
                    disabled={savingTranscript || !currentS2Voice}
                    className="h-7 gap-1 text-xs"
                  >
                    {savingTranscript
                      ? <Loader2 size={12} className="animate-spin" />
                      : <Save size={12} />}
                    Guardar transcripción
                  </Button>
                </div>
                <textarea
                  id="s2_ref_text"
                  rows={3}
                  {...form.register('s2_ref_text')}
                  className="mt-1.5 block w-full rounded-md border border-input bg-background px-3 py-2 text-sm font-mono"
                />
                <p className="text-xs text-muted-foreground mt-1">
                  Debe coincidir EXACTAMENTE con lo que se dice en el WAV. La discrepancia colapsa la calidad de clonación.
                  Al cambiar de voz se autocompleta con su transcripción guardada (si tiene sidecar).
                </p>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <div>
                  <Label htmlFor="s2_temperature">Temperature</Label>
                  <Input
                    id="s2_temperature"
                    type="number" step="0.05" min="0.1" max="1.5"
                    {...form.register('s2_temperature')}
                    className="mt-1.5"
                  />
                </div>
                <div>
                  <Label htmlFor="s2_top_p">Top-P</Label>
                  <Input
                    id="s2_top_p"
                    type="number" step="0.05" min="0.1" max="1.0"
                    {...form.register('s2_top_p')}
                    className="mt-1.5"
                  />
                </div>
                <div>
                  <Label htmlFor="s2_top_k">Top-K</Label>
                  <Input
                    id="s2_top_k"
                    type="number" step="1" min="1" max="200"
                    {...form.register('s2_top_k')}
                    className="mt-1.5"
                  />
                </div>
                <div>
                  <Label htmlFor="s2_max_tokens">Max tokens</Label>
                  <Input
                    id="s2_max_tokens"
                    type="number" step="64" min="128" max="2048"
                    {...form.register('s2_max_tokens')}
                    className="mt-1.5"
                  />
                </div>
              </div>
              <p className="text-xs text-muted-foreground">
                Servidor s2.cpp embebido en el contenedor. Modelo q6_k bind-mounted en <code>/models/s2pro</code>. Licencia non-commercial.
              </p>
            </>
          )}

          {engine === 'kokoro' && (
            <div>
              <Label htmlFor="kokoro_voice">Voz Kokoro</Label>
              <Select
                value={kokoroVoice}
                onValueChange={(v) => form.setValue('kokoro_voice', v, { shouldDirty: true })}
              >
                <SelectTrigger id="kokoro_voice" className="mt-1.5">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="em_alex">em_alex (masculino ES)</SelectItem>
                  <SelectItem value="em_santa">em_santa (masculino ES alternativo)</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground mt-1">
                Kokoro-82M usa GPU para inferencia rápida. Modelo descargado en build.
              </p>
            </div>
          )}
        </CardContent>
        <Separator />
        <div className="flex justify-end gap-2 p-4">
          <Button type="submit" disabled={!isDirty || updateMut.isPending}>
            {updateMut.isPending ? <Loader2 size={14} className="animate-spin" /> : <Save size={14} />}
            Guardar
          </Button>
        </div>
      </Card>
    </form>
  )
}

function OracleSection({ settings }) {
  const updateMut = useUpdateSettings()
  const { data: providers = [] } = useProviders()
  const defaults = settings?.processing_defaults || {}
  const form = useForm({
    resolver: zodResolver(oracleSchema),
    defaultValues: {
      provider_default: defaults.oracle_provider_default || '',
      timeout_seconds: defaults.oracle_timeout_seconds || 30,
    },
  })

  useEffect(() => {
    if (providers.length > 0 && !form.getValues('provider_default')) {
      form.setValue('provider_default', defaults.oracle_provider_default || providers[0].id)
    }
  }, [providers])

  const onSubmit = async (values) => {
    try {
      await updateMut.mutateAsync({
        processing_defaults: {
          ...defaults,
          oracle_provider_default: values.provider_default,
          oracle_timeout_seconds: values.timeout_seconds,
        },
      })
      toast.success('Oracle guardado')
      form.reset(values)
    } catch (e) {
      toast.error(e?.message || 'Error al guardar')
    }
  }

  const { isDirty } = form.formState
  const current = form.watch('provider_default')

  return (
    <form onSubmit={form.handleSubmit(onSubmit)}>
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Sparkles size={16} className="text-primary" />
            Oracle
            {isDirty && <Badge variant="outline" className="ml-2 border-amber-500/40 text-amber-500">sin guardar</Badge>}
          </CardTitle>
          <CardDescription>
            Provider por defecto y timeouts para resolver/scrapear instruccionales.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <Label>Provider por defecto</Label>
            <Select
              value={current || ''}
              onValueChange={(v) => form.setValue('provider_default', v, { shouldDirty: true })}
            >
              <SelectTrigger className="mt-1.5">
                <SelectValue placeholder="Seleccionar provider" />
              </SelectTrigger>
              <SelectContent>
                {providers.map((p) => (
                  <SelectItem key={p.id} value={p.id}>
                    {p.display_name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div>
            <Label htmlFor="timeout_seconds">Timeout (segundos)</Label>
            <Input
              id="timeout_seconds"
              type="number"
              min={5}
              max={300}
              {...form.register('timeout_seconds')}
              className="mt-1.5 font-mono"
            />
          </div>
        </CardContent>
        <Separator />
        <div className="flex justify-end gap-2 p-4">
          <Button type="submit" disabled={!isDirty || updateMut.isPending}>
            {updateMut.isPending ? <Loader2 size={14} className="animate-spin" /> : <Save size={14} />}
            Guardar
          </Button>
        </div>
      </Card>
    </form>
  )
}

function TelegramSection({ settings }) {
  const updateMut = useUpdateSettings()
  const [showHash, setShowHash] = useState(false)

  const form = useForm({
    resolver: zodResolver(telegramSchema),
    defaultValues: {
      telegram_api_id:
        settings?.telegram_api_id != null ? String(settings.telegram_api_id) : '',
      telegram_api_hash: settings?.telegram_api_hash || '',
    },
  })

  const onSubmit = async (values) => {
    try {
      const idRaw = typeof values.telegram_api_id === 'string'
        ? values.telegram_api_id.trim()
        : values.telegram_api_id
      const hashRaw = (values.telegram_api_hash || '').trim()
      const payload = {
        telegram_api_id: idRaw === '' || idRaw == null ? null : Number(idRaw),
        telegram_api_hash: hashRaw === '' ? null : hashRaw,
      }
      await updateMut.mutateAsync(payload)
      toast.success('Telegram guardado')
      form.reset(values)
    } catch (e) {
      toast.error(e?.message || 'Error al guardar')
    }
  }

  const { isDirty } = form.formState
  const savedId = settings?.telegram_api_id
  const savedHash = settings?.telegram_api_hash
  const configured = savedId != null && savedId !== '' && !!savedHash

  return (
    <form onSubmit={form.handleSubmit(onSubmit)}>
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Send size={16} className="text-primary" />
            Telegram
            {configured ? (
              <Badge variant="outline" className="ml-2 border-emerald-500/40 text-emerald-500">
                Configurado
              </Badge>
            ) : (
              <Badge variant="outline" className="ml-2 border-amber-500/40 text-amber-500">
                Incompleto
              </Badge>
            )}
            {isDirty && (
              <Badge variant="outline" className="ml-2 border-amber-500/40 text-amber-500">
                sin guardar
              </Badge>
            )}
          </CardTitle>
          <CardDescription>
            Credenciales de la API de Telegram para integraciones.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <Label htmlFor="telegram_api_id">API ID</Label>
            <Input
              id="telegram_api_id"
              type="number"
              min={1}
              step={1}
              placeholder="123456"
              {...form.register('telegram_api_id')}
              className="mt-1.5 font-mono"
            />
            {form.formState.errors.telegram_api_id && (
              <p className="text-xs text-destructive mt-1">
                {form.formState.errors.telegram_api_id.message}
              </p>
            )}
          </div>

          <div>
            <Label htmlFor="telegram_api_hash">API Hash</Label>
            <div className="relative mt-1.5">
              <Input
                id="telegram_api_hash"
                type={showHash ? 'text' : 'password'}
                placeholder="32 caracteres hex"
                autoComplete="off"
                {...form.register('telegram_api_hash')}
                className="font-mono pr-10"
              />
              <button
                type="button"
                onClick={() => setShowHash((v) => !v)}
                aria-label={showHash ? 'Ocultar hash' : 'Mostrar hash'}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
              >
                {showHash ? <EyeOff size={14} /> : <Eye size={14} />}
              </button>
            </div>
            {form.formState.errors.telegram_api_hash && (
              <p className="text-xs text-destructive mt-1">
                {form.formState.errors.telegram_api_hash.message}
              </p>
            )}
          </div>

          <p className="text-xs text-muted-foreground">
            Obtén tus credenciales en{' '}
            <a
              href="https://my.telegram.org/apps"
              target="_blank"
              rel="noopener noreferrer"
              className="underline hover:text-primary"
            >
              my.telegram.org/apps
            </a>
            . Deja ambos campos vacíos para desactivar la integración.
          </p>
        </CardContent>
        <Separator />
        <div className="flex justify-end gap-2 p-4">
          <Button type="submit" disabled={!isDirty || updateMut.isPending}>
            {updateMut.isPending ? <Loader2 size={14} className="animate-spin" /> : <Save size={14} />}
            Guardar
          </Button>
        </div>
      </Card>
    </form>
  )
}

const OLLAMA_MODELS = ['qwen2.5:7b-instruct-q4_K_M', 'qwen2.5:14b-instruct-q4_K_M']
const OPENAI_MODELS = ['gpt-4o-mini', 'gpt-4o', 'gpt-4.1-mini', 'gpt-4.1']

function TranslationSection({ settings }) {
  const updateMut = useUpdateSettings()
  const [showOpenAI, setShowOpenAI] = useState(false)

  const form = useForm({
    resolver: zodResolver(translationSchema),
    defaultValues: {
      translation_provider: settings?.translation_provider || 'ollama',
      translation_model: settings?.translation_model || 'qwen2.5:7b-instruct-q4_K_M',
      translation_fallback_provider: settings?.translation_fallback_provider || 'openai',
      openai_api_key: settings?.openai_api_key || '',
    },
  })

  const provider = form.watch('translation_provider')

  useEffect(() => {
    const current = form.getValues('translation_model')
    if (provider === 'ollama' && !OLLAMA_MODELS.includes(current)) {
      form.setValue('translation_model', 'qwen2.5:7b-instruct-q4_K_M', { shouldDirty: true })
    } else if (provider === 'openai' && !OPENAI_MODELS.includes(current)) {
      form.setValue('translation_model', 'gpt-4o-mini', { shouldDirty: true })
    }
  }, [provider, form])

  const onSubmit = async (values) => {
    try {
      const raw = (s) => {
        const v = (s || '').trim()
        return v === '' ? null : v
      }
      await updateMut.mutateAsync({
        translation_provider: values.translation_provider,
        translation_model: raw(values.translation_model),
        translation_fallback_provider: values.translation_fallback_provider || null,
        openai_api_key: raw(values.openai_api_key),
      })
      toast.success('Traducción guardada')
      form.reset(values)
    } catch (e) {
      toast.error(e?.message || 'Error al guardar')
    }
  }

  const { isDirty } = form.formState
  const hasOpenAI = !!(settings?.openai_api_key)

  const backends = useBackendsHealth()
  const ollamaStatus = backends.find((b) => b.id === 'ollama')?.status
  const showOllamaWarning = provider === 'ollama' && ollamaStatus === 'down'

  return (
    <form onSubmit={form.handleSubmit(onSubmit)}>
      {showOllamaWarning && (
        <div className="mb-4 rounded-md border border-yellow-500/50 bg-yellow-500/10 p-3 text-sm">
          <p className="font-medium">Ollama no está respondiendo</p>
          <p className="mt-1 text-muted-foreground">
            El modelo se está descargando (~4.5 GB la primera vez) o el servicio no arrancó.
            Mientras tanto puedes cambiar a OpenAI en el selector.
          </p>
        </div>
      )}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Plug size={16} className="text-primary" />
            Traducción
            {isDirty && (
              <Badge variant="outline" className="ml-2 border-amber-500/40 text-amber-500">
                sin guardar
              </Badge>
            )}
          </CardTitle>
          <CardDescription>
            Ollama local (qwen2.5-7b) por defecto, OpenAI como fallback opt-in.
            Términos BJJ (guard, mount, armbar…) se mantienen en inglés.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <Label>Proveedor principal</Label>
              <Select
                value={form.watch('translation_provider')}
                onValueChange={(v) => form.setValue('translation_provider', v, { shouldDirty: true })}
              >
                <SelectTrigger className="mt-1.5"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="ollama">Ollama (local, gratis)</SelectItem>
                  <SelectItem value="openai">OpenAI (cloud, pago)</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div>
              <Label>Fallback</Label>
              <Select
                value={form.watch('translation_fallback_provider') || '__none__'}
                onValueChange={(v) =>
                  form.setValue('translation_fallback_provider', v === '__none__' ? '' : v, { shouldDirty: true })
                }
              >
                <SelectTrigger className="mt-1.5"><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="__none__">Ninguno</SelectItem>
                  <SelectItem value="ollama">Ollama (local)</SelectItem>
                  <SelectItem value="openai">OpenAI (cloud)</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <div>
            <Label htmlFor="translation_model">Modelo</Label>
            {provider === 'ollama' ? (
              <Select
                value={form.watch('translation_model') || 'qwen2.5:7b-instruct-q4_K_M'}
                onValueChange={(v) => form.setValue('translation_model', v, { shouldDirty: true })}
              >
                <SelectTrigger id="translation_model" className="mt-1.5 font-mono">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="qwen2.5:7b-instruct-q4_K_M">qwen2.5:7b-instruct-q4_K_M (default, ~4.5 GB VRAM)</SelectItem>
                  <SelectItem value="qwen2.5:14b-instruct-q4_K_M">qwen2.5:14b-instruct-q4_K_M (~9 GB VRAM, mejor calidad)</SelectItem>
                </SelectContent>
              </Select>
            ) : (
              <Select
                value={form.watch('translation_model') || 'gpt-4o-mini'}
                onValueChange={(v) => form.setValue('translation_model', v, { shouldDirty: true })}
              >
                <SelectTrigger id="translation_model" className="mt-1.5 font-mono">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="gpt-4o-mini">gpt-4o-mini</SelectItem>
                  <SelectItem value="gpt-4o">gpt-4o</SelectItem>
                  <SelectItem value="gpt-4.1-mini">gpt-4.1-mini</SelectItem>
                  <SelectItem value="gpt-4.1">gpt-4.1</SelectItem>
                </SelectContent>
              </Select>
            )}
          </div>

          <div>
            <Label htmlFor="openai_api_key" className="flex items-center gap-2">
              OpenAI API Key
              {hasOpenAI && (
                <Badge variant="outline" className="border-emerald-500/40 text-emerald-500">
                  Configurada
                </Badge>
              )}
            </Label>
            <div className="relative mt-1.5">
              <Input
                id="openai_api_key"
                type={showOpenAI ? 'text' : 'password'}
                placeholder="sk-..."
                autoComplete="off"
                {...form.register('openai_api_key')}
                className="font-mono pr-10"
              />
              <button
                type="button"
                onClick={() => setShowOpenAI((v) => !v)}
                aria-label={showOpenAI ? 'Ocultar key' : 'Mostrar key'}
                className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
              >
                {showOpenAI ? <EyeOff size={14} /> : <Eye size={14} />}
              </button>
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              <a href="https://platform.openai.com/api-keys" target="_blank" rel="noopener noreferrer" className="underline hover:text-primary">
                platform.openai.com/api-keys
              </a>
            </p>
          </div>

        </CardContent>
        <Separator />
        <div className="flex justify-end gap-2 p-4">
          <Button type="submit" disabled={!isDirty || updateMut.isPending}>
            {updateMut.isPending ? <Loader2 size={14} className="animate-spin" /> : <Save size={14} />}
            Guardar
          </Button>
        </div>
      </Card>
    </form>
  )
}

function AuthorAliasesSection({ settings }) {
  const updateMut = useUpdateSettings()
  const initial = settings?.author_aliases || {}
  const [rows, setRows] = useState(() =>
    Object.entries(initial).map(([from, to]) => ({ from, to })),
  )

  const initialSerialized = JSON.stringify(initial)
  const currentSerialized = JSON.stringify(
    Object.fromEntries(
      rows
        .map((r) => [r.from.trim(), r.to.trim()])
        .filter(([a, b]) => a && b),
    ),
  )
  const isDirty = initialSerialized !== currentSerialized

  const update = (i, key, val) => {
    setRows((rs) => rs.map((r, idx) => (idx === i ? { ...r, [key]: val } : r)))
  }
  const add = () => setRows((rs) => [...rs, { from: '', to: '' }])
  const remove = (i) => setRows((rs) => rs.filter((_, idx) => idx !== i))

  const onSave = async () => {
    const payload = Object.fromEntries(
      rows
        .map((r) => [r.from.trim(), r.to.trim()])
        .filter(([a, b]) => a && b),
    )
    try {
      await updateMut.mutateAsync({ author_aliases: payload })
      toast.success('Alias guardados')
    } catch (e) {
      toast.error(e?.message || 'Error al guardar')
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Sparkles size={16} className="text-primary" />
          Alias de autores
          {isDirty && (
            <Badge variant="outline" className="ml-2 border-amber-500/40 text-amber-500">
              sin guardar
            </Badge>
          )}
        </CardTitle>
        <CardDescription>
          Unifica nombres de autor mal parseados. Ej: <code>Powerride</code> → <code>Craig Jones</code>.
          Los grupos de la vista "Por autor" se mergean automáticamente.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        {rows.length === 0 && (
          <p className="text-sm text-muted-foreground">
            Sin alias configurados. Añade uno abajo.
          </p>
        )}
        {rows.map((r, i) => (
          <div key={i} className="flex items-center gap-2">
            <Input
              placeholder="Nombre crudo (ej: Powerride)"
              value={r.from}
              onChange={(e) => update(i, 'from', e.target.value)}
              className="flex-1"
            />
            <span className="text-muted-foreground text-sm">→</span>
            <Input
              placeholder="Canónico (ej: Craig Jones)"
              value={r.to}
              onChange={(e) => update(i, 'to', e.target.value)}
              className="flex-1"
            />
            <Button type="button" variant="ghost" size="sm" onClick={() => remove(i)}>
              <XCircle size={14} />
            </Button>
          </div>
        ))}
        <Button type="button" variant="secondary" size="sm" onClick={add}>
          Añadir alias
        </Button>
      </CardContent>
      <Separator />
      <div className="flex justify-end gap-2 p-4">
        <Button type="button" onClick={onSave} disabled={!isDirty || updateMut.isPending}>
          {updateMut.isPending ? <Loader2 size={14} className="animate-spin" /> : <Save size={14} />}
          Guardar
        </Button>
      </div>
    </Card>
  )
}

function AppearanceSection() {
  const { theme, setTheme } = useTheme()
  const options = useMemo(
    () => [
      { id: 'dark', label: 'Oscuro' },
      { id: 'light', label: 'Claro' },
      { id: 'system', label: 'Sistema' },
    ],
    [],
  )

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Palette size={16} className="text-primary" />
          Apariencia
        </CardTitle>
        <CardDescription>Tema visual de la interfaz.</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-3 gap-2">
          {options.map((o) => (
            <button
              key={o.id}
              type="button"
              onClick={() => setTheme(o.id)}
              aria-pressed={theme === o.id}
              className={`px-3 py-4 rounded-md border text-sm transition-colors ${
                theme === o.id
                  ? 'border-primary bg-primary/10 text-primary'
                  : 'border-border hover:bg-muted'
              }`}
            >
              {o.label}
            </button>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

function AdvancedSection({ settings }) {
  const updateMut = useUpdateSettings()
  const [importing, setImporting] = useState(false)

  const exportConfig = () => {
    try {
      const blob = new Blob([JSON.stringify(settings || {}, null, 2)], {
        type: 'application/json',
      })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = 'bjj-settings.json'
      a.click()
      URL.revokeObjectURL(url)
      toast.success('Configuración exportada')
    } catch (e) {
      toast.error(e?.message || 'Error exportando')
    }
  }

  const importConfig = async (file) => {
    if (!file) return
    setImporting(true)
    try {
      const text = await file.text()
      const data = JSON.parse(text)
      await updateMut.mutateAsync(data)
      toast.success('Configuración importada')
    } catch (e) {
      toast.error(e?.message || 'JSON inválido')
    } finally {
      setImporting(false)
    }
  }

  const resetConfig = async () => {
    if (!window.confirm('¿Restablecer configuración a valores por defecto? Esta acción no se puede deshacer.')) return
    try {
      await http.put('/settings', {
        library_path: '',
        voice_profile_default: null,
        processing_defaults: {},
        custom_prompts: {},
      })
      toast.success('Configuración restablecida')
      // soft reload to refetch
      window.location.reload()
    } catch (e) {
      toast.error(e?.message || 'Error al restablecer')
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Shield size={16} className="text-primary" />
          Avanzado
        </CardTitle>
        <CardDescription>Exportar/importar configuración o restablecer.</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex flex-wrap gap-2">
          <Button type="button" variant="secondary" onClick={exportConfig}>
            <Download size={14} />
            Exportar config
          </Button>

          <label className="inline-flex items-center">
            <input
              type="file"
              accept="application/json"
              className="hidden"
              onChange={(e) => {
                importConfig(e.target.files?.[0])
                e.target.value = ''
              }}
            />
            <span className="inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium h-9 px-4 py-2 bg-secondary text-secondary-foreground hover:bg-secondary/80 cursor-pointer">
              {importing ? <Loader2 size={14} className="animate-spin" /> : <Upload size={14} />}
              Importar config
            </span>
          </label>

          <Button type="button" variant="destructive" onClick={resetConfig}>
            <RotateCcw size={14} />
            Restablecer
          </Button>
        </div>
        <p className="text-xs text-muted-foreground">
          Importar sobreescribe los valores actuales. Restablecer vacía biblioteca, perfil de voz y
          prompts personalizados.
        </p>
      </CardContent>
    </Card>
  )
}

function MaintenanceSection() {
  const { data: settings } = useSettings()
  const libraryPath = settings?.library_path || ''
  const [busy, setBusy] = useState(false)
  const [restarting, setRestarting] = useState(false)
  const [restartingDubbing, setRestartingDubbing] = useState(false)
  const [pulling, setPulling] = useState(false)
  const [pullPct, setPullPct] = useState(null)
  const [pullStatus, setPullStatus] = useState('')

  const clearLocks = async () => {
    setBusy(true)
    try {
      const res = await http.post('/subtitles/maintenance/clear-locks', {})
      const removed = res?.removed ?? 0
      if (removed > 0) toast.success(`Locks eliminados: ${removed}`)
      else toast('Nada que limpiar', { description: 'No había locks residuales.' })
      if (res?.errors?.length) {
        toast.error(`Errores: ${res.errors.length}`, { description: res.errors[0] })
      }
    } catch (e) {
      toast.error('Falló la limpieza', { description: e?.message || 'Error' })
    } finally {
      setBusy(false)
    }
  }

  const pullOllamaModel = async () => {
    setPulling(true)
    setPullPct(0)
    setPullStatus('Conectando…')
    const model = settings?.translation_model || 'qwen2.5:7b-instruct-q4_K_M'
    try {
      const resp = await fetch(`/api/pipeline/pull-ollama-model`, { method: 'POST' })
      if (!resp.ok) { toast.error('Error al iniciar descarga'); return }
      const reader = resp.body.getReader()
      const decoder = new TextDecoder()
      let buf = ''
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buf += decoder.decode(value, { stream: true })
        const lines = buf.split('\n')
        buf = lines.pop()
        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          try {
            const data = JSON.parse(line.slice(6))
            if (data.status === 'error') {
              toast.error('Error descargando', { description: data.error })
              return
            }
            if (data.status === 'success') {
              setPullPct(100)
              setPullStatus('¡Descarga completa!')
              toast.success('Modelo descargado', { description: model })
              return
            }
            if (data.pct != null) setPullPct(data.pct)
            if (data.status) setPullStatus(data.status)
          } catch { /* ignore malformed */ }
        }
      }
    } catch (e) {
      toast.error('Error al descargar', { description: e?.message || 'Error' })
    } finally {
      setPulling(false)
      setTimeout(() => { setPullPct(null); setPullStatus('') }, 3000)
    }
  }

  const restartSubtitles = async () => {
    setRestarting(true)
    try {
      await http.post('/subtitles/maintenance/restart', {})
      toast.success('Subtitle Generator reiniciando…', {
        description: 'Volverá a estar disponible en ~15 segundos.',
      })
    } catch (e) {
      // El servicio se cae antes de responder — es esperado
      toast.success('Subtitle Generator reiniciando…', {
        description: 'Volverá a estar disponible en ~15 segundos.',
      })
    } finally {
      setTimeout(() => setRestarting(false), 15000)
    }
  }

  const restartDubbing = async () => {
    setRestartingDubbing(true)
    try {
      await http.post('/dubbing/maintenance/restart', {})
      toast.success('Dubbing Generator reiniciando…', {
        description: 'Volverá a estar disponible en ~15 segundos.',
      })
    } catch (e) {
      toast.success('Dubbing Generator reiniciando…', {
        description: 'Volverá a estar disponible en ~15 segundos.',
      })
    } finally {
      setTimeout(() => setRestartingDubbing(false), 15000)
    }
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Wrench size={16} />
            Mantenimiento del sistema
          </CardTitle>
          <CardDescription>
            Operaciones puntuales para recuperar el sistema cuando un proceso queda atascado.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="rounded-md border border-border/60 p-4 space-y-3">
            <div className="flex items-start justify-between gap-4">
              <div className="min-w-0">
                <p className="text-sm font-medium">Descargar modelo Ollama</p>
                <p className="text-xs text-muted-foreground mt-1">
                  Descarga el modelo configurado en Traducción ({settings?.translation_model || 'qwen2.5:7b-instruct-q4_K_M'}) si no está presente.
                  Necesario antes del primer pipeline con Ollama. Tarda varios minutos (~4.5 GB).
                </p>
              </div>
              <Button onClick={pullOllamaModel} disabled={pulling} variant="outline" size="sm" className="shrink-0">
                {pulling ? <Loader2 className="mr-2 animate-spin" size={14} /> : <Download className="mr-2" size={14} />}
                {pulling ? 'Descargando…' : 'Descargar'}
              </Button>
            </div>
            {pullPct != null && (
              <div className="space-y-1">
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span className="truncate">{pullStatus}</span>
                  <span className="shrink-0 ml-2">{pullPct.toFixed(1)}%</span>
                </div>
                <Progress value={pullPct} className="h-1.5" />
              </div>
            )}
          </div>
          <div className="flex items-start justify-between gap-4 rounded-md border border-border/60 p-4">
            <div className="min-w-0">
              <p className="text-sm font-medium">Liberar VRAM (reiniciar Subtitle Generator)</p>
              <p className="text-xs text-muted-foreground mt-1">
                Reinicia el servicio de subtítulos para liberar la VRAM tras un OOM o cancelación de job.
                El contenedor vuelve solo en ~15 segundos.
              </p>
            </div>
            <Button onClick={restartSubtitles} disabled={restarting} variant="outline" size="sm" className="shrink-0">
              {restarting ? <Loader2 className="mr-2 animate-spin" size={14} /> : <RefreshCw className="mr-2" size={14} />}
              {restarting ? 'Reiniciando…' : 'Reiniciar'}
            </Button>
          </div>
          <div className="flex items-start justify-between gap-4 rounded-md border border-border/60 p-4">
            <div className="min-w-0">
              <p className="text-sm font-medium">Liberar VRAM (reiniciar Dubbing Generator)</p>
              <p className="text-xs text-muted-foreground mt-1">
                Reinicia el servicio de doblaje para liberar la VRAM tras un OOM o cancelación de job.
                Mata también el server S2-Pro residente. El contenedor vuelve solo en ~15 segundos.
              </p>
            </div>
            <Button onClick={restartDubbing} disabled={restartingDubbing} variant="outline" size="sm" className="shrink-0">
              {restartingDubbing ? <Loader2 className="mr-2 animate-spin" size={14} /> : <RefreshCw className="mr-2" size={14} />}
              {restartingDubbing ? 'Reiniciando…' : 'Reiniciar'}
            </Button>
          </div>
          <div className="flex items-start justify-between gap-4 rounded-md border border-border/60 p-4">
            <div className="min-w-0">
              <p className="text-sm font-medium">Limpiar locks de HuggingFace</p>
              <p className="text-xs text-muted-foreground mt-1">
                Elimina los ficheros <code>.lock</code> residuales del cache del modelo Whisper.
                Útil si al relanzar un pipeline de subtítulos se queda esperando indefinidamente.
              </p>
            </div>
            <Button onClick={clearLocks} disabled={busy} variant="outline" size="sm">
              {busy ? <Loader2 className="mr-2 animate-spin" size={14} /> : <Wrench className="mr-2" size={14} />}
              Limpiar
            </Button>
          </div>
        </CardContent>
      </Card>

      <CleanupSubsection libraryPath={libraryPath} />
      <DuplicatesSubsection libraryPath={libraryPath} />
    </div>
  )
}

// --- Cleanup ---

const CLEANUP_CATEGORY_META = {
  orphan_srt: { label: 'Subtítulos huérfanos', reason: 'Sin vídeo hermano' },
  old_dubbed:  { label: 'Doblajes obsoletos',  reason: '_DOBLADO más antiguo que .ES.srt' },
  temp_files:  { label: 'Ficheros temporales', reason: '.tmp/.part/.bak/~' },
  empty_dirs:  { label: 'Directorios vacíos',  reason: 'Sin archivos ni subdirectorios' },
}
const CLEANUP_ORDER = ['orphan_srt', 'old_dubbed', 'temp_files', 'empty_dirs']

function CleanupSubsection({ libraryPath }) {
  const [jobId, setJobId] = useState(null)
  const [selected, setSelected] = useState(() => new Set())

  const scanMut = useCleanupScan()
  const applyMut = useCleanupApply()
  const jobQ = useCleanupJob(jobId)

  const job = jobQ.data
  const status = job?.status
  const isScanning = ['running', 'pending', 'queued'].includes(status)
  const result = status === 'completed' ? job?.result : null

  useEffect(() => {
    if (status === 'failed') toast.error(job?.error || 'El escaneo falló')
  }, [status, job?.error])

  const sizeByPath = useMemo(() => {
    const m = new Map()
    if (!result) return m
    for (const cat of Object.values(result.categories || {}))
      for (const it of cat) m.set(it.path, it.size || 0)
    return m
  }, [result])

  const selectedBytes = useMemo(() => {
    let t = 0
    for (const p of selected) t += sizeByPath.get(p) || 0
    return t
  }, [selected, sizeByPath])

  const doScan = async () => {
    if (!libraryPath) { toast.error('Configura library_path antes de escanear'); return }
    setSelected(new Set())
    try {
      const res = await scanMut.mutateAsync({ path: libraryPath })
      setJobId(res?.job_id || null)
    } catch (e) { toast.error(e?.message || 'No se pudo iniciar el escaneo') }
  }

  const toggle = (p) => setSelected((prev) => {
    const next = new Set(prev)
    next.has(p) ? next.delete(p) : next.add(p)
    return next
  })

  const toggleCategory = (cat, allChecked) => {
    const items = result?.categories?.[cat] || []
    setSelected((prev) => {
      const next = new Set(prev)
      if (allChecked) for (const it of items) next.delete(it.path)
      else for (const it of items) next.add(it.path)
      return next
    })
  }

  const doApply = async () => {
    if (selected.size === 0) return
    try {
      const res = await applyMut.mutateAsync({ paths: Array.from(selected), dryRun: false })
      toast.success(`Borrados ${res?.deleted?.length ?? 0} · liberados ${formatBytes(res?.freed_bytes || 0)}`)
      if (res?.errors?.length) toast.error(`${res.errors.length} errores al borrar`)
      setSelected(new Set())
      doScan()
    } catch (e) { toast.error(e?.message || 'Error aplicando borrado') }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Trash2 size={16} className="text-primary" /> Limpieza de biblioteca
        </CardTitle>
        <CardDescription>
          Encuentra artefactos borrables: subtítulos huérfanos, doblajes obsoletos, temporales y carpetas vacías.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap items-center gap-2">
          <Button onClick={doScan} disabled={isScanning || scanMut.isPending} variant="outline" size="sm">
            {isScanning || scanMut.isPending
              ? <Loader2 className="mr-2 h-3.5 w-3.5 animate-spin" />
              : <RefreshCw className="mr-2 h-3.5 w-3.5" />}
            Escanear
          </Button>
          <AlertDialog>
            <AlertDialogTrigger asChild>
              <Button variant="destructive" size="sm" disabled={selected.size === 0 || applyMut.isPending}>
                {applyMut.isPending
                  ? <Loader2 className="mr-2 h-3.5 w-3.5 animate-spin" />
                  : <Trash2 className="mr-2 h-3.5 w-3.5" />}
                Borrar selección ({selected.size})
              </Button>
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle className="flex items-center gap-2">
                  <AlertTriangle className="h-5 w-5 text-red-500" /> Confirmar borrado
                </AlertDialogTitle>
                <AlertDialogDescription>
                  Se eliminarán <strong>{selected.size}</strong> elementos, liberando{' '}
                  <strong>{formatBytes(selectedBytes)}</strong>. Irreversible.
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Cancelar</AlertDialogCancel>
                <AlertDialogAction onClick={doApply}>Borrar</AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
          {result && (
            <span className="text-xs text-muted-foreground ml-auto">
              {result.total_items} items · {formatBytes(result.total_bytes)} detectados
            </span>
          )}
        </div>

        {isScanning && (
          <div className="space-y-1.5">
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>Escaneando…{job?.message ? ` ${job.message}` : ''}</span>
              <span>{job?.progress != null ? `${Math.round(job.progress)}%` : ''}</span>
            </div>
            <Progress value={job?.progress ?? 0} />
          </div>
        )}

        {result && (
          <Accordion type="multiple" defaultValue={CLEANUP_ORDER}>
            {CLEANUP_ORDER.map((cat) => {
              const meta = CLEANUP_CATEGORY_META[cat]
              const items = result.categories?.[cat] || []
              const allChecked = items.length > 0 && items.every((it) => selected.has(it.path))
              const someChecked = items.some((it) => selected.has(it.path))
              const catBytes = items.reduce((s, it) => s + (it.size || 0), 0)
              return (
                <AccordionItem key={cat} value={cat}>
                  <AccordionTrigger>
                    <div className="flex items-center gap-3">
                      <span className="font-medium">{meta.label}</span>
                      <Badge variant="secondary" className="text-[10px]">{items.length}</Badge>
                      <span className="text-xs text-muted-foreground">{formatBytes(catBytes)}</span>
                    </div>
                  </AccordionTrigger>
                  <AccordionContent>
                    {items.length === 0 ? (
                      <p className="py-2 text-sm text-muted-foreground">Sin items.</p>
                    ) : (
                      <>
                        <div className="mb-2 flex items-center gap-3 border-b pb-2 text-xs text-muted-foreground">
                          <Checkbox
                            checked={allChecked}
                            indeterminate={!allChecked && someChecked}
                            onCheckedChange={() => toggleCategory(cat, allChecked)}
                          />
                          <span>{allChecked ? 'Deseleccionar todo' : 'Seleccionar todo'} · {meta.reason}</span>
                        </div>
                        <ul className="divide-y">
                          {items.map((it) => (
                            <li key={it.path} className="flex items-center gap-3 py-2 text-sm">
                              <Checkbox checked={selected.has(it.path)} onCheckedChange={() => toggle(it.path)} />
                              <span className="flex-1 truncate font-mono text-xs">{it.path}</span>
                              <span className="w-20 text-right text-xs text-muted-foreground">{formatBytes(it.size)}</span>
                            </li>
                          ))}
                        </ul>
                      </>
                    )}
                  </AccordionContent>
                </AccordionItem>
              )
            })}
          </Accordion>
        )}

        {!result && !isScanning && (
          <p className="text-sm text-muted-foreground">
            Pulsa <strong>Escanear</strong> para buscar artefactos en{' '}
            <code className="text-xs">{libraryPath || '(sin library_path)'}</code>.
          </p>
        )}
      </CardContent>
    </Card>
  )
}

// --- Duplicados ---

function DuplicatesSubsection({ libraryPath }) {
  const [jobId, setJobId] = useState(null)
  const [selections, setSelections] = useState({})

  const scanMut = useDuplicatesScan()
  const applyMut = useCleanupApply()
  const jobQ = useDuplicatesJob(jobId)

  const job = jobQ.data
  const status = job?.status
  const isScanning = ['running', 'pending', 'queued'].includes(status)
  const result = status === 'completed' ? job?.result : null
  const isDeep = !!job?.params?.deep

  useEffect(() => {
    if (status === 'failed') toast.error(job?.error || 'El escaneo falló')
  }, [status, job?.error])

  const doScan = async (deep) => {
    if (!libraryPath) { toast.error('Configura library_path antes de escanear'); return }
    setSelections({})
    try {
      const res = await scanMut.mutateAsync({ path: libraryPath, deep })
      setJobId(res?.job_id || null)
    } catch (e) { toast.error(e?.message || 'No se pudo iniciar el escaneo') }
  }

  const setKeep = (idx, path, group) => setSelections((prev) => {
    const next = { ...prev }
    next[idx] = { keep: path, remove: new Set(group.map((e) => e.path).filter((p) => p !== path)) }
    return next
  })

  const toggleRemove = (idx, path) => setSelections((prev) => {
    const cur = prev[idx] || { keep: null, remove: new Set() }
    const remove = new Set(cur.remove)
    remove.has(path) ? remove.delete(path) : remove.add(path)
    return { ...prev, [idx]: { ...cur, remove } }
  })

  const bulkKeepNewest = () => {
    if (!result) return
    const next = {}
    result.groups.forEach((group, idx) => {
      const newest = [...group].sort((a, b) => (b.mtime || 0) - (a.mtime || 0))[0]
      const keep = newest?.path || group[0]?.path
      next[idx] = { keep, remove: new Set(group.map((e) => e.path).filter((p) => p !== keep)) }
    })
    setSelections(next)
  }

  const bulkKeepShortestPath = () => {
    if (!result) return
    const next = {}
    result.groups.forEach((group, idx) => {
      const shortest = [...group].sort((a, b) => (a.path?.length || 0) - (b.path?.length || 0))[0]
      const keep = shortest?.path || group[0]?.path
      next[idx] = { keep, remove: new Set(group.map((e) => e.path).filter((p) => p !== keep)) }
    })
    setSelections(next)
  }

  const pathsToDelete = useMemo(() => {
    const out = []
    for (const sel of Object.values(selections))
      if (sel?.remove) for (const p of sel.remove) out.push(p)
    return out
  }, [selections])

  const bytesToFree = useMemo(() => {
    if (!result) return 0
    const byPath = new Map()
    for (const g of result.groups) for (const e of g) byPath.set(e.path, e.size || 0)
    return pathsToDelete.reduce((s, p) => s + (byPath.get(p) || 0), 0)
  }, [pathsToDelete, result])

  const doApply = async () => {
    if (pathsToDelete.length === 0) return
    try {
      const res = await applyMut.mutateAsync({ paths: pathsToDelete, dryRun: false })
      toast.success(`Borrados ${res?.deleted?.length ?? 0} · liberados ${formatBytes(res?.freed_bytes || 0)}`)
      if (res?.errors?.length) toast.error(`${res.errors.length} errores al borrar`)
      setSelections({})
      doScan(isDeep)
    } catch (e) { toast.error(e?.message || 'Error aplicando borrado') }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Copy size={16} className="text-primary" /> Duplicados
        </CardTitle>
        <CardDescription>
          Detecta vídeos repetidos por tamaño + duración. Modo profundo confirma con md5.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap items-center gap-2">
          <Button onClick={() => doScan(false)} disabled={isScanning || scanMut.isPending} variant="outline" size="sm">
            {(isScanning && !isDeep) || scanMut.isPending
              ? <Loader2 className="mr-2 h-3.5 w-3.5 animate-spin" />
              : <ScanSearch className="mr-2 h-3.5 w-3.5" />}
            Scan rápido
          </Button>
          <Button onClick={() => doScan(true)} disabled={isScanning || scanMut.isPending} variant="outline" size="sm">
            {isScanning && isDeep
              ? <Loader2 className="mr-2 h-3.5 w-3.5 animate-spin" />
              : <ScanSearch className="mr-2 h-3.5 w-3.5" />}
            Scan profundo (md5)
          </Button>
          {isScanning && (
            <Button onClick={() => setJobId(null)} variant="ghost" size="sm">
              <Ban className="mr-2 h-3.5 w-3.5" /> Abortar
            </Button>
          )}
        </div>

        {isScanning && (
          <div className="space-y-1.5">
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>Escaneando{isDeep ? ' (deep)' : ''}…{job?.message ? ` ${job.message}` : ''}</span>
              <span>{job?.progress != null ? `${Math.round(job.progress)}%` : ''}</span>
            </div>
            <Progress value={job?.progress ?? 0} />
          </div>
        )}

        {result && (
          <>
            <div className="flex flex-wrap items-center justify-between gap-2 rounded-md border border-border/60 p-3">
              <div className="text-sm">
                <span className="text-foreground">{result.stats.total_videos} vídeos · {result.stats.groups_found} grupos</span>
                <span className="ml-2 text-xs text-muted-foreground">
                  Espacio desperdiciado: <strong>{formatBytes(result.stats.wasted_bytes)}</strong>
                </span>
              </div>
              <div className="flex flex-wrap items-center gap-2">
                <Button size="sm" variant="outline" onClick={bulkKeepNewest}>
                  <Clock className="mr-1.5 h-3.5 w-3.5" /> Más reciente
                </Button>
                <Button size="sm" variant="outline" onClick={bulkKeepShortestPath}>
                  <Route className="mr-1.5 h-3.5 w-3.5" /> Path corto
                </Button>
                <AlertDialog>
                  <AlertDialogTrigger asChild>
                    <Button size="sm" variant="destructive" disabled={pathsToDelete.length === 0 || applyMut.isPending}>
                      {applyMut.isPending
                        ? <Loader2 className="mr-2 h-3.5 w-3.5 animate-spin" />
                        : <Trash2 className="mr-2 h-3.5 w-3.5" />}
                      Borrar ({pathsToDelete.length})
                    </Button>
                  </AlertDialogTrigger>
                  <AlertDialogContent>
                    <AlertDialogHeader>
                      <AlertDialogTitle className="flex items-center gap-2">
                        <AlertTriangle className="h-5 w-5 text-red-500" /> Confirmar borrado
                      </AlertDialogTitle>
                      <AlertDialogDescription>
                        Se eliminarán <strong>{pathsToDelete.length}</strong> duplicados,
                        liberando <strong>{formatBytes(bytesToFree)}</strong>.
                      </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                      <AlertDialogCancel>Cancelar</AlertDialogCancel>
                      <AlertDialogAction onClick={doApply}>Borrar</AlertDialogAction>
                    </AlertDialogFooter>
                  </AlertDialogContent>
                </AlertDialog>
              </div>
            </div>

            {result.groups.length === 0 ? (
              <p className="text-sm text-muted-foreground">Sin duplicados encontrados.</p>
            ) : (
              <Accordion type="multiple">
                {result.groups.map((group, idx) => {
                  const sel = selections[idx] || { keep: null, remove: new Set() }
                  const first = group[0] || {}
                  return (
                    <AccordionItem key={idx} value={`g-${idx}`}>
                      <AccordionTrigger>
                        <div className="flex flex-wrap items-center gap-3">
                          <span className="font-medium">Grupo #{idx + 1}</span>
                          <Badge variant="secondary" className="text-[10px]">{group.length} copias</Badge>
                          <span className="text-xs text-muted-foreground">
                            {formatBytes(first.size)} · {formatDuration(first.duration_sec)}
                          </span>
                        </div>
                      </AccordionTrigger>
                      <AccordionContent>
                        <div className="overflow-x-auto">
                          <table className="w-full text-sm">
                            <thead>
                              <tr className="border-b text-left text-xs text-muted-foreground">
                                <th className="py-2 pr-2 font-normal">Borrar</th>
                                <th className="py-2 pr-2 font-normal">Mantener</th>
                                <th className="py-2 pr-2 font-normal">Path</th>
                                <th className="py-2 pr-2 text-right font-normal">Size</th>
                                <th className="py-2 pr-2 text-right font-normal">Duración</th>
                                {isDeep && <th className="py-2 pr-2 font-normal">md5</th>}
                              </tr>
                            </thead>
                            <tbody className="divide-y">
                              {group.map((entry) => (
                                <tr key={entry.path}>
                                  <td className="py-2 pr-2">
                                    <Checkbox
                                      checked={!!sel.remove?.has(entry.path)}
                                      disabled={sel.keep === entry.path}
                                      onCheckedChange={() => toggleRemove(idx, entry.path)}
                                    />
                                  </td>
                                  <td className="py-2 pr-2">
                                    <input
                                      type="radio"
                                      name={`keep-${idx}`}
                                      checked={sel.keep === entry.path}
                                      onChange={() => setKeep(idx, entry.path, group)}
                                      className="h-4 w-4 accent-emerald-500"
                                    />
                                  </td>
                                  <td className="py-2 pr-2 break-all font-mono text-xs">{entry.path}</td>
                                  <td className="py-2 pr-2 text-right text-xs text-muted-foreground">{formatBytes(entry.size)}</td>
                                  <td className="py-2 pr-2 text-right text-xs text-muted-foreground">{formatDuration(entry.duration_sec)}</td>
                                  {isDeep && (
                                    <td className="py-2 pr-2 font-mono text-xs text-muted-foreground">
                                      {entry.md5 ? entry.md5.slice(0, 10) : '—'}
                                    </td>
                                  )}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </AccordionContent>
                    </AccordionItem>
                  )
                })}
              </Accordion>
            )}
          </>
        )}

        {!result && !isScanning && (
          <p className="text-sm text-muted-foreground">
            Pulsa <strong>Scan rápido</strong> para buscar duplicados en{' '}
            <code className="text-xs">{libraryPath || '(sin library_path)'}</code>.
          </p>
        )}
      </CardContent>
    </Card>
  )
}

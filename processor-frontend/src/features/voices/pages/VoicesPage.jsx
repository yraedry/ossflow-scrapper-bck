// WIRE_ROUTE_VOICES: /voices → src/features/voices/pages/VoicesPage.jsx
//
// Voice cloning profile manager. Grid of profile cards with audio preview,
// delete (AlertDialog), and "add profile" Dialog with RHF + zod.
//
// NOTE on the upload form: processor-api's POST /api/voice-profiles expects
// `{ video_path, instructor, start_sec, duration }` — it extracts the sample
// from an existing video on disk. There is no multipart file upload endpoint,
// so the spec's "file upload WAV/MP3 max 30s" is translated to the closest
// real-world equivalent: "ruta de vídeo + ventana de extracción (<=30s)".
import { useMemo, useState } from 'react'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Loader2,
  Mic2,
  Plus,
  Trash2,
  Upload,
} from 'lucide-react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/Card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Skeleton } from '@/components/ui/skeleton'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from '@/components/ui/alert-dialog'
import {
  useVoices,
  useUploadVoice,
  useDeleteVoice,
} from '../api/useVoices'

const uploadSchema = z.object({
  instructor: z.string().trim().min(2, 'Nombre demasiado corto'),
  language: z.enum(['es', 'en']),
  video_path: z.string().trim().min(3, 'Ruta requerida'),
  start_sec: z
    .number({ invalid_type_error: 'Número' })
    .min(0, '>=0')
    .default(60),
  duration: z
    .number({ invalid_type_error: 'Número' })
    .positive('> 0')
    .max(30, 'Máximo 30s')
    .default(15),
})

function formatDate(v) {
  if (!v) return null
  const d = typeof v === 'number' ? new Date(v * (v < 1e12 ? 1000 : 1)) : new Date(v)
  if (Number.isNaN(d.getTime())) return null
  return d.toLocaleDateString()
}

function VoiceCard({ profile }) {
  const del = useDeleteVoice()
  const sampleUrl = profile.sample_url || profile.preview_url || null
  const slug = profile.slug || profile.id || profile.name

  const onDelete = async () => {
    const tid = toast.loading(`Eliminando ${profile.name || slug}...`)
    try {
      await del.mutateAsync({ slug })
      toast.success('Perfil eliminado', { id: tid })
    } catch (e) {
      toast.error(`Error: ${e?.message || 'desconocido'}`, { id: tid })
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.15 }}
    >
      <Card className="flex h-full flex-col gap-3 border-border/60 bg-card/60 p-4">
        <div className="flex items-start justify-between">
          <div className="flex h-11 w-11 items-center justify-center rounded-xl border border-primary/30 bg-primary/10 text-primary">
            <Mic2 className="h-5 w-5" />
          </div>

          <AlertDialog>
            <AlertDialogTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                className="text-muted-foreground hover:text-destructive"
                aria-label="Eliminar perfil"
              >
                <Trash2 className="h-4 w-4" />
              </Button>
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>Eliminar perfil de voz</AlertDialogTitle>
                <AlertDialogDescription>
                  Se borrará el perfil <b>{profile.name || slug}</b> y su muestra
                  asociada. Esta acción no se puede deshacer.
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Cancelar</AlertDialogCancel>
                <AlertDialogAction
                  onClick={onDelete}
                  className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
                >
                  Eliminar
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        </div>

        <div className="min-w-0">
          <h3 className="truncate text-sm font-semibold">{profile.name || slug}</h3>
          <div className="mt-0.5 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-xs text-muted-foreground">
            {profile.language ? <span className="uppercase">{profile.language}</span> : null}
            {profile.sample_duration ? (
              <span>{Math.round(profile.sample_duration)}s</span>
            ) : null}
            {formatDate(profile.created_at || profile.mtime) ? (
              <span>{formatDate(profile.created_at || profile.mtime)}</span>
            ) : null}
          </div>
        </div>

        {sampleUrl ? (
          <audio controls preload="none" src={sampleUrl} className="w-full h-9">
            Your browser does not support audio playback.
          </audio>
        ) : (
          <div className="rounded border border-dashed border-border/60 px-2 py-1.5 text-center text-[11px] text-muted-foreground">
            Sin muestra reproducible
          </div>
        )}
      </Card>
    </motion.div>
  )
}

function AddProfileCard({ onClick }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="group flex h-full min-h-[180px] flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-border/60 bg-card/20 p-6 text-muted-foreground transition hover:border-primary/60 hover:bg-primary/5 hover:text-primary"
    >
      <Plus className="h-6 w-6" />
      <span className="text-sm font-medium">Añadir perfil</span>
    </button>
  )
}

function UploadDialog({ open, onOpenChange }) {
  const upload = useUploadVoice()
  const {
    register,
    handleSubmit,
    setValue,
    watch,
    formState: { errors, isSubmitting },
    reset,
  } = useForm({
    resolver: zodResolver(uploadSchema),
    defaultValues: { instructor: '', language: 'es', video_path: '', start_sec: 60, duration: 15 },
  })

  const language = watch('language')

  const onSubmit = async (values) => {
    const tid = toast.loading(`Extrayendo muestra de ${values.instructor}...`)
    try {
      await upload.mutateAsync({
        instructor: values.instructor,
        language: values.language,
        video_path: values.video_path,
        start_sec: values.start_sec,
        duration: values.duration,
      })
      toast.success('Perfil creado', { id: tid })
      reset()
      onOpenChange(false)
    } catch (e) {
      toast.error(`Error: ${e?.message || 'desconocido'}`, { id: tid })
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>Añadir perfil de voz</DialogTitle>
          <DialogDescription>
            Extrae una muestra desde un vídeo para clonar la voz del instructor.
          </DialogDescription>
        </DialogHeader>

        <form onSubmit={handleSubmit(onSubmit)} className="space-y-3">
          <div className="space-y-1.5">
            <Label htmlFor="instructor">Nombre del instructor</Label>
            <Input id="instructor" placeholder="John Danaher" {...register('instructor')} />
            {errors.instructor ? (
              <p className="text-xs text-destructive">{errors.instructor.message}</p>
            ) : null}
          </div>

          <div className="space-y-1.5">
            <Label>Idioma</Label>
            <Select value={language} onValueChange={(v) => setValue('language', v, { shouldValidate: true })}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="es">Español (ES)</SelectItem>
                <SelectItem value="en">English (EN)</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-1.5">
            <Label htmlFor="video_path">Ruta del vídeo fuente</Label>
            <Input
              id="video_path"
              placeholder="Z:\instruccionales\...\video.mp4"
              {...register('video_path')}
            />
            {errors.video_path ? (
              <p className="text-xs text-destructive">{errors.video_path.message}</p>
            ) : null}
            <p className="text-[11px] text-muted-foreground">
              La muestra se extrae en el servidor (sin upload multipart).
            </p>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div className="space-y-1.5">
              <Label htmlFor="start_sec">Inicio (s)</Label>
              <Input
                id="start_sec"
                type="number"
                step="1"
                min={0}
                {...register('start_sec', { valueAsNumber: true })}
              />
              {errors.start_sec ? (
                <p className="text-xs text-destructive">{errors.start_sec.message}</p>
              ) : null}
            </div>
            <div className="space-y-1.5">
              <Label htmlFor="duration">Duración (s, máx 30)</Label>
              <Input
                id="duration"
                type="number"
                step="1"
                min={1}
                max={30}
                {...register('duration', { valueAsNumber: true })}
              />
              {errors.duration ? (
                <p className="text-xs text-destructive">{errors.duration.message}</p>
              ) : null}
            </div>
          </div>

          <DialogFooter className="pt-2">
            <Button type="button" variant="outline" onClick={() => onOpenChange(false)}>
              Cancelar
            </Button>
            <Button type="submit" disabled={isSubmitting || upload.isPending} className="gap-1">
              {isSubmitting || upload.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <Upload className="h-4 w-4" />
              )}
              Extraer muestra
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}

function VoicesSkeletons({ count = 6 }) {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
      {Array.from({ length: count }).map((_, i) => (
        <Card key={i} className="space-y-3 border-border/60 bg-card/40 p-4">
          <Skeleton className="h-11 w-11 rounded-xl" />
          <Skeleton className="h-4 w-3/4" />
          <Skeleton className="h-3 w-1/2" />
          <Skeleton className="h-9 w-full" />
        </Card>
      ))}
    </div>
  )
}

export default function VoicesPage() {
  const { data, isLoading, isError, error, refetch } = useVoices()
  const [dialogOpen, setDialogOpen] = useState(false)

  const profiles = useMemo(() => (Array.isArray(data) ? data : []), [data])

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.18 }}
      className="space-y-5"
    >
      <div className="flex items-center justify-between">
        <div>
          <h1 className="flex items-center gap-2 text-2xl font-semibold">
            <Mic2 className="h-5 w-5 text-primary" />
            Perfiles de voz
          </h1>
          <p className="text-sm text-muted-foreground">
            Muestras de audio para clonación de voz del instructor.
          </p>
        </div>
        <Button onClick={() => setDialogOpen(true)} className="gap-1">
          <Plus className="h-4 w-4" />
          Añadir perfil
        </Button>
      </div>

      {isError ? (
        <div className="rounded-lg border border-destructive/40 bg-destructive/5 p-6 text-sm text-destructive">
          <div className="flex items-center justify-between gap-4">
            <span>Error cargando perfiles: {error?.message || 'desconocido'}</span>
            <Button variant="outline" size="sm" onClick={refetch}>
              Reintentar
            </Button>
          </div>
        </div>
      ) : isLoading ? (
        <VoicesSkeletons />
      ) : profiles.length === 0 ? (
        <div className="rounded-xl border border-dashed border-border/60 bg-card/30 p-12 text-center">
          <Mic2 className="mx-auto h-10 w-10 text-muted-foreground/40" />
          <h3 className="mt-4 text-lg font-semibold">Aún no hay perfiles</h3>
          <p className="mt-1 text-sm text-muted-foreground">
            Extrae el primer perfil desde un vídeo para empezar a doblar.
          </p>
          <Button onClick={() => setDialogOpen(true)} className="mt-5 gap-1">
            <Plus className="h-4 w-4" />
            Añadir perfil
          </Button>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          <AnimatePresence initial={false}>
            {profiles.map((p) => (
              <VoiceCard key={p.slug || p.id || p.name} profile={p} />
            ))}
          </AnimatePresence>
          <AddProfileCard onClick={() => setDialogOpen(true)} />
        </div>
      )}

      <UploadDialog open={dialogOpen} onOpenChange={setDialogOpen} />
    </motion.div>
  )
}

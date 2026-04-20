// Metadata editor for `.bjj-meta.json` using react-hook-form + zod.
// Uses `useInstructionalMetadata` to seed and `useUpdateMetadata` to persist.
import { useEffect } from 'react'
import { useForm } from 'react-hook-form'
import { z } from 'zod'
import { zodResolver } from '@hookform/resolvers/zod'
import { Loader2, Save, AlertTriangle } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Badge } from '@/components/ui/Badge'
import { Skeleton } from '@/components/ui/skeleton'
import { toast } from 'sonner'
import {
  useInstructionalMetadata,
  useUpdateMetadata,
  useVoiceProfiles,
} from '../api/useLibrary'

const schema = z.object({
  instructor: z.string().optional().default(''),
  topic: z.string().optional().default(''),
  synopsis: z.string().optional().default(''),
  year: z
    .union([z.string(), z.number()])
    .optional()
    .transform((v) => (v === '' || v == null ? '' : String(v))),
  tags: z.string().optional().default(''),
  url_bjjfanatics: z.string().url('URL inválida').optional().or(z.literal('')),
  poster_url: z.string().url('URL inválida').optional().or(z.literal('')),
  voice_profile: z.string().optional().default(''),
})

function toFormValues(data) {
  const d = data || {}
  return {
    instructor: d.instructor || '',
    topic: d.topic || '',
    synopsis: d.synopsis || '',
    year: d.year == null ? '' : String(d.year),
    tags: Array.isArray(d.tags) ? d.tags.join(', ') : d.tags || '',
    url_bjjfanatics: d.url_bjjfanatics || '',
    poster_url: d.poster_url || '',
    voice_profile: d.voice_profile || '',
  }
}

function toPayload(values) {
  const tags = (values.tags || '')
    .split(',')
    .map((t) => t.trim())
    .filter(Boolean)
  const yearNum =
    values.year === '' || values.year == null ? null : parseInt(values.year, 10)
  return {
    instructor: values.instructor || '',
    topic: values.topic || '',
    synopsis: values.synopsis || '',
    year: Number.isNaN(yearNum) ? null : yearNum,
    tags,
    url_bjjfanatics: values.url_bjjfanatics || '',
    poster_url: values.poster_url || '',
    voice_profile: values.voice_profile || '',
  }
}

export default function MetadataTab({ instructional }) {
  const name = instructional?.name
  const { data, isLoading, isError, error } = useInstructionalMetadata(name)
  const update = useUpdateMetadata()
  const { data: voices = [] } = useVoiceProfiles()

  const {
    register,
    handleSubmit,
    reset,
    formState: { errors, isDirty },
  } = useForm({
    resolver: zodResolver(schema),
    defaultValues: toFormValues(null),
  })

  useEffect(() => {
    if (data) reset(toFormValues(data))
  }, [data, reset])

  const onSubmit = handleSubmit(async (values) => {
    try {
      await update.mutateAsync({ name, data: toPayload(values) })
      toast.success('Metadatos guardados')
      reset(values)
    } catch (e) {
      toast.error(`Error: ${e.message || 'desconocido'}`)
    }
  })

  if (isLoading) {
    return (
      <div className="space-y-3">
        <Skeleton className="h-10 w-full" />
        <Skeleton className="h-10 w-full" />
        <Skeleton className="h-24 w-full" />
      </div>
    )
  }
  if (isError) {
    return (
      <div
        role="alert"
        className="flex items-center gap-2 rounded-lg border border-red-900/60 bg-red-950/30 p-3 text-sm text-red-300"
      >
        <AlertTriangle className="h-4 w-4" />
        {error?.message || 'Error cargando metadatos'}
      </div>
    )
  }

  return (
    <Card className="border-zinc-800 bg-zinc-950/60">
      <CardHeader className="flex flex-row items-center justify-between pb-3">
        <CardTitle className="text-sm">Metadatos</CardTitle>
        {isDirty && (
          <Badge variant="outline" className="border-amber-500/40 text-amber-400">
            Sin guardar
          </Badge>
        )}
      </CardHeader>
      <CardContent>
        <form onSubmit={onSubmit} className="space-y-4">
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            <div>
              <Label htmlFor="instructor" className="text-xs">Instructor / autor</Label>
              <Input id="instructor" {...register('instructor')} className="mt-1 h-9" />
            </div>
            <div>
              <Label htmlFor="topic" className="text-xs">Tema</Label>
              <Input id="topic" {...register('topic')} className="mt-1 h-9" />
            </div>
            <div>
              <Label htmlFor="year" className="text-xs">Año</Label>
              <Input
                id="year"
                type="number"
                {...register('year')}
                className="mt-1 h-9"
              />
            </div>
            <div>
              <Label htmlFor="tags" className="text-xs">Tags (separados por coma)</Label>
              <Input id="tags" {...register('tags')} className="mt-1 h-9" />
            </div>
            <div className="md:col-span-2">
              <Label htmlFor="url_bjjfanatics" className="text-xs">
                URL BJJFanatics
              </Label>
              <Input
                id="url_bjjfanatics"
                {...register('url_bjjfanatics')}
                placeholder="https://bjjfanatics.com/products/…"
                className="mt-1 h-9 font-mono text-xs"
              />
              {errors.url_bjjfanatics && (
                <p className="mt-1 text-xs text-red-400">
                  {errors.url_bjjfanatics.message}
                </p>
              )}
            </div>
            <div className="md:col-span-2">
              <Label htmlFor="poster_url" className="text-xs">Poster URL</Label>
              <Input
                id="poster_url"
                {...register('poster_url')}
                placeholder="https://…"
                className="mt-1 h-9 font-mono text-xs"
              />
              {errors.poster_url && (
                <p className="mt-1 text-xs text-red-400">
                  {errors.poster_url.message}
                </p>
              )}
            </div>
          </div>

          <div>
            <Label htmlFor="voice_profile" className="text-xs">
              Voz para doblaje
            </Label>
            <select
              id="voice_profile"
              {...register('voice_profile')}
              className="mt-1 h-9 w-full rounded-md border border-zinc-800 bg-zinc-950 px-3 text-sm text-zinc-100 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-amber-500"
            >
              <option value="">Clonar voz del instructor (por defecto)</option>
              {voices.map((v) => (
                <option key={v.id} value={v.id}>
                  {v.id}
                </option>
              ))}
            </select>
            <p className="mt-1 text-[11px] text-zinc-500">
              Añade WAV de narrador ES en{' '}
              <code className="font-mono">dubbing-generator/voices/</code>{' '}
              para eliminar el acento inglés. Vacío = clonar la voz original.
            </p>
          </div>

          <div>
            <Label htmlFor="synopsis" className="text-xs">Sinopsis / notas</Label>
            <textarea
              id="synopsis"
              rows={5}
              {...register('synopsis')}
              className="mt-1 w-full resize-y rounded-md border border-zinc-800 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-amber-500"
            />
          </div>

          <div className="flex items-center gap-2">
            <Button type="submit" disabled={!isDirty || update.isPending}>
              {update.isPending ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <Save className="mr-2 h-4 w-4" />
              )}
              Guardar
            </Button>
            {isDirty && (
              <Button
                type="button"
                variant="outline"
                onClick={() => reset(toFormValues(data))}
              >
                Descartar
              </Button>
            )}
          </div>
        </form>
      </CardContent>
    </Card>
  )
}

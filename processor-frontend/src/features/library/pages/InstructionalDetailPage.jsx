// WIRE_ROUTE_INSTRUCTIONAL_DETAIL: /library/:name → src/features/library/pages/InstructionalDetailPage.jsx
//
// Detalle de instructional: hero (póster + CTAs) + tabs (Capítulos / Pipeline
// / Metadatos / Logs / Oracle). La pestaña activa se persiste en el search
// param `?tab=…` para compartir URL.
import { useMemo, useState } from 'react'
import { Link, useNavigate, useParams, useSearchParams } from 'react-router-dom'
import { ArrowLeft, AlertCircle } from 'lucide-react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'
import { toast } from 'sonner'
import { useInstructional } from '@/features/library/api/useLibrary'
import InstructionalHero from '@/features/library/components/InstructionalHero'
import ChaptersTab from '@/features/library/components/ChaptersTab'
import MetadataTab from '@/features/library/components/MetadataTab'
import OracleTab from '@/features/library/components/OracleTab'
import QaTab from '@/features/library/components/QaTab'

const TABS = [
  { id: 'chapters', label: 'Capítulos' },
  { id: 'qa', label: 'QA doblaje' },
  { id: 'metadata', label: 'Metadatos' },
  { id: 'oracle', label: 'Oracle' },
]

function HeroSkeleton() {
  return (
    <div className="flex flex-col gap-6 rounded-xl border border-zinc-800/80 bg-zinc-950 p-6 md:flex-row md:p-8">
      <Skeleton className="h-[300px] w-[200px] md:w-[280px]" />
      <div className="flex-1 space-y-3">
        <Skeleton className="h-8 w-2/3" />
        <Skeleton className="h-4 w-1/3" />
        <Skeleton className="h-4 w-1/4" />
        <div className="flex gap-2 pt-2">
          <Skeleton className="h-9 w-32" />
          <Skeleton className="h-9 w-24" />
          <Skeleton className="h-9 w-28" />
        </div>
      </div>
    </div>
  )
}

export default function InstructionalDetailPage() {
  const { name } = useParams()
  const nav = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()

  const tab = useMemo(() => {
    const raw = searchParams.get('tab')
    if (raw && TABS.some((t) => t.id === raw)) return raw
    return 'chapters'
  }, [searchParams])

  const setTab = (next) => {
    const sp = new URLSearchParams(searchParams)
    sp.set('tab', next)
    setSearchParams(sp, { replace: true })
  }

  const { data, isLoading, isError, error } = useInstructional(name)
  const [starting, setStarting] = useState(false)

  const goProcessAll = async (steps, extra = {}) => {
    if (!data?.path) return
    setStarting(true)
    try {
      const videos = data?.videos || []
      const seasonMap = new Map()
      for (const v of videos) {
        const key = v.season || 'Sin temporada'
        if (!seasonMap.has(key)) seasonMap.set(key, [])
        seasonMap.get(key).push(v)
      }

      const seasonPaths = []
      for (const [, list] of [...seasonMap.entries()].sort(([a], [b]) => String(a).localeCompare(String(b), undefined, { numeric: true }))) {
        const first = list[0]?.path
        if (!first) continue
        const sep = first.includes('\\') ? '\\' : '/'
        const idx = first.lastIndexOf(sep)
        if (idx > 0) seasonPaths.push(first.slice(0, idx))
      }

      const paths = seasonPaths.length > 0 ? seasonPaths : [data.path]

      const opts = { mode: 'oracle' }
      if (extra?.force) opts.force = true
      const continueOnFail = extra?.continueOnFail !== false

      // Server-side batch: el orquestador vive en processor-api y sobrevive
      // al cierre del navegador. Antes el bucle corría en JS aquí y si el
      // usuario cerraba la pestaña entre seasons, solo la season 1 (ya
      // lanzada al backend) terminaba y las demás nunca arrancaban.
      const res = await fetch('/api/pipeline/batch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: data.name || data.path,
          paths,
          steps,
          options: opts,
          continue_on_fail: continueOnFail,
        }),
      })
      if (!res.ok) {
        let detail = ''
        try {
          const j = await res.json()
          detail = j?.error || j?.detail || ''
        } catch { /* noop */ }
        toast.error(`No se pudo lanzar el batch: ${detail || res.status}`)
        return
      }
      const batch = await res.json()
      try {
        localStorage.setItem(`process_all_batch_${name}`, batch.batch_id)
      } catch { /* noop */ }
      toast.success(
        paths.length > 1
          ? `Batch lanzado: ${paths.length} seasons en cola (sigue corriendo aunque cierres el navegador)`
          : 'Pipeline lanzado'
      )
    } catch (err) {
      toast.error(`Error iniciando pipeline: ${err.message || 'desconocido'}`)
    } finally {
      setStarting(false)
    }
  }
  const goMetadata = () => setTab('metadata')

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="flex items-center gap-1 text-sm text-zinc-500">
          <Link to="/library" className="hover:text-zinc-200">Biblioteca</Link>
        </div>
        <HeroSkeleton />
        <div className="space-y-3">
          <Skeleton className="h-10 w-full max-w-md" />
          <Skeleton className="h-64 w-full" />
        </div>
      </div>
    )
  }

  const notFound =
    isError &&
    (error?.status === 404 ||
      /not found/i.test(error?.message || '') ||
      /404/.test(String(error?.message || '')))

  if (notFound || !data) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-4 py-16 text-center">
        <AlertCircle className="h-10 w-10 text-zinc-600" />
        <div>
          <h2 className="text-lg font-semibold text-zinc-200">
            Instructional no encontrado
          </h2>
          <p className="text-sm text-zinc-500">
            {error?.message || 'Puede que el escaneo no lo haya indexado.'}
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={() => nav('/library')}>
          <ArrowLeft className="mr-1.5 h-3.5 w-3.5" /> Volver a Biblioteca
        </Button>
      </div>
    )
  }

  if (isError) {
    return (
      <div className="rounded-lg border border-red-900/60 bg-red-950/30 p-4 text-sm text-red-300">
        Error cargando instructional: {error?.message || 'desconocido'}
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Breadcrumb */}
      <nav className="flex items-center gap-1.5 text-sm text-zinc-500">
        <Link to="/library" className="hover:text-zinc-200">Biblioteca</Link>
        <span className="text-zinc-700">/</span>
        <span className="truncate text-zinc-300">{data.name}</span>
      </nav>

      <InstructionalHero
        instructional={data}
        onProcessAll={goProcessAll}
        onEditMetadata={goMetadata}
        processingAll={starting}
      />

      <Tabs value={tab} onValueChange={setTab} className="space-y-4">
        <TabsList className="bg-zinc-900/60">
          {TABS.map((t) => (
            <TabsTrigger key={t.id} value={t.id}>
              {t.label}
            </TabsTrigger>
          ))}
        </TabsList>

        <TabsContent value="chapters" className="mt-0">
          <ChaptersTab instructional={data} />
        </TabsContent>
        <TabsContent value="qa" className="mt-0">
          <QaTab instructional={data} />
        </TabsContent>
        <TabsContent value="metadata" className="mt-0">
          <MetadataTab instructional={data} />
        </TabsContent>
        <TabsContent value="oracle" className="mt-0">
          <OracleTab instructional={data} />
        </TabsContent>
      </Tabs>
    </div>
  )
}

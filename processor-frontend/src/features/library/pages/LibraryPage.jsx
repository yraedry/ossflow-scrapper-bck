import { useMemo, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { FolderSearch, AlertTriangle, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card } from '@/components/ui/Card'
import { Skeleton } from '@/components/ui/skeleton'
import { useLibrary, useScanLibrary } from '../api/useLibrary'
import { useSettings } from '@/features/settings/api/useSettings'
import { PosterCard } from '../components/PosterCard'
import { LibraryFilters } from '../components/LibraryFilters'
import { LibraryToolbar } from '../components/LibraryToolbar'
import { toast } from 'sonner'

// TODO: virtualize via @tanstack/react-virtual if list > 200 items.

// Mirror the backend heuristic: "Title - Author" → extract author from name when field is empty.
function resolveAuthor(item) {
  if (item?.author) return item.author
  const name = item?.name || ''
  if (name.includes(' - ')) {
    const parts = name.split(' - ')
    const left = parts[0].trim()
    const right = parts[parts.length - 1].trim()
    const leftWords = left.split(' ').length
    const rightWords = right.split(' ').length
    if (leftWords <= 3 && rightWords > leftWords) return left
    return right
  }
  return ''
}

function groupByAuthor(items) {
  const map = new Map()
  for (const item of items) {
    const author = resolveAuthor(item) || 'Sin autor'
    if (!map.has(author)) map.set(author, [])
    map.get(author).push(item)
  }
  return Array.from(map.entries())
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([author, list]) => ({ author, list }))
}

function isProcessed(item) {
  return (
    (item?.chapters_detected ?? 0) > 0 ||
    (item?.subtitled ?? 0) > 0 ||
    (item?.dubbed ?? 0) > 0
  )
}

function matchesFilter(item, filter) {
  switch (filter) {
    case 'processed':
      return isProcessed(item)
    case 'unprocessed':
      return !isProcessed(item)
    case 'no_poster':
      return !(item?.has_poster || item?.poster_filename)
    default:
      return true
  }
}

function matchesQuery(item, q) {
  if (!q) return true
  const needle = q.toLowerCase()
  return (
    (item?.name || '').toLowerCase().includes(needle) ||
    resolveAuthor(item).toLowerCase().includes(needle)
  )
}

function sortItems(items, sort) {
  const arr = [...items]
  switch (sort) {
    case 'recent':
      arr.sort((a, b) => (b?.mtime || 0) - (a?.mtime || 0))
      break
    case 'size':
      arr.sort((a, b) => (b?.total_videos || 0) - (a?.total_videos || 0))
      break
    case 'name':
    default:
      arr.sort((a, b) => (a?.name || '').localeCompare(b?.name || ''))
  }
  return arr
}

function LibrarySkeletons({ count = 18 }) {
  return (
    <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 2xl:grid-cols-8">
      {Array.from({ length: count }).map((_, i) => (
        <Card key={i} className="overflow-hidden border-border/60">
          <Skeleton className="aspect-[2/3] w-full rounded-none" />
          <div className="space-y-1.5 p-2">
            <Skeleton className="h-3 w-3/4" />
            <Skeleton className="h-2.5 w-1/2" />
          </div>
        </Card>
      ))}
    </div>
  )
}

function EmptyState({ onScan, scanning }) {
  return (
    <div className="rounded-xl border border-dashed border-border/60 bg-card/30 p-12 text-center">
      <FolderSearch className="mx-auto h-12 w-12 text-muted-foreground/50" />
      <h3 className="mt-4 text-lg font-semibold">Biblioteca vacía</h3>
      <p className="mt-1 text-sm text-muted-foreground">
        Configura la ruta en{' '}
        <Link to="/settings" className="text-primary underline">
          Settings
        </Link>{' '}
        y escanea para detectar instructionals.
      </p>
      <Button onClick={onScan} disabled={scanning} className="mt-5">
        {scanning ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
        Escanear biblioteca
      </Button>
    </div>
  )
}

function ErrorState({ error, onRetry }) {
  return (
    <div className="rounded-xl border border-destructive/40 bg-destructive/5 p-10 text-center">
      <AlertTriangle className="mx-auto h-10 w-10 text-destructive" />
      <h3 className="mt-3 text-lg font-semibold">Error cargando biblioteca</h3>
      <p className="mt-1 text-sm text-muted-foreground">
        {error?.message || 'Error desconocido'}
      </p>
      <Button variant="outline" onClick={onRetry} className="mt-5">
        Reintentar
      </Button>
    </div>
  )
}

export default function LibraryPage() {
  const [params] = useSearchParams()
  const q = params.get('q') || ''
  const filter = params.get('filter') || 'all'
  const sort = params.get('sort') || 'name'

  const [view, setView] = useState('grid')
  const { data, isLoading, isError, error, refetch } = useLibrary()
  const { data: settings } = useSettings()
  const scan = useScanLibrary()

  const items = useMemo(() => {
    const list = Array.isArray(data) ? data : []
    return sortItems(
      list.filter((it) => matchesFilter(it, filter) && matchesQuery(it, q)),
      sort,
    )
  }, [data, filter, q, sort])

  const groups = useMemo(() => view === 'author' ? groupByAuthor(items) : [], [items, view])

  const handleScan = async () => {
    const path = settings?.library_path
    if (!path) {
      toast.error('Configura la ruta en Settings antes de escanear.')
      return
    }
    const tid = toast.loading('Escaneando biblioteca...')
    try {
      await scan.mutateAsync({ path })
      toast.success('Biblioteca actualizada', { id: tid })
    } catch (err) {
      toast.error(`Error escaneando: ${err?.message || 'desconocido'}`, { id: tid })
    }
  }

  const totalRaw = Array.isArray(data) ? data.length : 0
  const showEmpty = !isLoading && !isError && totalRaw === 0

  return (
    <div className="space-y-5">
      <LibraryToolbar count={items.length} view={view} onViewChange={setView} />

      {!showEmpty && !isError ? <LibraryFilters /> : null}

      {isLoading ? (
        <LibrarySkeletons />
      ) : isError ? (
        <ErrorState error={error} onRetry={refetch} />
      ) : showEmpty ? (
        <EmptyState onScan={handleScan} scanning={scan.isPending} />
      ) : items.length === 0 ? (
        <div className="rounded-lg border border-border/60 bg-card/30 p-10 text-center text-sm text-muted-foreground">
          Ningún instructional coincide con los filtros actuales.
        </div>
      ) : view === 'list' ? (
        <ul className="divide-y divide-border/60 rounded-lg border border-border/60 bg-card/30">
          {items.map((it) => (
            <li key={it.name} className="px-4 py-2.5">
              <Link
                to={`/library/${encodeURIComponent(it.name)}`}
                className="flex items-center justify-between gap-4 text-sm hover:text-primary"
              >
                <span className="truncate font-medium">{it.name}</span>
                <span className="shrink-0 text-xs text-muted-foreground">
                  {it.total_videos ?? 0} vídeos · {it.chapters_detected ?? 0} cap ·{' '}
                  {it.subtitled ?? 0} subs · {it.dubbed ?? 0} dub
                </span>
              </Link>
            </li>
          ))}
        </ul>
      ) : view === 'author' ? (
        <div className="space-y-8">
          {groups.map(({ author, list }) => (
            <div key={author}>
              <h2 className="mb-3 text-base font-semibold text-foreground/80 border-b border-border/40 pb-1">
                {author}
                <span className="ml-2 text-xs font-normal text-muted-foreground">{list.length}</span>
              </h2>
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 2xl:grid-cols-8">
                {list.map((it) => (
                  <PosterCard key={it.name} instructional={it} />
                ))}
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 2xl:grid-cols-8">
          {items.map((it) => (
            <PosterCard key={it.name} instructional={it} />
          ))}
        </div>
      )}
    </div>
  )
}

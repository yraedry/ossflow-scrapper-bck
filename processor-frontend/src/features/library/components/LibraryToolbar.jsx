import { useEffect, useState } from 'react'
import { Loader2, RefreshCw, LayoutGrid, List, Users } from 'lucide-react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'
import { useScanLibrary } from '../api/useLibrary'
import { useSettings } from '@/features/settings/api/useSettings'

const VIEW_KEY = 'library_view_v3'
const VALID_VIEWS = ['grid', 'list', 'author']

function loadView() {
  try {
    const v = localStorage.getItem(VIEW_KEY)
    return VALID_VIEWS.includes(v) ? v : 'grid'
  } catch {
    return 'grid'
  }
}

export function LibraryToolbar({ count = 0, view, onViewChange }) {
  const [localView, setLocalView] = useState(loadView)
  const effectiveView = view ?? localView
  const setView = (v) => {
    setLocalView(v)
    onViewChange?.(v)
    try {
      localStorage.setItem(VIEW_KEY, v)
    } catch {
      /* noop */
    }
  }

  useEffect(() => {
    // Hydrate parent on mount if it's tracking view externally.
    if (view === undefined) onViewChange?.(localView)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const { data: settings } = useSettings()
  const scan = useScanLibrary()

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

  return (
    <div className="sticky top-0 z-10 flex flex-wrap items-center justify-between gap-3 border-b border-border/60 bg-background/85 py-3 backdrop-blur">
      <div className="flex items-baseline gap-3">
        <h1 className="text-2xl font-bold tracking-tight">Biblioteca</h1>
        <span className="text-sm text-muted-foreground">
          {count} {count === 1 ? 'instructional' : 'instructionals'}
        </span>
      </div>

      <div className="flex items-center gap-2">
        <div
          className="inline-flex rounded-md border border-border bg-card/40 p-0.5"
          role="group"
          aria-label="Vista"
        >
          <button
            type="button"
            aria-pressed={effectiveView === 'grid'}
            aria-label="Vista grid"
            onClick={() => setView('grid')}
            className={cn(
              'inline-flex h-8 w-8 items-center justify-center rounded-sm transition-colors',
              effectiveView === 'grid'
                ? 'bg-primary text-primary-foreground'
                : 'text-muted-foreground hover:bg-accent',
            )}
          >
            <LayoutGrid className="h-4 w-4" />
          </button>
          <button
            type="button"
            aria-pressed={effectiveView === 'list'}
            aria-label="Vista lista"
            onClick={() => setView('list')}
            className={cn(
              'inline-flex h-8 w-8 items-center justify-center rounded-sm transition-colors',
              effectiveView === 'list'
                ? 'bg-primary text-primary-foreground'
                : 'text-muted-foreground hover:bg-accent',
            )}
          >
            <List className="h-4 w-4" />
          </button>
          <button
            type="button"
            aria-pressed={effectiveView === 'author'}
            aria-label="Agrupar por autor"
            onClick={() => setView('author')}
            className={cn(
              'inline-flex h-8 w-8 items-center justify-center rounded-sm transition-colors',
              effectiveView === 'author'
                ? 'bg-primary text-primary-foreground'
                : 'text-muted-foreground hover:bg-accent',
            )}
          >
            <Users className="h-4 w-4" />
          </button>
        </div>

        <Button onClick={handleScan} disabled={scan.isPending} size="sm">
          {scan.isPending ? (
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
          ) : (
            <RefreshCw className="mr-2 h-4 w-4" />
          )}
          Escanear
        </Button>
      </div>
    </div>
  )
}

export default LibraryToolbar

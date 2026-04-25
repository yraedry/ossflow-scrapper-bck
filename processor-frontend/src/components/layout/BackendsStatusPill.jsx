import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import { useBackendsHealth } from './useBackendsHealth'
import { cn } from '@/lib/utils'

const DOT = {
  up: 'bg-emerald-500',
  loading: 'bg-amber-400 animate-pulse',
  down: 'bg-red-500',
}

const STATUS_LABEL = { up: 'en línea', down: 'caído', loading: 'cargando' }

export function BackendsStatusPill() {
  const backends = useBackendsHealth()
  const allUp = backends.every((b) => b.status === 'up')
  const anyDown = backends.some((b) => b.status === 'down')

  // Resumen textual accesible: "Backends: 3 en línea, 1 caído"; así el dot de
  // color deja de ser el único indicador (WCAG 1.4.1 Use of Color).
  const countsByStatus = backends.reduce((acc, b) => {
    acc[b.status] = (acc[b.status] || 0) + 1
    return acc
  }, {})
  const statusSummary = Object.entries(countsByStatus)
    .map(([k, v]) => `${v} ${STATUS_LABEL[k] || k}`)
    .join(', ')
  const ariaLabel = `Backends: ${statusSummary || 'cargando'}`

  return (
    <TooltipProvider delayDuration={150}>
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            type="button"
            aria-label={ariaLabel}
            className={cn(
              'inline-flex items-center gap-2 h-8 px-2.5 rounded-md border border-border bg-background/40 hover:bg-accent text-xs text-muted-foreground transition-colors'
            )}
          >
            <div className="flex items-center gap-1" aria-hidden="true">
              {backends.map((b) => (
                <span key={b.id} className={cn('w-2 h-2 rounded-full', DOT[b.status])} />
              ))}
            </div>
            <span className="hidden sm:inline">
              {allUp ? 'Backends' : anyDown ? 'Backends caídos' : 'Backends…'}
            </span>
          </button>
        </TooltipTrigger>
        <TooltipContent side="bottom" className="min-w-[220px] p-0">
          <div className="py-1.5">
            {backends.map((b) => (
              <div
                key={b.id}
                className="flex items-center justify-between gap-4 px-3 py-1.5 text-xs"
              >
                <div className="flex items-center gap-2">
                  <span
                    className={cn('w-2 h-2 rounded-full', DOT[b.status])}
                    aria-hidden="true"
                  />
                  <span className="font-medium">{b.label}</span>
                </div>
                <span className="text-muted-foreground font-mono">
                  {b.status === 'up' ? 'ok' : b.status === 'loading' ? '…' : 'down'}
                </span>
              </div>
            ))}
          </div>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  )
}

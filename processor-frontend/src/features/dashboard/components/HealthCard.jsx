// HealthCard — backend liveness using useBackendsHealth.
import { useRef } from 'react'
import { ServerCog } from 'lucide-react'
import { toast } from 'sonner'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { cn } from '@/lib/utils'
import { useBackendsHealth } from '@/components/layout/useBackendsHealth'

const TONE = {
  up: { dot: 'bg-emerald-500', text: 'text-emerald-300', label: 'en línea' },
  down: { dot: 'bg-rose-500', text: 'text-rose-300', label: 'fuera de línea' },
  loading: { dot: 'bg-muted-foreground/40', text: 'text-muted-foreground', label: '…' },
}

export function HealthCard({ className }) {
  const backends = useBackendsHealth()
  const startRef = useRef({})

  function onClick(b) {
    const detail =
      b.status === 'up'
        ? `OK · ${b.data?.body?.service || b.id}`
        : b.status === 'down'
          ? b.error || 'Sin respuesta'
          : 'Verificando…'
    toast(b.label, { description: detail })
  }

  return (
    <Card className={cn('h-full', className)}>
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-sm font-semibold">
          <ServerCog className="h-4 w-4 text-sky-400" /> Servicios
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        {backends.map((b) => {
          const tone = TONE[b.status] || TONE.loading
          // Track first time we saw a status to approximate "last check".
          if (!startRef.current[b.id]) startRef.current[b.id] = Date.now()
          return (
            <button
              key={b.id}
              type="button"
              onClick={() => onClick(b)}
              className="flex w-full items-center justify-between gap-3 rounded-lg border bg-card/50 px-3 py-2 text-left transition-colors hover:bg-accent/40 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              <div className="flex min-w-0 items-center gap-2">
                <span
                  className={cn('h-2 w-2 shrink-0 rounded-full', tone.dot, b.status === 'up' && 'animate-pulse')}
                  aria-hidden
                />
                <div className="min-w-0">
                  <div className="truncate text-sm font-medium">{b.label}</div>
                  <div className={cn('text-[10px]', tone.text)}>{tone.label}</div>
                </div>
              </div>
              <div className="text-right text-[10px] text-muted-foreground">
                {b.status === 'up' ? 'funcional' : b.status === 'down' ? 'revisar' : '...'}
              </div>
            </button>
          )
        })}
      </CardContent>
    </Card>
  )
}

export default HealthCard

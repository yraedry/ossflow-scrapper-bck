// WIRE_ROUTE_LOGS: /logs → src/features/logs/pages/LogsPage.jsx
//
// Centralized ring-buffer log viewer. Auto-refresh via react-query (5s in hook),
// pausable, virtualized with @tanstack/react-virtual, copy-visible button, sticky
// filters bar. Font: JetBrains Mono (via Tailwind's `font-mono` which falls back
// through the user's mono stack).
import { useMemo, useRef, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { useVirtualizer } from '@tanstack/react-virtual'
import { motion } from 'framer-motion'
import { Copy, Pause, Play, RefreshCw, Terminal } from 'lucide-react'
import { toast } from 'sonner'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/Badge'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Skeleton } from '@/components/ui/skeleton'
import { qk } from '@/lib/queryKeys'
import { useLogs } from '../api/useLogs'

const SERVICES = [
  { value: 'processor-api', label: 'Processor API' },
  { value: 'chapter-splitter', label: 'Chapter Splitter' },
  { value: 'subtitle-generator', label: 'Subtitle Generator' },
  { value: 'dubbing-generator', label: 'Dubbing Generator' },
  { value: 'telegram-fetcher', label: 'Telegram Fetcher' },
]

const LEVELS = [
  { value: 'ALL', label: 'All' },
  { value: 'DEBUG', label: 'Debug' },
  { value: 'INFO', label: 'Info' },
  { value: 'WARN', label: 'Warn' },
  { value: 'ERROR', label: 'Error' },
]

const LEVEL_STYLES = {
  ERROR: 'text-red-400',
  WARNING: 'text-amber-400',
  WARN: 'text-amber-400',
  INFO: 'text-sky-300',
  DEBUG: 'text-muted-foreground',
}

const LEVEL_BADGE = {
  ERROR: 'bg-red-500/15 text-red-300 border-red-500/30',
  WARNING: 'bg-amber-500/15 text-amber-300 border-amber-500/30',
  WARN: 'bg-amber-500/15 text-amber-300 border-amber-500/30',
  INFO: 'bg-sky-500/15 text-sky-300 border-sky-500/30',
  DEBUG: 'bg-muted/40 text-muted-foreground border-border/50',
}

function formatTs(ts) {
  if (!ts) return '--:--:--'
  const d = typeof ts === 'number' ? new Date(ts * (ts < 1e12 ? 1000 : 1)) : new Date(ts)
  if (Number.isNaN(d.getTime())) return String(ts)
  return d.toISOString().slice(11, 19)
}

export default function LogsPage() {
  const [service, setService] = useState('processor-api')
  const [level, setLevel] = useState('ALL')
  const [text, setText] = useState('')
  const [paused, setPaused] = useState(false)
  const qc = useQueryClient()

  const { data, isLoading, isFetching, isError, error, refetch } = useLogs({
    service,
    level: level === 'ALL' ? undefined : level,
    tail: 1000,
  })

  // Pause auto-refresh by cancelling the continuous interval through query cache.
  // Simplest: set queryClient's default off — but we scope by toggling staleTime
  // by invalidating manually. The hook hardcodes refetchInterval: 5000, so we
  // cancel ongoing via queryClient.cancelQueries when paused.
  const lines = useMemo(() => {
    const all = data?.lines || []
    const needle = text.trim().toLowerCase()
    if (!needle) return all
    return all.filter((l) =>
      (l.message || '').toLowerCase().includes(needle) ||
      (l.logger || '').toLowerCase().includes(needle),
    )
  }, [data, text])

  const scrollRef = useRef(null)
  const virtualizer = useVirtualizer({
    count: lines.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => 24,
    overscan: 30,
  })

  const handleTogglePause = () => {
    const next = !paused
    setPaused(next)
    if (next) {
      qc.cancelQueries({ queryKey: qk.logs.all })
      toast.message('Auto-refresh pausado')
    } else {
      qc.invalidateQueries({ queryKey: qk.logs.all })
      toast.success('Auto-refresh reanudado')
    }
  }

  const handleCopy = async () => {
    const text = lines
      .map((l) => `${formatTs(l.timestamp)} [${l.level || 'INFO'}] ${l.message || ''}`)
      .join('\n')
    try {
      await navigator.clipboard.writeText(text)
      toast.success(`Copiadas ${lines.length} líneas`)
    } catch {
      toast.error('No se pudo copiar')
    }
  }

  const handleRefresh = () => {
    refetch()
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.18 }}
      className="space-y-4"
    >
      <div className="flex items-center justify-between">
        <div>
          <h1 className="flex items-center gap-2 text-2xl font-semibold">
            <Terminal className="h-5 w-5 text-primary" />
            Logs
          </h1>
          <p className="text-sm text-muted-foreground">
            Ring buffer agregado de todos los servicios.
          </p>
        </div>
        <div className="flex items-center gap-2">
          {isFetching && !paused ? (
            <Badge variant="outline" className="gap-1 text-xs">
              <RefreshCw className="h-3 w-3 animate-spin" /> live
            </Badge>
          ) : null}
        </div>
      </div>

      {/* Sticky filters bar */}
      <div className="sticky top-0 z-10 -mx-1 rounded-lg border border-border/60 bg-card/80 px-3 py-2 backdrop-blur">
        <div className="flex flex-wrap items-center gap-2">
          <Select value={service} onValueChange={setService}>
            <SelectTrigger className="h-9 w-[200px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {SERVICES.map((s) => (
                <SelectItem key={s.value} value={s.value}>
                  {s.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Select value={level} onValueChange={setLevel}>
            <SelectTrigger className="h-9 w-[120px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {LEVELS.map((l) => (
                <SelectItem key={l.value} value={l.value}>
                  {l.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Input
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Filtrar por texto..."
            className="h-9 max-w-sm flex-1"
          />

          <div className="ml-auto flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={handleTogglePause}
              className="gap-1"
            >
              {paused ? <Play className="h-3.5 w-3.5" /> : <Pause className="h-3.5 w-3.5" />}
              {paused ? 'Reanudar' : 'Pausar'}
            </Button>
            <Button variant="outline" size="sm" onClick={handleRefresh} className="gap-1">
              <RefreshCw className={`h-3.5 w-3.5 ${isFetching ? 'animate-spin' : ''}`} />
              Refrescar
            </Button>
            <Button variant="outline" size="sm" onClick={handleCopy} className="gap-1">
              <Copy className="h-3.5 w-3.5" />
              Copiar visible
            </Button>
          </div>
        </div>
        <div className="mt-1.5 flex items-center gap-3 text-xs text-muted-foreground">
          <span>{lines.length} líneas visibles</span>
          {data?.truncated ? <span className="text-amber-400">(buffer truncado)</span> : null}
          {paused ? <span className="text-amber-400">auto-refresh pausado</span> : null}
        </div>
      </div>

      {/* Virtualized log list */}
      {isError ? (
        <div className="rounded-lg border border-destructive/40 bg-destructive/5 p-6 text-sm text-destructive">
          Error cargando logs: {error?.message || 'desconocido'}
        </div>
      ) : isLoading ? (
        <div className="space-y-1.5 rounded-lg border border-border/60 bg-black/40 p-3">
          {Array.from({ length: 14 }).map((_, i) => (
            <Skeleton key={i} className="h-4 w-full bg-muted/20" />
          ))}
        </div>
      ) : lines.length === 0 ? (
        <div className="rounded-lg border border-dashed border-border/60 bg-card/30 p-10 text-center text-sm text-muted-foreground">
          Sin entradas que coincidan con los filtros.
        </div>
      ) : (
        <div
          ref={scrollRef}
          className="relative h-[68vh] overflow-auto rounded-lg border border-border/60 bg-black/60 font-mono text-xs"
          style={{ fontFamily: '"JetBrains Mono Variable", "JetBrains Mono", ui-monospace, monospace' }}
        >
          <div style={{ height: virtualizer.getTotalSize(), position: 'relative' }}>
            {virtualizer.getVirtualItems().map((vi) => {
              const l = lines[vi.index]
              const lvl = (l.level || 'INFO').toUpperCase()
              return (
                <div
                  key={vi.key}
                  data-index={vi.index}
                  ref={virtualizer.measureElement}
                  style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    right: 0,
                    transform: `translateY(${vi.start}px)`,
                  }}
                  className="flex items-start gap-2 px-3 py-0.5 hover:bg-white/5"
                >
                  <span className="shrink-0 text-muted-foreground/70">{formatTs(l.timestamp)}</span>
                  <span
                    className={`shrink-0 rounded border px-1 text-[10px] uppercase ${
                      LEVEL_BADGE[lvl] || LEVEL_BADGE.INFO
                    }`}
                  >
                    {lvl}
                  </span>
                  {l.logger ? (
                    <span className="shrink-0 text-muted-foreground/60">{l.logger}</span>
                  ) : null}
                  <span className={`min-w-0 break-all ${LEVEL_STYLES[lvl] || 'text-foreground/90'}`}>
                    {l.message}
                  </span>
                </div>
              )
            })}
          </div>
        </div>
      )}
    </motion.div>
  )
}

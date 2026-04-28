// Pipeline detail header: title, status, total elapsed, action buttons.
// Cancel uses an AlertDialog confirm; Retry creates a new pipeline and navigates to it.
import React, { useEffect, useState } from "react"
import { useNavigate } from "react-router-dom"
import { ArrowLeft, X, RotateCw, Loader2, BookOpen } from "lucide-react"
import { Button } from "@/components/ui/button"
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
} from "@/components/ui/alert-dialog"
import {
  useCancelPipeline,
  useRetryPipeline,
} from "@/features/pipeline/api/usePipeline"
import { cn } from "@/lib/utils"
import { toast } from "sonner"

/**
 * Extract the instructional name from a pipeline path.
 * e.g. "/media/instruccionales/Tripod Passing - Jozef Chen/Season 01/S01E01.mp4"
 *   → "Tripod Passing - Jozef Chen"
 */
function deriveInstructionalName(path) {
  if (!path) return null
  const parts = path.replace(/\\/g, "/").split("/").filter(Boolean)
  // Find the "Season NN" segment and return the part before it
  for (let i = 0; i < parts.length; i++) {
    if (/^Season\s+\d+$/i.test(parts[i]) && i > 0) {
      return parts[i - 1]
    }
  }
  // Fallback: look for parent of a file with SNNENN pattern
  for (let i = parts.length - 1; i >= 0; i--) {
    if (/S\d{2}E\d{2,3}/i.test(parts[i]) && i >= 2) {
      return parts[i - 2]
    }
  }
  return null
}

const STATUS_BADGE = {
  pending: "bg-zinc-800 text-zinc-300",
  running: "bg-amber-500/15 text-amber-400 ring-1 ring-amber-500/30",
  completed: "bg-emerald-500/15 text-emerald-400 ring-1 ring-emerald-500/30",
  failed: "bg-red-500/15 text-red-400 ring-1 ring-red-500/30",
  cancelled: "bg-zinc-800 text-zinc-400",
  cancelling: "bg-amber-500/10 text-amber-400 ring-1 ring-amber-500/20",
}

const STATUS_LABEL = {
  pending: "pendiente",
  running: "ejecutando",
  completed: "completado",
  failed: "fallido",
  cancelled: "cancelado",
  cancelling: "cancelando",
}

function fmt(sec) {
  if (sec == null || sec < 0) return "—"
  const s = Math.round(sec)
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  const r = s % 60
  if (m < 60) return r ? `${m}m ${r}s` : `${m}m`
  const h = Math.floor(m / 60)
  return `${h}h ${m % 60}m`
}

function elapsed(p, tick) {
  if (!p?.created_at) return null
  const t0 = new Date(p.created_at).getTime()
  const t1 = p.completed_at ? new Date(p.completed_at).getTime() : tick
  return Math.max(0, (t1 - t0) / 1000)
}

export default function PipelineHeader({ pipeline }) {
  const navigate = useNavigate()
  const cancel = useCancelPipeline()
  const retry = useRetryPipeline()
  const [tick, setTick] = useState(Date.now())

  const isActive = pipeline && ["running", "pending", "cancelling"].includes(pipeline.status)
  const isFinal = pipeline && ["failed", "cancelled"].includes(pipeline.status)

  // Live tick for elapsed counter while running
  useEffect(() => {
    if (!isActive) return
    const i = setInterval(() => setTick(Date.now()), 1000)
    return () => clearInterval(i)
  }, [isActive])

  if (!pipeline) return null

  const title = pipeline.path?.split(/[\\/]/).filter(Boolean).pop() || pipeline.pipeline_id
  const status = pipeline.status || "pending"

  const onCancel = async () => {
    try {
      await cancel.mutateAsync({ id: pipeline.pipeline_id })
      toast.success("Cancelación solicitada")
    } catch (e) {
      toast.error(`No se pudo cancelar: ${e?.message || e}`)
    }
  }

  const onRetry = async () => {
    try {
      const res = await retry.mutateAsync({ id: pipeline.pipeline_id })
      const newId = res?.pipeline_id
      toast.success(`Reintento lanzado: ${newId}`)
      if (newId) navigate(`/pipelines/${newId}`)
    } catch (e) {
      toast.error(`No se pudo reintentar: ${e?.message || e}`)
    }
  }

  const instructionalName = deriveInstructionalName(pipeline.path)

  return (
    <div className="flex flex-col gap-3 border-b border-zinc-800/80 bg-zinc-950/60 px-4 py-3 sm:flex-row sm:items-center sm:justify-between sm:gap-4 sm:px-6 sm:py-4">
      <div className="flex min-w-0 items-center gap-2 sm:gap-3">
        <Button variant="ghost" size="icon" onClick={() => navigate(-1)} title="Volver" className="shrink-0">
          <ArrowLeft className="h-4 w-4" />
        </Button>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <h1 className="min-w-0 max-w-full truncate text-sm sm:text-base font-semibold text-zinc-100" title={pipeline.path}>
              {title}
            </h1>
            <span
              className={cn(
                "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] sm:text-[11px] font-medium uppercase tracking-wide shrink-0",
                STATUS_BADGE[status] || STATUS_BADGE.pending
              )}
            >
              {isActive && <Loader2 className="h-3 w-3 animate-spin" />}
              {STATUS_LABEL[status] || status}
            </span>
          </div>
          <div className="mt-0.5 flex flex-wrap items-center gap-x-2 gap-y-0.5 text-[11px] sm:text-xs text-zinc-500">
            <span className="font-mono truncate max-w-[80px] sm:max-w-none">{pipeline.pipeline_id}</span>
            <span className="hidden sm:inline">·</span>
            <span className="tabular-nums">{fmt(elapsed(pipeline, tick))}</span>
            <span className="hidden sm:inline">·</span>
            <span>{pipeline.steps?.length ?? 0} pasos</span>
          </div>
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-2 sm:flex-nowrap">
        {instructionalName && (
          <Button
            variant="outline"
            size="sm"
            onClick={() => navigate(`/library/${encodeURIComponent(instructionalName)}`)}
            title="Volver al instruccional"
          >
            <BookOpen className="h-3.5 w-3.5 sm:mr-1.5" />
            <span className="hidden sm:inline">Volver al instruccional</span>
          </Button>
        )}
        {isActive && (
          <AlertDialog>
            <AlertDialogTrigger asChild>
              <Button variant="outline" size="sm" disabled={cancel.isPending} title="Cancelar">
                <X className="h-3.5 w-3.5 sm:mr-1.5" />
                <span className="hidden sm:inline">Cancelar</span>
              </Button>
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>¿Cancelar pipeline?</AlertDialogTitle>
                <AlertDialogDescription>
                  Se interrumpirá el paso en curso. Los archivos ya generados se conservan.
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Volver</AlertDialogCancel>
                <AlertDialogAction onClick={onCancel}>Sí, cancelar</AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        )}
        {isFinal && (
          <Button size="sm" variant="default" onClick={onRetry} disabled={retry.isPending} title="Reintentar">
            <RotateCw className={cn("h-3.5 w-3.5 sm:mr-1.5", retry.isPending && "animate-spin")} />
            <span className="hidden sm:inline">Reintentar</span>
          </Button>
        )}
      </div>
    </div>
  )
}

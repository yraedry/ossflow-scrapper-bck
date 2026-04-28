// Pipeline detail page: header + 2-column layout (timeline + LogPanel SSE).
// LogPanel handles its own width persistence (drag handle on its left edge).
import React, { useEffect, useMemo, useRef } from "react"
import { useNavigate, useParams } from "react-router-dom"
import { toast } from "sonner"
import { Skeleton } from "@/components/ui/skeleton"
import { Button } from "@/components/ui/button"
import { AlertCircle, ArrowLeft } from "lucide-react"
import {
  usePipeline,
  usePipelineEta,
} from "@/features/pipeline/api/usePipeline"
import PipelineHeader from "@/features/pipeline/components/PipelineHeader"
import StepTimeline from "@/features/pipeline/components/StepTimeline"
import StepDiff from "@/features/pipeline/components/StepDiff"
import LogPanel from "@/features/pipeline/components/LogPanel"

export default function PipelineDetailPage() {
  const { id } = useParams()
  const navigate = useNavigate()
  const { data: pipeline, isLoading, isError, error } = usePipeline(id)

  const stepsKey = useMemo(
    () => (pipeline?.steps || []).map((s) => s.name).join(","),
    [pipeline]
  )
  const { data: eta } = usePipelineEta(
    stepsKey ? { steps: stepsKey, path: pipeline?.path } : undefined
  )

  // Toast on terminal state — fire once per pipeline.
  const toastedRef = useRef(null)
  useEffect(() => {
    if (!pipeline) return
    if (toastedRef.current === pipeline.pipeline_id) return
    if (pipeline.status === "completed") {
      toastedRef.current = pipeline.pipeline_id
      toast.success("Pipeline completado")
    } else if (pipeline.status === "failed") {
      toastedRef.current = pipeline.pipeline_id
      toast.error("Pipeline falló")
    } else if (pipeline.status === "cancelled") {
      toastedRef.current = pipeline.pipeline_id
      toast("Pipeline cancelado")
    }
  }, [pipeline?.pipeline_id, pipeline?.status])

  // 404 → redirect to list after a brief message
  const notFound =
    isError &&
    (error?.status === 404 ||
      /not found/i.test(error?.message || "") ||
      /404/.test(String(error?.message || "")))

  useEffect(() => {
    if (!notFound) return
    const t = setTimeout(() => navigate("/pipelines"), 1500)
    return () => clearTimeout(t)
  }, [notFound, navigate])

  if (isLoading) {
    return (
      <div className="flex h-full flex-col">
        <div className="flex items-center gap-3 border-b border-zinc-800/80 px-6 py-4">
          <Skeleton className="h-9 w-9 rounded-md" />
          <div className="space-y-2">
            <Skeleton className="h-4 w-64" />
            <Skeleton className="h-3 w-40" />
          </div>
        </div>
        <div className="flex flex-col lg:flex-row flex-1">
          <div className="flex-1 space-y-4 p-4 sm:p-6">
            {[0, 1, 2, 3].map((i) => (
              <div key={i} className="flex gap-4">
                <Skeleton className="h-9 w-9 rounded-full" />
                <div className="flex-1 space-y-2">
                  <Skeleton className="h-4 w-32" />
                  <Skeleton className="h-3 w-72" />
                </div>
              </div>
            ))}
          </div>
          <div className="hidden lg:block w-[480px] border-l border-zinc-800/80 p-4">
            <Skeleton className="h-full w-full" />
          </div>
        </div>
      </div>
    )
  }

  if (notFound) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-4 text-center">
        <AlertCircle className="h-10 w-10 text-zinc-600" />
        <div>
          <h2 className="text-lg font-semibold text-zinc-200">Pipeline no encontrado</h2>
          <p className="text-sm text-zinc-500">Redirigiendo a la lista…</p>
        </div>
      </div>
    )
  }

  if (isError) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-4 text-center">
        <AlertCircle className="h-10 w-10 text-red-500" />
        <div className="space-y-1">
          <h2 className="text-lg font-semibold text-zinc-200">Error cargando pipeline</h2>
          <p className="text-sm text-zinc-500">{error?.message || "Error desconocido"}</p>
        </div>
        <Button variant="outline" size="sm" onClick={() => navigate("/pipelines")}>
          <ArrowLeft className="mr-1.5 h-3.5 w-3.5" /> Volver
        </Button>
      </div>
    )
  }

  return (
    <div className="flex h-full flex-col bg-zinc-950">
      <PipelineHeader pipeline={pipeline} />
      <div className="flex flex-col lg:flex-row min-h-0 flex-1">
        <div className="min-w-0 flex-1 overflow-auto px-4 sm:px-6 py-4 sm:py-6">
          <StepTimeline steps={pipeline.steps || []} eta={eta}>
            {(step) => <StepDiff diff={step.diff} />}
          </StepTimeline>
        </div>
        {/* LogPanel is a fixed 480px wide rail; on <lg viewports it crushes
            the timeline, so we hide it. Logs are still reachable from the
            Library/Logs tab. */}
        <div className="hidden lg:block">
          <LogPanel pipelineId={pipeline.pipeline_id} steps={pipeline.steps || []} />
        </div>
      </div>
    </div>
  )
}

// WIRE_ROUTE_PIPELINE_DETAIL: /pipelines/:id → src/features/pipeline/pages/PipelineDetailPage.jsx

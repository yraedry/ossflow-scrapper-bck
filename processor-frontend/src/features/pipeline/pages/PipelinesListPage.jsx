// WIRE_ROUTE_PIPELINES_LIST: /pipelines → src/features/pipeline/pages/PipelinesListPage.jsx
//
// Tabla de pipelines con filtros (estado, rango de fecha, búsqueda),
// ordenación por columna, acciones inline (Cancel/Retry) y polling 2s
// gestionado por el hook usePipelines (refetchInterval adaptativo).
import React, { useMemo, useState } from "react"
import { useNavigate } from "react-router-dom"
import { motion, AnimatePresence } from "framer-motion"
import { toast } from "sonner"
import {
  Loader2,
  RotateCw,
  X,
  ArrowUpDown,
  ArrowUp,
  ArrowDown,
  Play,
  Search,
  Filter,
  RefreshCw,
  Scissors,
  Captions,
  Mic,
  Layers,
} from "lucide-react"
import {
  usePipelines,
  useCancelPipeline,
  useRetryPipeline,
} from "@/features/pipeline/api/usePipeline"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/Badge"
import { Skeleton } from "@/components/ui/skeleton"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { cn } from "@/lib/utils"

const STATUS_OPTIONS = [
  { value: "all", label: "Todos los estados" },
  { value: "pending", label: "Pendiente" },
  { value: "running", label: "Ejecutando" },
  { value: "completed", label: "Completado" },
  { value: "failed", label: "Fallido" },
  { value: "cancelled", label: "Cancelado" },
]

const STATUS_BADGE = {
  pending: "bg-zinc-800 text-zinc-300",
  queued: "bg-zinc-800 text-zinc-300",
  running: "bg-amber-500/15 text-amber-400 ring-1 ring-amber-500/30",
  completed: "bg-emerald-500/15 text-emerald-400 ring-1 ring-emerald-500/30",
  failed: "bg-red-500/15 text-red-400 ring-1 ring-red-500/30",
  cancelled: "bg-zinc-800 text-zinc-400",
  cancelling: "bg-amber-500/10 text-amber-400 ring-1 ring-amber-500/20",
}

const STATUS_LABEL = {
  pending: "pendiente",
  queued: "en cola",
  running: "ejecutando",
  completed: "completado",
  failed: "fallido",
  cancelled: "cancelado",
  cancelling: "cancelando",
}

const PAGE_SIZE = 25

const STEP_META = {
  chapters:  { label: "Troceado",   Icon: Scissors, cls: "bg-violet-500/15 text-violet-300 border-violet-500/30" },
  subtitles: { label: "Subtítulos", Icon: Captions, cls: "bg-sky-500/15 text-sky-300 border-sky-500/30" },
  dubbing:   { label: "Doblaje",    Icon: Mic,      cls: "bg-rose-500/15 text-rose-300 border-rose-500/30" },
}

const PIPELINE_TYPE_OPTIONS = [
  { value: "all",       label: "Todos los tipos" },
  { value: "global",    label: "Global (todos los pasos)" },
  { value: "chapters",  label: "Troceado" },
  { value: "subtitles", label: "Subtítulos" },
  { value: "dubbing",   label: "Doblaje" },
]

function extractStepNames(p) {
  const raw = p.steps || []
  if (!raw.length) return []
  return raw.map((s) => (typeof s === "string" ? s : s?.name || "")).filter(Boolean)
}

function pipelineTypeLabel(stepNames) {
  if (!stepNames.length) return null
  const has = (n) => stepNames.includes(n)
  if (has("chapters") && has("subtitles") && has("dubbing")) return "global"
  if (stepNames.length === 1) return stepNames[0]
  return stepNames.join("+")
}

function StepBadges({ stepNames }) {
  if (!stepNames.length) return <span className="text-xs text-zinc-600">—</span>
  const isGlobal = stepNames.includes("chapters") && stepNames.includes("subtitles") && stepNames.includes("dubbing")
  if (isGlobal) {
    return (
      <span className={cn(
        "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-medium",
        "bg-amber-500/15 text-amber-300 border-amber-500/30"
      )}>
        <Layers className="h-3 w-3" /> Global
      </span>
    )
  }
  return (
    <div className="flex flex-wrap gap-1">
      {stepNames.map((name) => {
        const meta = STEP_META[name]
        if (!meta) return (
          <span key={name} className="inline-flex items-center rounded-full border border-zinc-700/50 bg-zinc-800/50 px-2 py-0.5 text-[10px] text-zinc-400">
            {name}
          </span>
        )
        return (
          <span key={name} className={cn(
            "inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-medium",
            meta.cls,
          )}>
            <meta.Icon className="h-3 w-3" /> {meta.label}
          </span>
        )
      })}
    </div>
  )
}

function extractName(path) {
  if (!path) return "—"
  const parts = path.replace(/\\/g, "/").split("/").filter(Boolean)
  return parts[parts.length - 1] || path
}

function timeAgo(iso) {
  if (!iso) return "—"
  const diff = Date.now() - new Date(iso).getTime()
  const mins = Math.floor(diff / 60000)
  if (mins < 1) return "ahora"
  if (mins < 60) return `hace ${mins}m`
  const hours = Math.floor(mins / 60)
  if (hours < 24) return `hace ${hours}h`
  return `hace ${Math.floor(hours / 24)}d`
}

function fmtDuration(p) {
  if (!p?.created_at) return "—"
  const t0 = new Date(p.created_at).getTime()
  const t1 = p.completed_at ? new Date(p.completed_at).getTime() : Date.now()
  const sec = Math.max(0, Math.round((t1 - t0) / 1000))
  if (sec < 60) return `${sec}s`
  const m = Math.floor(sec / 60)
  const r = sec % 60
  if (m < 60) return r ? `${m}m ${r}s` : `${m}m`
  const h = Math.floor(m / 60)
  return `${h}h ${m % 60}m`
}

function stepsSummary(p) {
  const raw = Array.isArray(p.steps) && p.steps.length && typeof p.steps[0] === "object"
    ? p.steps
    : p.steps_detail || (p.steps || []).map((n) => ({ name: n, status: "pending" }))
  const total = raw.length
  const done = raw.filter((s) => s.status === "completed").length
  return { raw, total, done }
}

function MiniProgress({ pipeline }) {
  const { raw, total, done } = stepsSummary(pipeline)
  if (!total) return <span className="text-xs text-zinc-500">—</span>
  return (
    <div className="flex items-center gap-2 min-w-[120px]">
      <div className="flex-1 flex gap-0.5">
        {raw.map((s, i) => (
          <div
            key={`${s.name}-${i}`}
            title={`${s.name}: ${s.status}`}
            className={cn(
              "h-1.5 flex-1 rounded-sm",
              s.status === "completed" && "bg-emerald-500",
              s.status === "running" && "bg-amber-400 animate-pulse",
              s.status === "failed" && "bg-red-500",
              (s.status === "pending" || !s.status) && "bg-zinc-700",
              s.status === "cancelled" && "bg-zinc-600",
            )}
          />
        ))}
      </div>
      <span className="text-[11px] tabular-nums text-zinc-500 font-mono">
        {done}/{total}
      </span>
    </div>
  )
}

function SortButton({ label, column, sort, setSort }) {
  const active = sort.column === column
  const Icon = !active ? ArrowUpDown : sort.dir === "asc" ? ArrowUp : ArrowDown
  return (
    <button
      type="button"
      onClick={() =>
        setSort(
          active
            ? { column, dir: sort.dir === "asc" ? "desc" : "asc" }
            : { column, dir: "desc" },
        )
      }
      className={cn(
        "inline-flex items-center gap-1 text-xs font-medium uppercase tracking-wide",
        active ? "text-zinc-200" : "text-zinc-400 hover:text-zinc-200",
      )}
    >
      {label}
      <Icon className="h-3 w-3" />
    </button>
  )
}

export default function PipelinesListPage() {
  const navigate = useNavigate()
  const { data: pipelines = [], isLoading, isFetching, refetch, error } = usePipelines()
  const cancel = useCancelPipeline()
  const retry = useRetryPipeline()

  const [statusFilter, setStatusFilter] = useState("all")
  const [typeFilter, setTypeFilter] = useState("all")
  const [search, setSearch] = useState("")
  const [fromDate, setFromDate] = useState("")
  const [toDate, setToDate] = useState("")
  const [sort, setSort] = useState({ column: "created_at", dir: "desc" })
  const [page, setPage] = useState(1)

  const filtered = useMemo(() => {
    let list = pipelines
    if (statusFilter !== "all") list = list.filter((p) => p.status === statusFilter)
    if (typeFilter !== "all") {
      list = list.filter((p) => {
        const names = extractStepNames(p)
        const type = pipelineTypeLabel(names)
        if (typeFilter === "global") return type === "global"
        return names.includes(typeFilter)
      })
    }
    if (search.trim()) {
      const q = search.trim().toLowerCase()
      list = list.filter((p) => (p.path || "").toLowerCase().includes(q))
    }
    if (fromDate) {
      const t = new Date(fromDate).getTime()
      list = list.filter((p) => p.created_at && new Date(p.created_at).getTime() >= t)
    }
    if (toDate) {
      const t = new Date(toDate).getTime() + 86400000 // include day
      list = list.filter((p) => p.created_at && new Date(p.created_at).getTime() <= t)
    }
    const sorted = [...list].sort((a, b) => {
      const { column, dir } = sort
      const mult = dir === "asc" ? 1 : -1
      if (column === "instructional") {
        return extractName(a.path).localeCompare(extractName(b.path)) * mult
      }
      if (column === "status") {
        return String(a.status || "").localeCompare(String(b.status || "")) * mult
      }
      if (column === "duration") {
        const durA = (a.completed_at ? new Date(a.completed_at).getTime() : Date.now()) - new Date(a.created_at || 0).getTime()
        const durB = (b.completed_at ? new Date(b.completed_at).getTime() : Date.now()) - new Date(b.created_at || 0).getTime()
        return (durA - durB) * mult
      }
      // created_at
      const ta = new Date(a.created_at || 0).getTime()
      const tb = new Date(b.created_at || 0).getTime()
      return (ta - tb) * mult
    })
    return sorted
  }, [pipelines, statusFilter, search, fromDate, toDate, sort])

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE))
  const currentPage = Math.min(page, totalPages)
  const paged = filtered.slice((currentPage - 1) * PAGE_SIZE, currentPage * PAGE_SIZE)

  const hasRunning = pipelines.some((p) =>
    ["running", "pending", "queued"].includes(p.status),
  )

  const handleCancel = async (id, e) => {
    e.stopPropagation()
    try {
      await cancel.mutateAsync({ id })
      toast.success("Cancelación solicitada")
    } catch (err) {
      toast.error(`No se pudo cancelar: ${err?.message || err}`)
    }
  }

  const handleRetry = async (id, e) => {
    e.stopPropagation()
    try {
      const res = await retry.mutateAsync({ id })
      const newId = res?.pipeline_id
      toast.success(`Reintento lanzado${newId ? `: ${newId}` : ""}`)
      if (newId) navigate(`/pipelines/${newId}`)
    } catch (err) {
      toast.error(`No se pudo reintentar: ${err?.message || err}`)
    }
  }

  return (
    <div className="p-4 sm:p-6">
      <div className="mb-4 sm:mb-6 flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h1 className="text-xl sm:text-2xl font-semibold text-zinc-100">Pipelines</h1>
          <p className="text-xs sm:text-sm text-zinc-500 mt-1">
            Historial y monitorización de pipelines. {hasRunning && (
              <span className="text-amber-400ml-1 inline-flex items-center gap-1">
                <Loader2 className="h-3 w-3 animate-spin" /> actualizando cada 2s
              </span>
            )}
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={() => refetch()} disabled={isFetching} className="shrink-0">
          <RefreshCw className={cn("h-3.5 w-3.5 sm:mr-1.5", isFetching && "animate-spin")} />
          <span className="hidden sm:inline">Actualizar</span>
        </Button>
      </div>

      {/* Filters */}
      <div className="mb-4 flex flex-wrap items-center gap-2">
        <div className="relative flex-1 min-w-[220px] max-w-sm">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-4 w-4 text-zinc-500 pointer-events-none" />
          <Input
            placeholder="Buscar instructional..."
            value={search}
            onChange={(e) => {
              setSearch(e.target.value)
              setPage(1)
            }}
            className="pl-8"
          />
        </div>
        <Select
          value={statusFilter}
          onValueChange={(v) => {
            setStatusFilter(v)
            setPage(1)
          }}
        >
          <SelectTrigger className="w-full sm:w-[180px]">
            <Filter className="h-3.5 w-3.5 mr-1.5 text-zinc-500" />
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {STATUS_OPTIONS.map((o) => (
              <SelectItem key={o.value} value={o.value}>
                {o.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Select
          value={typeFilter}
          onValueChange={(v) => {
            setTypeFilter(v)
            setPage(1)
          }}
        >
          <SelectTrigger className="w-full sm:w-[180px]">
            <Layers className="h-3.5 w-3.5 mr-1.5 text-zinc-500" />
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {PIPELINE_TYPE_OPTIONS.map((o) => (
              <SelectItem key={o.value} value={o.value}>
                {o.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Input
          type="date"
          value={fromDate}
          onChange={(e) => {
            setFromDate(e.target.value)
            setPage(1)
          }}
          className="hidden sm:block sm:w-[160px]"
          title="Desde"
        />
        <Input
          type="date"
          value={toDate}
          onChange={(e) => {
            setToDate(e.target.value)
            setPage(1)
          }}
          className="hidden sm:block sm:w-[160px]"
          title="Hasta"
        />
        {(statusFilter !== "all" || typeFilter !== "all" || search || fromDate || toDate) && (
          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              setStatusFilter("all")
              setTypeFilter("all")
              setSearch("")
              setFromDate("")
              setToDate("")
              setPage(1)
            }}
          >
            Limpiar
          </Button>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="mb-4 rounded-md border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-400">
          {error?.message || "Error al cargar pipelines"}
        </div>
      )}

      {/* Table */}
      <div className="rounded-md border border-zinc-800/60 bg-zinc-950/40 overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>
                <SortButton label="Instructional" column="instructional" sort={sort} setSort={setSort} />
              </TableHead>
              <TableHead className="w-[140px]">
                <SortButton label="Estado" column="status" sort={sort} setSort={setSort} />
              </TableHead>
              <TableHead className="hidden md:table-cell w-[180px]">Tipo</TableHead>
              <TableHead className="hidden lg:table-cell w-[160px]">Pasos</TableHead>
              <TableHead className="hidden md:table-cell w-[110px]">
                <SortButton label="Duración" column="duration" sort={sort} setSort={setSort} />
              </TableHead>
              <TableHead className="hidden md:table-cell w-[120px]">
                <SortButton label="Inicio" column="created_at" sort={sort} setSort={setSort} />
              </TableHead>
              <TableHead className="w-[110px] text-right">Acciones</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {isLoading ? (
              Array.from({ length: 6 }).map((_, i) => (
                <TableRow key={`sk-${i}`}>
                  <TableCell><Skeleton className="h-4 w-3/4" /></TableCell>
                  <TableCell><Skeleton className="h-5 w-20" /></TableCell>
                  <TableCell className="hidden md:table-cell"><Skeleton className="h-5 w-28" /></TableCell>
                  <TableCell className="hidden lg:table-cell"><Skeleton className="h-3 w-24" /></TableCell>
                  <TableCell className="hidden md:table-cell"><Skeleton className="h-4 w-12" /></TableCell>
                  <TableCell className="hidden md:table-cell"><Skeleton className="h-4 w-16" /></TableCell>
                  <TableCell><Skeleton className="h-7 w-20 ml-auto" /></TableCell>
                </TableRow>
              ))
            ) : paged.length === 0 ? (
              <TableRow>
                <TableCell colSpan={7} className="h-40 text-center">
                  <div className="flex flex-col items-center justify-center gap-2 text-zinc-500">
                    <Play className="h-8 w-8 opacity-40" />
                    <p className="text-sm">
                      {pipelines.length === 0
                        ? "No hay pipelines registrados. Lanza uno desde la Biblioteca."
                        : "No hay pipelines que coincidan con los filtros."}
                    </p>
                  </div>
                </TableCell>
              </TableRow>
            ) : (
              <AnimatePresence initial={false}>
                {paged.map((p) => {
                  const status = p.status || "pending"
                  const isActive = ["running", "pending", "queued", "cancelling"].includes(status)
                  const isFinal = ["completed", "failed", "cancelled"].includes(status)
                  return (
                    <motion.tr
                      key={p.pipeline_id}
                      layout
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      transition={{ duration: 0.15 }}
                      onClick={() => navigate(`/pipelines/${p.pipeline_id}`)}
                      className="cursor-pointer border-b border-zinc-800/40 hover:bg-zinc-900/40"
                    >
                      <TableCell className="max-w-[320px]">
                        <div className="flex flex-col min-w-0">
                          <span className="truncate text-sm font-medium text-zinc-100">
                            {extractName(p.path)}
                          </span>
                          <span className="truncate text-xs text-zinc-500 font-mono">
                            {p.path}
                          </span>
                        </div>
                      </TableCell>
                      <TableCell>
                        <Badge
                          className={cn(
                            "border-0 text-[10px] uppercase tracking-wide",
                            STATUS_BADGE[status] || STATUS_BADGE.pending,
                          )}
                        >
                          {isActive && status !== "pending" && (
                            <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                          )}
                          {STATUS_LABEL[status] || status}
                        </Badge>
                      </TableCell>
                      <TableCell className="hidden md:table-cell">
                        <StepBadges stepNames={extractStepNames(p)} />
                      </TableCell>
                      <TableCell className="hidden lg:table-cell">
                        <MiniProgress pipeline={p} />
                      </TableCell>
                      <TableCell className="hidden md:table-cell text-xs tabular-nums text-zinc-400">
                        {fmtDuration(p)}
                      </TableCell>
                      <TableCell className="hidden md:table-cell text-xs text-zinc-400">
                        {timeAgo(p.created_at)}
                      </TableCell>
                      <TableCell className="text-right">
                        <div className="flex items-center justify-end gap-1">
                          {isActive && (
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={(e) => handleCancel(p.pipeline_id, e)}
                              disabled={cancel.isPending}
                              title="Cancelar"
                              className="h-7 px-2 text-red-400 hover:text-red-300"
                            >
                              <X className="h-3.5 w-3.5" />
                            </Button>
                          )}
                          {isFinal && (
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={(e) => handleRetry(p.pipeline_id, e)}
                              disabled={retry.isPending}
                              title="Reintentar"
                              className="h-7 px-2 text-amber-400 hover:text-amber-300"
                            >
                              <RotateCw className="h-3.5 w-3.5" />
                            </Button>
                          )}
                        </div>
                      </TableCell>
                    </motion.tr>
                  )
                })}
              </AnimatePresence>
            )}
          </TableBody>
        </Table>
      </div>

      {/* Pagination */}
      {filtered.length > PAGE_SIZE && (
        <div className="mt-4 flex items-center justify-between text-xs text-zinc-500">
          <span>
            {(currentPage - 1) * PAGE_SIZE + 1}–{Math.min(currentPage * PAGE_SIZE, filtered.length)} de {filtered.length}
          </span>
          <div className="flex items-center gap-1">
            <Button
              variant="outline"
              size="sm"
              disabled={currentPage <= 1}
              onClick={() => setPage((p) => Math.max(1, p - 1))}
            >
              Anterior
            </Button>
            <span className="px-2 tabular-nums">
              {currentPage} / {totalPages}
            </span>
            <Button
              variant="outline"
              size="sm"
              disabled={currentPage >= totalPages}
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            >
              Siguiente
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}

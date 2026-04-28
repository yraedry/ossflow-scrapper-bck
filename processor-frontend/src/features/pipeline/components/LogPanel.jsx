// Live SSE log panel with virtualized list, filters, auto-scroll and resize.
// SSE event shape (from processor-api/api/pipeline.py):
//   { type: "step_started"|"step_progress"|"step_completed"|"step_failed"
//          |"step_diff"|"pipeline_started"|"pipeline_completed"|"pipeline_failed"|"log",
//     step?: string, step_index?: number, message?: string, progress?: number,
//     data?: { message?: string }, seq: number, ... }
// Levels are inferred: failed → error, step_started/step_completed → info, log → info, etc.
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { useVirtualizer } from "@tanstack/react-virtual"
import {
  Pause,
  Play,
  Trash2,
  Copy,
  Filter,
  ChevronDown,
} from "lucide-react"
import { Button } from "@/components/ui/button"
import { subscribeSSE } from "@/lib/sse"
import { pipelineEventsUrl } from "@/features/pipeline/api/usePipeline"
import { cn } from "@/lib/utils"
import { toast } from "sonner"

const LEVEL_FROM_TYPE = {
  step_failed: "error",
  pipeline_failed: "error",
  step_started: "info",
  step_completed: "info",
  pipeline_started: "info",
  pipeline_completed: "info",
  step_progress: "info",
  step_diff: "info",
  log: "info",
}

const LEVEL_STYLE = {
  error: "text-red-400",
  warn: "text-amber-400",
  info: "text-zinc-300",
}

const WIDTH_KEY = "bjj.logPanel.width"
const MIN_W = 280
const MAX_W = 900

function entryFrom(evt) {
  const type = evt?.type || "log"
  let level = LEVEL_FROM_TYPE[type] || "info"
  const raw = evt?.message || evt?.data?.message || ""
  const lower = (raw || "").toLowerCase()
  if (level === "info") {
    if (/error|failed|exception|traceback/.test(lower)) level = "error"
    else if (/warn|warning/.test(lower)) level = "warn"
  }
  return {
    seq: evt?.seq ?? 0,
    type,
    step: evt?.step || null,
    progress: evt?.progress ?? null,
    level,
    message:
      raw ||
      (type === "step_started"
        ? `→ ${evt.step} iniciado`
        : type === "step_completed"
        ? `✓ ${evt.step} completado`
        : type === "step_failed"
        ? `✗ ${evt.step} falló`
        : type === "step_diff"
        ? `Δ ${evt.step}: +${(evt.added || []).length} ~${(evt.modified || []).length} -${(evt.removed || []).length}`
        : type === "pipeline_started"
        ? "Pipeline iniciado"
        : type === "pipeline_completed"
        ? "Pipeline completado"
        : type === "pipeline_failed"
        ? "Pipeline falló"
        : ""),
    ts: Date.now(),
  }
}

export default function LogPanel({ pipelineId, steps = [], onTerminal }) {
  const [width, setWidth] = useState(() => {
    const stored = Number(localStorage.getItem(WIDTH_KEY))
    return Number.isFinite(stored) && stored >= MIN_W ? stored : 480
  })
  const [entries, setEntries] = useState([])
  const [paused, setPaused] = useState(false)
  const [levelFilter, setLevelFilter] = useState("all")
  const [stepFilter, setStepFilter] = useState("all")

  const pausedRef = useRef(paused)
  pausedRef.current = paused
  const seenSeqRef = useRef(new Set())
  const scrollRef = useRef(null)
  const onTerminalRef = useRef(onTerminal)
  onTerminalRef.current = onTerminal

  // SSE subscription
  useEffect(() => {
    if (!pipelineId) return
    seenSeqRef.current = new Set()
    setEntries([])
    const sub = subscribeSSE(pipelineEventsUrl(pipelineId), {
      onMessage: (evt) => {
        const seq = evt?.seq
        if (seq != null) {
          if (seenSeqRef.current.has(seq)) return
          seenSeqRef.current.add(seq)
        }
        const entry = entryFrom(evt)
        setEntries((prev) => {
          const next = [...prev, entry]
          if (next.length > 5000) next.splice(0, next.length - 5000)
          return next
        })
        if (
          evt?.type === "pipeline_completed" ||
          evt?.type === "pipeline_failed"
        ) {
          onTerminalRef.current?.(evt)
        }
      },
    })
    return () => sub.close()
  }, [pipelineId])

  // Persist width
  useEffect(() => {
    localStorage.setItem(WIDTH_KEY, String(width))
  }, [width])

  // Drag-resize
  const startDrag = useCallback(
    (e) => {
      e.preventDefault()
      const startX = e.clientX
      const startW = width
      const onMove = (ev) => {
        const delta = startX - ev.clientX // panel grows when dragging left
        const next = Math.min(MAX_W, Math.max(MIN_W, startW + delta))
        setWidth(next)
      }
      const onUp = () => {
        window.removeEventListener("mousemove", onMove)
        window.removeEventListener("mouseup", onUp)
      }
      window.addEventListener("mousemove", onMove)
      window.addEventListener("mouseup", onUp)
    },
    [width]
  )

  const filtered = useMemo(() => {
    return entries.filter((e) => {
      if (levelFilter !== "all" && e.level !== levelFilter) return false
      if (stepFilter !== "all" && e.step !== stepFilter) return false
      return true
    })
  }, [entries, levelFilter, stepFilter])

  const virtualizer = useVirtualizer({
    count: filtered.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => 22,
    overscan: 30,
  })

  // Auto-scroll to bottom unless paused.
  // Uses virtualizer API so it works with dynamic row heights (tracebacks
  // expand past the estimate, which breaks naive scrollTop=scrollHeight).
  useEffect(() => {
    if (pausedRef.current) return
    if (filtered.length === 0) return
    const id = requestAnimationFrame(() => {
      try {
        virtualizer.scrollToIndex(filtered.length - 1, { align: "end" })
      } catch {
        const el = scrollRef.current
        if (el) el.scrollTop = el.scrollHeight
      }
    })
    return () => cancelAnimationFrame(id)
  }, [filtered.length, virtualizer])

  const handleCopy = useCallback(async () => {
    const text = filtered
      .map((e) => `[${e.level}] ${e.step ? `(${e.step}) ` : ""}${e.message}`)
      .join("\n")
    // navigator.clipboard requires a secure context (HTTPS or localhost).
    // The app is served over plain HTTP from the LXC at 10.10.100.14 so we
    // fall back to a hidden <textarea> + execCommand("copy"), which works
    // in any context but needs a real DOM element on the page.
    if (window.isSecureContext && navigator.clipboard?.writeText) {
      try {
        await navigator.clipboard.writeText(text)
        toast.success("Logs copiados al portapapeles")
        return
      } catch {
        // fall through to execCommand fallback
      }
    }
    try {
      const ta = document.createElement("textarea")
      ta.value = text
      ta.setAttribute("readonly", "")
      ta.style.position = "fixed"
      ta.style.opacity = "0"
      document.body.appendChild(ta)
      ta.select()
      const ok = document.execCommand("copy")
      document.body.removeChild(ta)
      if (ok) toast.success("Logs copiados al portapapeles")
      else toast.error("No se pudo copiar")
    } catch {
      toast.error("No se pudo copiar")
    }
  }, [filtered])

  const handleClear = useCallback(() => {
    setEntries([])
    seenSeqRef.current = new Set()
  }, [])

  const stepNames = useMemo(() => steps.map((s) => s.name), [steps])

  return (
    <div
      className="relative flex shrink-0 flex-col self-start border-l border-zinc-800/80 bg-zinc-950"
      style={{ width, height: "calc(100vh - 180px)" }}
    >
      {/* drag handle */}
      <div
        role="separator"
        aria-orientation="vertical"
        title="Arrastra para redimensionar"
        onMouseDown={startDrag}
        className="absolute left-0 top-0 z-20 h-full w-1 cursor-col-resize bg-transparent transition hover:bg-amber-500/30"
      />

      {/* header */}
      <div className="flex items-center justify-between gap-2 border-b border-zinc-800/80 px-3 py-2">
        <div className="flex items-center gap-2 text-xs font-medium uppercase tracking-wide text-zinc-400">
          <Filter className="h-3.5 w-3.5" />
          Logs
          <span className="rounded bg-zinc-900 px-1.5 py-0.5 text-[10px] tabular-nums text-zinc-500">
            {filtered.length}/{entries.length}
          </span>
        </div>
        <div className="flex items-center gap-1">
          <Button
            size="icon"
            variant="ghost"
            onClick={() => setPaused((p) => !p)}
            title={paused ? "Reanudar auto-scroll" : "Pausar auto-scroll"}
            className="h-7 w-7"
          >
            {paused ? <Play className="h-3.5 w-3.5" /> : <Pause className="h-3.5 w-3.5" />}
          </Button>
          <Button
            size="icon"
            variant="ghost"
            onClick={handleCopy}
            title="Copiar logs"
            className="h-7 w-7"
          >
            <Copy className="h-3.5 w-3.5" />
          </Button>
          <Button
            size="icon"
            variant="ghost"
            onClick={handleClear}
            title="Limpiar"
            className="h-7 w-7"
          >
            <Trash2 className="h-3.5 w-3.5" />
          </Button>
        </div>
      </div>

      {/* filters */}
      <div className="flex items-center gap-2 border-b border-zinc-800/60 px-3 py-2 text-xs">
        <SelectInline value={levelFilter} onChange={setLevelFilter} options={[
          { v: "all", label: "Todos" },
          { v: "info", label: "Info" },
          { v: "warn", label: "Warn" },
          { v: "error", label: "Error" },
        ]} />
        <SelectInline value={stepFilter} onChange={setStepFilter} options={[
          { v: "all", label: "Todos los pasos" },
          ...stepNames.map((n) => ({ v: n, label: n })),
        ]} />
        {paused && (
          <span className="ml-auto rounded bg-amber-500/10 px-2 py-0.5 text-[10px] text-amber-400">
            Auto-scroll pausado
          </span>
        )}
      </div>

      {/* virtualized list */}
      <div
        ref={scrollRef}
        onScroll={(e) => {
          const el = e.currentTarget
          const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 24
          if (!atBottom && !pausedRef.current) setPaused(true)
          if (atBottom && pausedRef.current) setPaused(false)
        }}
        className="flex-1 overflow-auto font-mono text-[12px] leading-snug"
      >
        <div
          style={{
            height: virtualizer.getTotalSize(),
            width: "100%",
            position: "relative",
          }}
        >
          {virtualizer.getVirtualItems().map((vi) => {
            const e = filtered[vi.index]
            return (
              <div
                key={vi.key}
                ref={virtualizer.measureElement}
                data-index={vi.index}
                style={{
                  position: "absolute",
                  top: 0,
                  left: 0,
                  transform: `translateY(${vi.start}px)`,
                  width: "100%",
                }}
                className={cn(
                  "flex gap-2 border-b border-zinc-900/60 px-3 py-1",
                  LEVEL_STYLE[e.level]
                )}
              >
                <span className="w-12 shrink-0 text-zinc-600">[{e.level}]</span>
                {e.step && (
                  <span className="w-20 shrink-0 truncate text-zinc-500">{e.step}</span>
                )}
                <span className="whitespace-pre-wrap break-words">{e.message}</span>
              </div>
            )
          })}
        </div>
        {filtered.length === 0 && (
          <div className="flex h-full items-center justify-center p-6 text-center text-xs text-zinc-600">
            Esperando logs…
          </div>
        )}
      </div>
    </div>
  )
}

function SelectInline({ value, onChange, options }) {
  return (
    <label className="relative inline-flex items-center">
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="appearance-none rounded border border-zinc-800 bg-zinc-900 py-1 pl-2 pr-6 text-xs text-zinc-200 outline-none focus:border-amber-500/60"
      >
        {options.map((o) => (
          <option key={o.v} value={o.v}>
            {o.label}
          </option>
        ))}
      </select>
      <ChevronDown className="pointer-events-none absolute right-1.5 h-3 w-3 text-zinc-500" />
    </label>
  )
}

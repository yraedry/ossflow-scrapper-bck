// Vertical step timeline for pipeline detail.
// Visual states: pending / running / success(completed) / error(failed) / skipped / cancelled.
// Animates state transitions with framer-motion and shows per-step duration + ETA when running.
import React from "react"
import { motion, AnimatePresence } from "framer-motion"
import {
  Circle,
  Loader2,
  CheckCircle2,
  XCircle,
  MinusCircle,
  Scissors,
  Captions,
  Languages,
  Mic,
} from "lucide-react"
import { cn } from "@/lib/utils"

const STEP_ICONS = {
  chapters: Scissors,
  subtitles: Captions,
  translate: Languages,
  dubbing: Mic,
}

// Human-readable Spanish labels. The backend step name is the key.
// "translate" with dubbing_mode=true gets a distinct label so the user can
// tell iso-sync adaptation apart from literal translation at a glance.
const STEP_LABELS = {
  chapters: 'Capítulos',
  subtitles: 'Subtítulos',
  translate: 'Traducción',
  translate_dub: 'Guion doblaje',
  dubbing: 'Doblaje',
}

function displayStepName(step) {
  const isDubScript =
    step.name === 'translate' && step.options?.dubbing_mode === true
  const key = isDubScript ? 'translate_dub' : step.name
  return STEP_LABELS[key] || step.name
}

function fmtDuration(sec) {
  if (!sec || sec < 0) return null
  const s = Math.round(sec)
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  const r = s % 60
  if (m < 60) return r ? `${m}m ${r}s` : `${m}m`
  const h = Math.floor(m / 60)
  return `${h}h ${m % 60}m`
}

function durationFor(step) {
  if (!step.started_at) return null
  const start = new Date(step.started_at).getTime()
  const end = step.completed_at ? new Date(step.completed_at).getTime() : Date.now()
  return Math.max(0, (end - start) / 1000)
}

function statusToVisual(status) {
  switch (status) {
    case "running":
      return {
        Icon: Loader2,
        ring: "ring-amber-500/40",
        bg: "bg-amber-500/10",
        text: "text-amber-400",
        line: "bg-amber-500/40",
        pulse: true,
        spin: true,
        label: "En curso",
      }
    case "completed":
      return {
        Icon: CheckCircle2,
        ring: "ring-emerald-500/40",
        bg: "bg-emerald-500/10",
        text: "text-emerald-400",
        line: "bg-emerald-500/50",
        label: "OK",
      }
    case "failed":
      return {
        Icon: XCircle,
        ring: "ring-red-500/40",
        bg: "bg-red-500/10",
        text: "text-red-400",
        line: "bg-red-500/40",
        label: "Error",
      }
    case "skipped":
      return {
        Icon: MinusCircle,
        ring: "ring-zinc-700",
        bg: "bg-zinc-900",
        text: "text-zinc-500",
        line: "bg-zinc-800",
        label: "Saltado",
      }
    case "cancelled":
      return {
        Icon: XCircle,
        ring: "ring-zinc-600",
        bg: "bg-zinc-900",
        text: "text-zinc-400",
        line: "bg-zinc-800",
        label: "Cancelado",
      }
    default:
      return {
        Icon: Circle,
        ring: "ring-zinc-700",
        bg: "bg-zinc-900",
        text: "text-zinc-500",
        line: "bg-zinc-800",
        label: "Pendiente",
      }
  }
}

export default function StepTimeline({ steps = [], eta = null, children }) {
  return (
    <ol className="relative flex flex-col">
      {steps.map((step, i) => {
        const visual = statusToVisual(step.status)
        const StepIcon = STEP_ICONS[step.name] || Circle
        const dur = durationFor(step)
        const etaSec = eta?.per_step?.[step.name]
        const isLast = i === steps.length - 1
        return (
          <li key={`${step.name}-${i}`} className="relative flex gap-4 pb-6 last:pb-0">
            {/* connector line */}
            {!isLast && (
              <span
                aria-hidden
                className={cn(
                  "absolute left-[18px] top-10 h-[calc(100%-2.5rem)] w-px",
                  statusToVisual(step.status === "completed" ? "completed" : "pending").line
                )}
              />
            )}
            {/* node */}
            <AnimatePresence mode="wait">
              <motion.div
                key={step.status}
                initial={{ scale: 0.85, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                exit={{ scale: 0.85, opacity: 0 }}
                transition={{ type: "spring", stiffness: 350, damping: 22 }}
                className={cn(
                  "relative z-10 flex h-9 w-9 shrink-0 items-center justify-center rounded-full ring-1",
                  visual.ring,
                  visual.bg,
                  visual.pulse && "animate-pulse"
                )}
              >
                <visual.Icon
                  className={cn("h-4 w-4", visual.text, visual.spin && "animate-spin")}
                />
              </motion.div>
            </AnimatePresence>

            {/* content */}
            <div className="flex-1 pt-1.5">
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <StepIcon className="h-3.5 w-3.5 text-zinc-500" />
                  <h3 className="text-sm font-medium text-zinc-100">
                    {displayStepName(step)}
                  </h3>
                  <span
                    className={cn(
                      "rounded-full px-2 py-0.5 text-[10px] font-medium uppercase tracking-wide",
                      visual.bg,
                      visual.text
                    )}
                  >
                    {visual.label}
                  </span>
                </div>
                <div className="text-xs tabular-nums text-zinc-500">
                  {step.status === "running" && etaSec
                    ? `~${fmtDuration(etaSec)}`
                    : fmtDuration(dur) || "—"}
                </div>
              </div>
              {step.status === "running" && (
                <div className="mt-2 h-1 overflow-hidden rounded-full bg-zinc-900">
                  <motion.div
                    className="h-full bg-amber-500"
                    initial={{ width: 0 }}
                    animate={{ width: `${Math.max(2, step.progress || 0)}%` }}
                    transition={{ duration: 0.4 }}
                  />
                </div>
              )}
              {step.message && (
                <p className="mt-1.5 line-clamp-2 text-xs text-zinc-400">{step.message}</p>
              )}
              {/* slot for diff under the step */}
              {typeof children === "function" ? children(step, i) : null}
            </div>
          </li>
        )
      })}
    </ol>
  )
}

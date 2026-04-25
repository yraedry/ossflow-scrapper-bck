// QA tab — aggregate view of automatic dubbing quality across all
// chapters of an instructional. Reads /api/dubbing/qa/instructional/{name}
// (one sidecar read per chapter, cheap) and renders:
//
//  1. Headline stats (avg MOS, verdict breakdown, worst chapter).
//  2. Filtered chapter table with verdict, MOS, boundary counts and a
//     "Detalle" button that opens a drawer with the full boundary list.
//  3. Empty state if no chapter has been dubbed yet.
import { useMemo, useState } from 'react'
import {
  AlertTriangle,
  CheckCircle2,
  Download,
  Gauge,
  Loader2,
  Music,
  XCircle,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/Badge'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { cn } from '@/lib/utils'
import { useInstructionalQa } from '../api/useDubQa'

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

function formatTs(ms) {
  if (ms == null) return '--:--'
  const total = Math.max(0, Math.round(ms / 1000))
  const m = Math.floor(total / 60)
  const s = total % 60
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

function seasonEpisodeCode(filename) {
  if (!filename) return ''
  const m = filename.match(/S\d{2}E\d{2,3}/i)
  return m ? m[0].toUpperCase() : ''
}

const LEVEL_META = {
  green: {
    cls: 'border-emerald-500/40 bg-emerald-500/15 text-emerald-400',
    Icon: CheckCircle2,
  },
  amber: {
    cls: 'border-amber-500/40 bg-amber-500/15 text-amber-400',
    Icon: AlertTriangle,
  },
  red: {
    cls: 'border-red-500/40 bg-red-500/15 text-red-400',
    Icon: XCircle,
  },
}

// ----------------------------------------------------------------------
// Component
// ----------------------------------------------------------------------

export default function QaTab({ instructional }) {
  const name = instructional?.name
  const { data, isLoading, error } = useInstructionalQa(name)
  const [filter, setFilter] = useState('all')  // all | red | amber | missing
  const [activeChapter, setActiveChapter] = useState(null)

  const rows = useMemo(() => {
    const all = data?.chapters || []
    if (filter === 'all') return all
    if (filter === 'missing') return all.filter((c) => !c.qa)
    return all.filter((c) => c.qa?.verdict?.level === filter)
  }, [data, filter])

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-10 text-zinc-400">
        <Loader2 className="mr-2 h-4 w-4 animate-spin" />
        Analizando calidad…
      </div>
    )
  }

  if (error) {
    return (
      <div className="rounded-lg border border-red-500/40 bg-red-500/10 p-4 text-sm text-red-300">
        Error cargando QA: {error?.message || 'desconocido'}
      </div>
    )
  }

  if (!data || !data.chapters?.length) {
    return (
      <EmptyState message="Este instruccional no tiene capítulos dub-eados aún." />
    )
  }

  const withQa = data.summary?.with_qa || 0
  if (withQa === 0) {
    return (
      <EmptyState message="Ningún capítulo tiene sidecar QA todavía. Re-dobla un capítulo y vuelve." />
    )
  }

  const exportAll = () => {
    const blob = new Blob([JSON.stringify(data, null, 2)], {
      type: 'application/json',
    })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `${name}.qa-summary.json`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between gap-2">
        <h2 className="text-sm font-medium uppercase tracking-wide text-zinc-500">
          QA automático del doblaje
        </h2>
        <Button size="sm" variant="ghost" onClick={exportAll} title="Descargar JSON agregado">
          <Download className="mr-1 h-3 w-3" />
          Exportar todo
        </Button>
      </div>
      <Summary summary={data.summary} />
      <FilterBar
        filter={filter}
        setFilter={setFilter}
        counts={computeCounts(data.chapters)}
      />
      <ChaptersTable
        rows={rows}
        onOpen={(row) => setActiveChapter(row)}
      />
      <DetailDialog
        chapter={activeChapter}
        onClose={() => setActiveChapter(null)}
      />
    </div>
  )
}

// ----------------------------------------------------------------------
// Subcomponents
// ----------------------------------------------------------------------

function EmptyState({ message }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 rounded-lg border border-zinc-800 bg-zinc-900/40 py-10 text-sm text-zinc-500">
      <Music className="h-8 w-8 opacity-60" />
      {message}
    </div>
  )
}

function Summary({ summary }) {
  const avg = summary.avg_mos
  const levels = summary.levels || {}
  const total = summary.total_chapters || 0
  const worst = summary.worst

  return (
    <div className="grid grid-cols-1 gap-3 md:grid-cols-4">
      <StatTile
        label="MOS promedio"
        value={avg != null ? avg.toFixed(2) : '—'}
        hint="1 malo · 5 natural"
        tone={avg == null ? 'muted' : avg < 3 ? 'red' : avg < 3.5 ? 'amber' : 'green'}
      />
      <StatTile
        label="Con QA"
        value={`${summary.with_qa}/${total}`}
        hint="Sidecars disponibles"
      />
      <StatTile
        label="Verdictos"
        value={
          <div className="flex gap-2 text-sm">
            <LevelPill level="green" count={levels.green || 0} />
            <LevelPill level="amber" count={levels.amber || 0} />
            <LevelPill level="red" count={levels.red || 0} />
          </div>
        }
      />
      <StatTile
        label="Peor capítulo"
        value={worst?.filename ? seasonEpisodeCode(worst.filename) || worst.filename : '—'}
        hint={worst?.mos != null ? `MOS ${worst.mos.toFixed(2)}` : ''}
        tone={worst ? 'red' : 'muted'}
      />
    </div>
  )
}

function StatTile({ label, value, hint, tone = 'default' }) {
  const toneCls = {
    red: 'text-red-400',
    amber: 'text-amber-400',
    green: 'text-emerald-400',
    muted: 'text-zinc-500',
    default: 'text-zinc-100',
  }[tone]
  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-900/60 px-3 py-2">
      <div className="text-[11px] uppercase tracking-wide text-zinc-500">{label}</div>
      <div className={cn('mt-1 text-lg font-semibold', toneCls)}>
        {typeof value === 'string' || typeof value === 'number' ? value : value}
      </div>
      {hint && <div className="mt-0.5 text-[11px] text-zinc-500">{hint}</div>}
    </div>
  )
}

function LevelPill({ level, count }) {
  const { cls, Icon } = LEVEL_META[level] || LEVEL_META.green
  return (
    <span className={cn('inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs', cls)}>
      <Icon className="h-3 w-3" />
      {count}
    </span>
  )
}

function FilterBar({ filter, setFilter, counts }) {
  const buttons = [
    { id: 'all', label: `Todos (${counts.total})` },
    { id: 'red', label: `Rojos (${counts.red})` },
    { id: 'amber', label: `Ámbar (${counts.amber})` },
    { id: 'missing', label: `Sin QA (${counts.missing})` },
  ]
  return (
    <div className="flex flex-wrap gap-2">
      {buttons.map((b) => (
        <Button
          key={b.id}
          size="sm"
          variant={filter === b.id ? 'default' : 'ghost'}
          onClick={() => setFilter(b.id)}
        >
          {b.label}
        </Button>
      ))}
    </div>
  )
}

function computeCounts(chapters) {
  const out = { total: chapters.length, red: 0, amber: 0, green: 0, missing: 0 }
  for (const c of chapters) {
    if (!c.qa) out.missing += 1
    else {
      const lvl = c.qa.verdict?.level
      if (lvl && out[lvl] != null) out[lvl] += 1
    }
  }
  return out
}

function ChaptersTable({ rows, onOpen }) {
  if (!rows.length) {
    return (
      <div className="rounded-lg border border-zinc-800 bg-zinc-900/40 p-4 text-center text-sm text-zinc-500">
        Sin capítulos que coincidan con el filtro.
      </div>
    )
  }
  return (
    <div className="overflow-hidden rounded-lg border border-zinc-800">
      <table className="w-full text-left text-sm">
        <thead className="bg-zinc-900/60 text-xs uppercase tracking-wide text-zinc-500">
          <tr>
            <th className="px-3 py-2">Episodio</th>
            <th className="px-3 py-2">Veredicto</th>
            <th className="px-3 py-2">MOS</th>
            <th className="px-3 py-2">Cortes duros</th>
            <th className="px-3 py-2">Avisos</th>
            <th className="px-3 py-2" />
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <Row key={r.path} row={r} onOpen={onOpen} />
          ))}
        </tbody>
      </table>
    </div>
  )
}

function Row({ row, onOpen }) {
  const code = seasonEpisodeCode(row.filename)
  const qa = row.qa
  const verdict = qa?.verdict || {}
  const level = verdict.level
  const meta = level && LEVEL_META[level]

  return (
    <tr className="border-t border-zinc-800/60 hover:bg-zinc-900/40">
      <td className="px-3 py-2">
        <div className="font-mono text-xs text-zinc-500">{code || '—'}</div>
        <div className="truncate text-zinc-200">{row.filename}</div>
      </td>
      <td className="px-3 py-2">
        {meta ? (
          <span
            className={cn(
              'inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs',
              meta.cls,
            )}
          >
            <meta.Icon className="h-3 w-3" />
            {level.toUpperCase()}
          </span>
        ) : (
          <span className="text-xs text-zinc-500">sin QA</span>
        )}
      </td>
      <td className="px-3 py-2 font-mono">
        {verdict.mos != null ? verdict.mos.toFixed(2) : '—'}
      </td>
      <td className="px-3 py-2">{verdict.hard_cuts ?? '—'}</td>
      <td className="px-3 py-2">{verdict.warnings ?? '—'}</td>
      <td className="px-3 py-2 text-right">
        {qa && (
          <Button size="sm" variant="ghost" onClick={() => onOpen(row)}>
            Detalle
          </Button>
        )}
      </td>
    </tr>
  )
}

function DetailDialog({ chapter, onClose }) {
  const open = Boolean(chapter)
  const qa = chapter?.qa
  const boundaries = qa?.boundaries?.issues || []
  const problematic = boundaries.filter((b) => b.severity !== 'ok')

  const download = () => {
    if (!qa) return
    const payload = {
      filename: chapter.filename,
      path: chapter.path,
      qa,
    }
    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: 'application/json',
    })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    // `filename` puede venir vacío/null del backend si el scan cache está
    // corrupto — sin este fallback `.replace()` crashaba la UI.
    const safeFilename = chapter.filename || 'chapter'
    a.download = `${safeFilename.replace(/\.[^.]+$/, '')}.dub-qa.json`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const copyClipboard = async () => {
    if (!qa) return
    try {
      await navigator.clipboard.writeText(
        JSON.stringify({ filename: chapter.filename, qa }, null, 2),
      )
    } catch {
      /* clipboard blocked (no HTTPS) — fall back silently; user still has Download */
    }
  }

  return (
    <Dialog open={open} onOpenChange={(v) => !v && onClose()}>
      <DialogContent className="max-w-3xl">
        <DialogHeader>
          <DialogTitle className="flex items-center justify-between gap-2">
            <span className="flex items-center gap-2">
              <Gauge className="h-4 w-4" />
              QA — {chapter?.filename}
            </span>
            {qa && (
              <span className="flex items-center gap-1 pr-6">
                <Button size="sm" variant="ghost" onClick={copyClipboard} title="Copiar JSON">
                  Copiar
                </Button>
                <Button size="sm" variant="ghost" onClick={download} title="Descargar JSON">
                  <Download className="mr-1 h-3 w-3" />
                  JSON
                </Button>
              </span>
            )}
          </DialogTitle>
        </DialogHeader>
        {qa ? (
          <div className="space-y-4">
            <div className="grid grid-cols-3 gap-3">
              <StatTile
                label="Veredicto"
                value={(qa.verdict?.level || '—').toUpperCase()}
                tone={qa.verdict?.level}
              />
              <StatTile
                label="MOS"
                value={qa.verdict?.mos != null ? qa.verdict.mos.toFixed(2) : '—'}
                hint={qa.mos?.model_name || 'UTMOS no disponible'}
              />
              <StatTile
                label="Transiciones"
                value={`${qa.verdict?.hard_cuts ?? 0} duras / ${qa.verdict?.warnings ?? 0} avisos`}
                hint={`${qa.boundaries?.total_boundaries ?? 0} totales`}
              />
            </div>
            <IssuesTable issues={problematic} />
          </div>
        ) : (
          <p className="text-sm text-zinc-500">Sin datos de QA.</p>
        )}
      </DialogContent>
    </Dialog>
  )
}

function IssuesTable({ issues }) {
  if (!issues.length) {
    return (
      <p className="text-sm text-emerald-400">
        Sin transiciones problemáticas en este capítulo. 🎧
      </p>
    )
  }
  return (
    <div className="max-h-80 overflow-y-auto rounded-lg border border-zinc-800">
      <table className="w-full text-left text-xs">
        <thead className="sticky top-0 bg-zinc-900 text-zinc-400">
          <tr>
            <th className="px-3 py-2">Timestamp</th>
            <th className="px-3 py-2">Severidad</th>
            <th className="px-3 py-2">Gap</th>
            <th className="px-3 py-2">RMS</th>
            <th className="px-3 py-2">Timbre</th>
            <th className="px-3 py-2">Pitch</th>
            <th className="px-3 py-2">Motivos</th>
          </tr>
        </thead>
        <tbody>
          {issues.map((b) => (
            <tr
              key={b.index}
              className="border-t border-zinc-800/60 hover:bg-zinc-900/40"
            >
              <td className="px-3 py-1.5 font-mono">{formatTs(b.timestamp_ms)}</td>
              <td className="px-3 py-1.5">
                <Badge
                  variant="secondary"
                  className={cn(
                    b.severity === 'hard' && 'bg-red-500/15 text-red-400',
                    b.severity === 'warn' && 'bg-amber-500/15 text-amber-400',
                  )}
                >
                  {b.severity}
                </Badge>
              </td>
              <td className="px-3 py-1.5">{b.gap_ms} ms</td>
              <td className="px-3 py-1.5">
                {b.rms_jump_db != null ? `${b.rms_jump_db.toFixed(1)} dB` : '—'}
              </td>
              <td className="px-3 py-1.5">
                {b.centroid_jump_hz != null ? `${Math.round(b.centroid_jump_hz)} Hz` : '—'}
              </td>
              <td className="px-3 py-1.5">
                {b.f0_jump_hz != null ? `${Math.round(b.f0_jump_hz)} Hz` : '—'}
              </td>
              <td className="px-3 py-1.5 text-zinc-300">
                {(b.reasons || []).join(' · ') || '—'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

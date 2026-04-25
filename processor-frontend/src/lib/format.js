// Small presentation helpers shared across feature hooks & components.
// Kept framework-agnostic (plain JS) so they can be imported anywhere.

export function formatBytes(bytes) {
  if (bytes == null || Number.isNaN(bytes)) return '—'
  const n = Number(bytes)
  if (n < 1024) return `${n} B`
  const units = ['KB', 'MB', 'GB', 'TB', 'PB']
  let v = n / 1024
  let i = 0
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i += 1 }
  return `${v.toFixed(v >= 10 ? 0 : 1)} ${units[i]}`
}

export function formatDuration(seconds) {
  if (seconds == null || Number.isNaN(seconds)) return '—'
  const s = Math.max(0, Math.floor(Number(seconds)))
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const sec = s % 60
  if (h > 0) return `${h}h ${String(m).padStart(2, '0')}m`
  if (m > 0) return `${m}m ${String(sec).padStart(2, '0')}s`
  return `${sec}s`
}

export function formatTimestamp(iso, { seconds = false } = {}) {
  if (!iso) return '—'
  const d = iso instanceof Date ? iso : new Date(iso)
  if (Number.isNaN(d.getTime())) return '—'
  const opts = {
    year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit',
    ...(seconds ? { second: '2-digit' } : {}),
  }
  return d.toLocaleString(undefined, opts)
}

export function formatRelativeTime(iso) {
  if (!iso) return '—'
  const d = iso instanceof Date ? iso : new Date(iso)
  if (Number.isNaN(d.getTime())) return '—'
  const diffSec = (Date.now() - d.getTime()) / 1000
  const abs = Math.abs(diffSec)
  const sign = diffSec >= 0 ? 'ago' : 'from now'
  if (abs < 60) return `${Math.floor(abs)}s ${sign}`
  if (abs < 3600) return `${Math.floor(abs / 60)}m ${sign}`
  if (abs < 86400) return `${Math.floor(abs / 3600)}h ${sign}`
  if (abs < 604800) return `${Math.floor(abs / 86400)}d ${sign}`
  return formatTimestamp(iso)
}

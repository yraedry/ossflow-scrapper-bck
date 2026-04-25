// Time helpers for Oracle chapter editor (HH:MM:SS / MM:SS)
export function parseTime(str) {
  if (str == null) return null
  const s = String(str).trim()
  if (!s) return null
  const m = s.match(/^(?:(\d+):)?(\d{1,2}):(\d{2})(?:\.(\d+))?$/)
  if (!m) return null
  const h = m[1] ? parseInt(m[1], 10) : 0
  const mm = parseInt(m[2], 10)
  const ss = parseInt(m[3], 10)
  if (mm >= 60 || ss >= 60) return null
  const frac = m[4] ? parseFloat(`0.${m[4]}`) : 0
  return h * 3600 + mm * 60 + ss + frac
}

export function formatTime(seconds) {
  if (seconds == null || isNaN(seconds)) return ''
  const total = Math.max(0, Math.round(seconds))
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const s = total % 60
  const pad = (n) => String(n).padStart(2, '0')
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${pad(m)}:${pad(s)}`
}

export function validateOracle(oracle) {
  const errors = []
  if (!oracle?.volumes?.length) return errors
  oracle.volumes.forEach((vol, vi) => {
    vol.chapters.forEach((ch, ci) => {
      const start = ch.start_s
      const end = ch.end_s
      if (start == null || isNaN(start)) errors.push({ vi, ci, field: 'start', msg: 'start inválido' })
      if (end == null || isNaN(end)) errors.push({ vi, ci, field: 'end', msg: 'end inválido' })
      if (start != null && end != null && !(end > start)) errors.push({ vi, ci, field: 'end', msg: 'end debe ser > start' })
    })
  })
  return errors
}

export function deepClone(obj) {
  return JSON.parse(JSON.stringify(obj))
}

export function oracleEqual(a, b) {
  return JSON.stringify(a) === JSON.stringify(b)
}

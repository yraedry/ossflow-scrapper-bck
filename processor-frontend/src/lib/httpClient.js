// Shared HTTP client for the v3 data layer.
// Thin wrapper over fetch() that:
//   - Prefixes every request with `/api` (Vite proxy → processor-api).
//   - Encodes JSON bodies (unless `body` is FormData).
//   - Raises a structured `HttpError` instead of a generic `Error`.
//   - Threads an optional `AbortSignal` through to fetch().
//
// This is the only place in the feature hooks where fetch() should be called
// directly; every `useQuery`/`useMutation` consumer calls `http.*` so errors,
// headers and base URL are handled uniformly.

const BASE = '/api'

export class HttpError extends Error {
  constructor(status, message, body) {
    super(message || `HTTP ${status}`)
    this.name = 'HttpError'
    this.status = status
    this.body = body
  }
}

async function parseBody(res) {
  const ct = res.headers.get('content-type') || ''
  if (res.status === 204) return null
  if (ct.includes('application/json')) {
    return res.json().catch(() => null)
  }
  return res.text().catch(() => null)
}

async function request(path, { method = 'GET', body, headers, signal, raw = false } = {}) {
  const url = path.startsWith('http') ? path : `${BASE}${path}`
  const isForm = typeof FormData !== 'undefined' && body instanceof FormData
  const finalHeaders = { ...(headers || {}) }
  let finalBody = body
  if (body != null && !isForm && typeof body !== 'string') {
    finalBody = JSON.stringify(body)
    if (!finalHeaders['Content-Type']) finalHeaders['Content-Type'] = 'application/json'
  }
  const res = await fetch(url, { method, headers: finalHeaders, body: finalBody, signal })
  if (!res.ok) {
    const errBody = await parseBody(res)
    const msg = (errBody && (errBody.detail || errBody.message || errBody.error)) || `HTTP ${res.status}`
    throw new HttpError(res.status, msg, errBody)
  }
  if (raw) return res
  return parseBody(res)
}

export const http = {
  get: (path, opts) => request(path, { ...opts, method: 'GET' }),
  post: (path, body, opts) => request(path, { ...opts, method: 'POST', body }),
  put: (path, body, opts) => request(path, { ...opts, method: 'PUT', body }),
  patch: (path, body, opts) => request(path, { ...opts, method: 'PATCH', body }),
  del: (path, opts) => request(path, { ...opts, method: 'DELETE' }),
  raw: request,
}

export function qs(params = {}) {
  const sp = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) {
    if (v == null || v === '') continue
    if (Array.isArray(v)) sp.set(k, v.join(','))
    else sp.set(k, String(v))
  }
  const s = sp.toString()
  return s ? `?${s}` : ''
}

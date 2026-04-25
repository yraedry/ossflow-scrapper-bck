// Thin wrapper over EventSource used by pipeline/job detail views.
// Backend endpoints (all under the /api proxy):
//   GET /api/pipeline/{id}/events
//   GET /api/jobs/{id}/events
//   GET /api/telegram/.../events
//
// Handles:
//   - JSON parsing of each `message` event.
//   - Terminal states (complete/failed) → stop retry, close source.
//   - Auto-reconnect with exponential backoff, up to 3 attempts.
//
// Returns { close() } so the caller can tear down (e.g. inside a useEffect
// cleanup).

const MAX_RETRIES = 3

export function subscribeSSE(url, { onMessage, onError, onOpen } = {}) {
  let retries = 0
  let terminated = false
  let source = null

  const open = () => {
    source = new EventSource(url)
    source.onopen = (e) => onOpen?.(e)
    source.onmessage = (e) => {
      let data
      try { data = JSON.parse(e.data) } catch { data = { raw: e.data } }
      onMessage?.(data)
      const t = data?.type
      const status = data?.status
      if (
        t === 'complete' || t === 'pipeline_completed' ||
        t === 'error' || t === 'pipeline_failed' ||
        status === 'completed' || status === 'failed'
      ) {
        terminated = true
        source?.close()
      }
    }
    source.onerror = (e) => {
      if (terminated) { source?.close(); return }
      if (source?.readyState === EventSource.CLOSED && retries < MAX_RETRIES) {
        retries += 1
        setTimeout(open, 500 * Math.pow(2, retries - 1))
        return
      }
      onError?.(e)
      source?.close()
    }
  }

  open()
  return {
    close: () => { terminated = true; source?.close() },
    get readyState() { return source?.readyState ?? EventSource.CLOSED },
  }
}

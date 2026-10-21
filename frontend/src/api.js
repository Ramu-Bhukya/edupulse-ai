// Small wrapper around the FastAPI backend (see api/main.py).

// Empty by default: in development Vite proxies /api, and in production the
// API serves the built app itself. Set VITE_API_BASE when they live apart.
const BASE = import.meta.env.VITE_API_BASE ?? ''

function headers(apiKey, json = true) {
  const h = {}
  if (json) h['Content-Type'] = 'application/json'
  if (apiKey) h['X-OpenAI-Key'] = apiKey
  return h
}

async function toError(response) {
  let message = `Request failed (${response.status})`
  try {
    const body = await response.json()
    if (typeof body.detail === 'string') message = body.detail
    else if (Array.isArray(body.detail)) message = body.detail.map((d) => d.msg).join(', ')
  } catch {
    // not JSON, keep the generic message
  }
  return new Error(message)
}

async function request(path, { apiKey, method = 'GET', body } = {}) {
  const response = await fetch(BASE + path, {
    method,
    headers: headers(apiKey, body !== undefined),
    body: body === undefined ? undefined : JSON.stringify(body),
  })
  if (!response.ok) throw await toError(response)
  return response.status === 204 ? null : response.json()
}

export const getPresets = () => request('/api/presets')

export const listDocuments = () => request('/api/documents')

export const deleteDocument = (docId) => request(`/api/documents/${docId}`, { method: 'DELETE' })

export const generateQuiz = (docId, { topic, n }, apiKey) =>
  request(`/api/documents/${docId}/quiz`, { method: 'POST', apiKey, body: { topic, n } })

export const generateFlashcards = (docId, { topic, n }, apiKey) =>
  request(`/api/documents/${docId}/flashcards`, { method: 'POST', apiKey, body: { topic, n } })

export async function uploadDocument(file, apiKey) {
  const form = new FormData()
  form.append('file', file)
  const response = await fetch(`${BASE}/api/documents`, {
    method: 'POST',
    headers: headers(apiKey, false),
    body: form,
  })
  if (!response.ok) throw await toError(response)
  return response.json()
}

/**
 * POST /api/chat and read the NDJSON stream line by line.
 * Calls onSources once, then onToken for every chunk of text.
 */
export async function streamChat(body, { apiKey, onSources, onToken, signal }) {
  const response = await fetch(`${BASE}/api/chat`, {
    method: 'POST',
    headers: headers(apiKey),
    body: JSON.stringify(body),
    signal,
  })
  if (!response.ok) throw await toError(response)

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  const handle = (line) => {
    if (!line.trim()) return
    const event = JSON.parse(line)
    if (event.type === 'sources') onSources?.(event.sources)
    else if (event.type === 'token') onToken?.(event.text)
    else if (event.type === 'error') throw new Error(event.message)
  }

  while (true) {
    const { value, done } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    let newline
    while ((newline = buffer.indexOf('\n')) >= 0) {
      handle(buffer.slice(0, newline))
      buffer = buffer.slice(newline + 1)
    }
  }
  handle(buffer)
}

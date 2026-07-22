const configuredBaseUrl = import.meta.env?.VITE_API_BASE_URL || import.meta.env?.VITE_API_BASE
const API_BASE_URL = (configuredBaseUrl || (import.meta.env?.DEV ? 'http://127.0.0.1:8000' : '')).replace(/\/$/, '')

export async function requestJson(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, options)
  let payload = null

  try {
    payload = await response.json()
  } catch {
    payload = null
  }

  if (!response.ok) {
    throw new Error(payload?.detail || `Request failed with status ${response.status}`)
  }

  return payload
}

export const getModelCard = () => requestJson('/api/model-card')
export const getModelHealth = () => requestJson('/api/model-health')
export const getMetrics = () => requestJson('/api/metrics')

export const predictVehicle = payload => requestJson('/api/predict', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(payload),
})

export const getHistory = (limit = 20) => requestJson(`/api/history?limit=${limit}`)

export const askAssistant = message => requestJson('/api/assistant', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ message }),
})

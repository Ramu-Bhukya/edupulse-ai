import { useCallback, useEffect, useState } from 'react'
import { getPresets, listDocuments } from './api.js'
import ChatTab from './components/ChatTab.jsx'
import DocumentPanel from './components/DocumentPanel.jsx'
import FlashcardsTab from './components/FlashcardsTab.jsx'
import QuizTab from './components/QuizTab.jsx'

const TABS = [
  { id: 'chat', label: '💬 Chat with Notes' },
  { id: 'quiz', label: '📝 Quiz Generator' },
  { id: 'cards', label: '🃏 Flashcards' },
]

export default function App() {
  const [presets, setPresets] = useState({ personas: [], tones: [] })
  const [persona, setPersona] = useState('')
  const [tone, setTone] = useState('')
  const [apiKey, setApiKey] = useState('') // only kept in memory, never stored
  const [documents, setDocuments] = useState([])
  const [activeDoc, setActiveDoc] = useState(null)
  const [tab, setTab] = useState('chat')
  const [messages, setMessages] = useState([])
  const [backendError, setBackendError] = useState('')

  const refreshDocuments = useCallback(async () => {
    try {
      setDocuments(await listDocuments())
    } catch (err) {
      setBackendError(err.message)
    }
  }, [])

  useEffect(() => {
    getPresets()
      .then((data) => {
        setPresets(data)
        setPersona(data.personas[0]?.name ?? '')
        setTone(data.tones[0] ?? '')
      })
      .catch(() =>
        setBackendError(
          "Can't reach the EduPulse API. Start it with: uvicorn api.main:app --port 8000",
        ),
      )
    refreshDocuments()
  }, [refreshDocuments])

  function selectDocument(doc) {
    // answers about the previous document would be confusing, start over
    if (doc?.doc_id !== activeDoc?.doc_id) setMessages([])
    setActiveDoc(doc)
  }

  const personaInfo = presets.personas.find((p) => p.name === persona)

  return (
    <div className="app">
      <aside className="sidebar">
        <h1>📚 EduPulse AI</h1>
        <p className="muted small">Your personal AI study companion.</p>

        <DocumentPanel
          documents={documents}
          activeDoc={activeDoc}
          apiKey={apiKey}
          onSelect={selectDocument}
          onLibraryChange={refreshDocuments}
        />

        <section className="panel">
          <h2>Your tutor</h2>
          <label>
            Teaching style
            <select value={persona} onChange={(e) => setPersona(e.target.value)}>
              {presets.personas.map((p) => (
                <option key={p.name}>{p.name}</option>
              ))}
            </select>
          </label>
          {personaInfo && <p className="muted small">{personaInfo.description}</p>}
          <label>
            Tone
            <select value={tone} onChange={(e) => setTone(e.target.value)}>
              {presets.tones.map((t) => (
                <option key={t}>{t}</option>
              ))}
            </select>
          </label>
        </section>

        <section className="panel">
          <details>
            <summary>OpenAI API key</summary>
            <label>
              Use your own key (optional)
              <input
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder="sk-..."
                autoComplete="off"
              />
            </label>
            <p className="muted small">
              Sent with each request, never saved. Leave empty to use the server&apos;s key.
            </p>
          </details>
        </section>
      </aside>

      <main className="main">
        {backendError && <div className="banner error">{backendError}</div>}

        <nav className="tabs" role="tablist">
          {TABS.map((t) => (
            <button
              key={t.id}
              role="tab"
              aria-selected={tab === t.id}
              className={tab === t.id ? 'tab active' : 'tab'}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </nav>

        {/* Tabs stay mounted (just hidden) so a half-finished quiz survives a
            look at the chat. Keying by document resets them when it changes. */}
        <section className="tab-panel" hidden={tab !== 'chat'}>
          <ChatTab
            messages={messages}
            setMessages={setMessages}
            activeDoc={activeDoc}
            persona={persona}
            tone={tone}
            apiKey={apiKey}
          />
        </section>
        <section className="tab-panel" hidden={tab !== 'quiz'}>
          <QuizTab key={activeDoc?.doc_id ?? 'none'} activeDoc={activeDoc} apiKey={apiKey} />
        </section>
        <section className="tab-panel" hidden={tab !== 'cards'}>
          <FlashcardsTab key={activeDoc?.doc_id ?? 'none'} activeDoc={activeDoc} apiKey={apiKey} />
        </section>
      </main>
    </div>
  )
}

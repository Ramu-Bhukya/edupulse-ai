import { useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { streamChat } from '../api.js'

const IDK = "i don't know"

export default function ChatTab({ messages, setMessages, activeDoc, persona, tone, apiKey }) {
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [error, setError] = useState('')
  const abortRef = useRef(null)
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages])

  const updateLast = (fn) =>
    setMessages((current) => [...current.slice(0, -1), fn(current[current.length - 1])])

  async function send(event) {
    event?.preventDefault()
    const question = input.trim()
    if (!question || streaming) return

    const history = [...messages, { role: 'user', content: question }]
    const placeholder = {
      role: 'assistant',
      content: '',
      sources: [],
      persona,
      document: activeDoc?.filename,
    }
    setMessages([...history, placeholder])
    setInput('')
    setError('')
    setStreaming(true)
    const controller = new AbortController()
    abortRef.current = controller

    try {
      await streamChat(
        {
          messages: history.map(({ role, content }) => ({ role, content })),
          doc_id: activeDoc?.doc_id ?? null,
          persona,
          tone,
        },
        {
          apiKey,
          signal: controller.signal,
          onSources: (sources) => updateLast((m) => ({ ...m, sources })),
          onToken: (text) => updateLast((m) => ({ ...m, content: m.content + text })),
        },
      )
    } catch (err) {
      if (err.name !== 'AbortError') {
        setError(err.message)
        // nothing was streamed: drop the question too, like the Streamlit app
        setMessages((current) =>
          current[current.length - 1]?.content ? current : current.slice(0, -2),
        )
      }
    } finally {
      setStreaming(false)
      abortRef.current = null
    }
  }

  function onKeyDown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      send()
    }
  }

  return (
    <div className="chat">
      <div className="chat-header">
        <p className="muted">
          {activeDoc ? (
            <>
              Answering from <strong>{activeDoc.filename}</strong> as the {persona}.
            </>
          ) : (
            <>Chatting with the {persona}. Upload a PDF to chat with your own notes.</>
          )}
        </p>
        <button onClick={() => setMessages([])} disabled={streaming || messages.length === 0}>
          Clear chat
        </button>
      </div>

      <div className="messages">
        {messages.length === 0 && (
          <div className="empty">
            Hi! I&apos;m your {persona.toLowerCase() || 'tutor'}. Ask me about anything you&apos;re
            studying.
          </div>
        )}
        {messages.map((message, i) => (
          <ChatMessage key={i} message={message} />
        ))}
        <div ref={bottomRef} />
      </div>

      {error && <p className="error">{error}</p>}

      <form className="composer" onSubmit={send}>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKeyDown}
          rows={2}
          placeholder={`Ask the ${persona.toLowerCase() || 'tutor'} anything... (Shift+Enter for a new line)`}
        />
        {streaming ? (
          <button type="button" onClick={() => abortRef.current?.abort()}>
            Stop
          </button>
        ) : (
          <button className="primary" type="submit" disabled={!input.trim()}>
            Send
          </button>
        )}
      </form>
    </div>
  )
}

function ChatMessage({ message }) {
  const isUser = message.role === 'user'
  const idk =
    !isUser && message.document && message.content.trim().toLowerCase().startsWith(IDK)

  return (
    <div className={`message ${message.role}`}>
      <div className="avatar" aria-hidden>
        {isUser ? '🧑‍🎓' : '🎓'}
      </div>
      <div className="bubble">
        {!isUser && (
          <div className="meta">
            {message.persona}
            {message.document && ` · from ${message.document}`}
          </div>
        )}
        {isUser ? (
          <p className="user-text">{message.content}</p>
        ) : (
          <ReactMarkdown>{message.content || '...'}</ReactMarkdown>
        )}
        {idk && (
          <p className="hint">
            That doesn&apos;t seem to be covered in your document. Try rephrasing, or ask about
            a related term.
          </p>
        )}
        {!idk && message.sources?.length > 0 && (
          <details className="sources">
            <summary>Sources ({message.sources.length})</summary>
            {message.sources.map((source, i) => (
              <div key={i} className="source">
                <strong>[{i + 1}]</strong> {source.page_label} ·{' '}
                {Math.round(source.similarity * 100)}% match
                <p>
                  {source.text.slice(0, 300)}
                  {source.text.length > 300 && '...'}
                </p>
              </div>
            ))}
          </details>
        )}
      </div>
    </div>
  )
}

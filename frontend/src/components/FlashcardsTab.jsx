import { useState } from 'react'
import { generateFlashcards } from '../api.js'

function shuffled(items) {
  const copy = [...items]
  for (let i = copy.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[copy[i], copy[j]] = [copy[j], copy[i]]
  }
  return copy
}

function toCsv(cards) {
  const cell = (text) => `"${text.replaceAll('"', '""')}"`
  return cards.map((c) => `${cell(c.term)},${cell(c.definition)}`).join('\n') + '\n'
}

function download(filename, text) {
  const url = URL.createObjectURL(new Blob([text], { type: 'text/csv' }))
  const link = document.createElement('a')
  link.href = url
  link.download = filename
  link.click()
  URL.revokeObjectURL(url)
}

export default function FlashcardsTab({ activeDoc, apiKey }) {
  const [topic, setTopic] = useState('')
  const [count, setCount] = useState(10)
  const [cards, setCards] = useState([])
  const [order, setOrder] = useState([])
  const [pos, setPos] = useState(0)
  const [flipped, setFlipped] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  if (!activeDoc) {
    return <p className="empty">Upload a PDF in the sidebar to turn it into flashcards.</p>
  }

  async function makeCards(event) {
    event.preventDefault()
    setLoading(true)
    setError('')
    try {
      const deck = await generateFlashcards(
        activeDoc.doc_id,
        { topic: topic || null, n: count },
        apiKey,
      )
      setCards(deck.cards)
      setOrder(deck.cards.map((_, i) => i))
      setPos(0)
      setFlipped(false)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  function go(delta) {
    setPos((p) => Math.min(Math.max(p + delta, 0), order.length - 1))
    setFlipped(false)
  }

  function onCardKey(event) {
    if (event.key === ' ' || event.key === 'Enter') {
      event.preventDefault()
      setFlipped((f) => !f)
    } else if (event.key === 'ArrowRight') go(1)
    else if (event.key === 'ArrowLeft') go(-1)
  }

  const card = cards[order[pos]]

  return (
    <div className="tool">
      <p className="muted">
        Key terms and definitions from <strong>{activeDoc.filename}</strong>.
      </p>
      <form className="tool-form" onSubmit={makeCards}>
        <label className="grow">
          Focus on a topic (optional)
          <input
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            placeholder="e.g. organelles"
          />
        </label>
        <label>
          Cards
          <select value={count} onChange={(e) => setCount(Number(e.target.value))}>
            {[5, 10, 15, 20].map((n) => (
              <option key={n}>{n}</option>
            ))}
          </select>
        </label>
        <button className="primary" disabled={loading}>
          {loading ? 'Making your flashcards...' : 'Generate flashcards'}
        </button>
      </form>
      {error && <p className="error">{error}</p>}

      {card && (
        <>
          <div
            className={flipped ? 'flip-card flipped' : 'flip-card'}
            role="button"
            tabIndex={0}
            aria-label={flipped ? 'Definition, press to show the term' : 'Term, press to flip'}
            onClick={() => setFlipped((f) => !f)}
            onKeyDown={onCardKey}
          >
            <div className="flip-inner">
              <div className="face front">
                <span className="side">Term</span>
                <span className="content">{card.term}</span>
              </div>
              <div className="face back">
                <span className="side">Definition</span>
                <span className="content">{card.definition}</span>
              </div>
            </div>
          </div>
          <p className="muted small center">
            Card {pos + 1} of {order.length} · click the card or press Space to flip, arrow keys
            to move
          </p>
          <div className="card-controls">
            <button onClick={() => go(-1)} disabled={pos === 0}>
              ◀ Prev
            </button>
            <button className="primary" onClick={() => setFlipped((f) => !f)}>
              Flip
            </button>
            <button onClick={() => go(1)} disabled={pos === order.length - 1}>
              Next ▶
            </button>
            <button
              onClick={() => {
                setOrder(shuffled(order))
                setPos(0)
                setFlipped(false)
              }}
            >
              Shuffle
            </button>
          </div>

          <details className="all-cards">
            <summary>All {cards.length} cards</summary>
            <dl>
              {cards.map((c) => (
                <div key={c.term}>
                  <dt>{c.term}</dt>
                  <dd>{c.definition}</dd>
                </div>
              ))}
            </dl>
            <button onClick={() => download('edupulse_flashcards.csv', toCsv(cards))}>
              Download for Anki / Quizlet (.csv)
            </button>
          </details>
        </>
      )}
    </div>
  )
}

import { useRef, useState } from 'react'
import { deleteDocument, uploadDocument } from '../api.js'

export default function DocumentPanel({ documents, activeDoc, apiKey, onSelect, onLibraryChange }) {
  const fileInput = useRef(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')

  async function handleFile(event) {
    const file = event.target.files?.[0]
    event.target.value = '' // so picking the same file again still fires onChange
    if (!file) return
    setBusy(true)
    setError('')
    setNotice('')
    try {
      const { document, cached } = await uploadDocument(file, apiKey)
      await onLibraryChange()
      onSelect(document)
      setNotice(
        cached
          ? `${document.filename} was indexed before, loaded it instantly.`
          : `${document.filename} is ready.`,
      )
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  async function handleDelete(doc) {
    if (!window.confirm(`Remove ${doc.filename} from the library?`)) return
    setError('')
    try {
      await deleteDocument(doc.doc_id)
      if (activeDoc?.doc_id === doc.doc_id) onSelect(null)
      await onLibraryChange()
    } catch (err) {
      setError(err.message)
    }
  }

  return (
    <section className="panel">
      <h2>Your document</h2>
      <input
        ref={fileInput}
        type="file"
        accept="application/pdf,.pdf"
        hidden
        onChange={handleFile}
      />
      <button className="primary wide" disabled={busy} onClick={() => fileInput.current.click()}>
        {busy ? 'Processing document...' : 'Upload a PDF'}
      </button>
      {busy && <p className="muted small">Reading, chunking and embedding. Big textbooks take a moment.</p>}
      {notice && <p className="notice">{notice}</p>}
      {error && <p className="error">{error}</p>}

      {documents.length > 0 && (
        <>
          <h3>Library</h3>
          <ul className="doc-list">
            {documents.map((doc) => (
              <li key={doc.doc_id} className={doc.doc_id === activeDoc?.doc_id ? 'active' : ''}>
                <button className="doc" onClick={() => onSelect(doc)}>
                  <strong>{doc.filename}</strong>
                  <span className="muted small">
                    {doc.page_count} pages · {doc.chunk_count} chunks
                  </span>
                </button>
                <button
                  className="icon"
                  title="Remove from library"
                  aria-label={`Remove ${doc.filename}`}
                  onClick={() => handleDelete(doc)}
                >
                  ✕
                </button>
              </li>
            ))}
          </ul>
        </>
      )}
    </section>
  )
}

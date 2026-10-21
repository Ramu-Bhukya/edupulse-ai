import { useState } from 'react'
import { generateQuiz } from '../api.js'

export default function QuizTab({ activeDoc, apiKey }) {
  const [topic, setTopic] = useState('')
  const [count, setCount] = useState(5)
  const [quiz, setQuiz] = useState(null)
  const [answers, setAnswers] = useState([])
  const [submitted, setSubmitted] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  if (!activeDoc) {
    return <p className="empty">Upload a PDF in the sidebar and EduPulse will quiz you on it.</p>
  }

  async function makeQuiz(event) {
    event.preventDefault()
    setLoading(true)
    setError('')
    try {
      const result = await generateQuiz(activeDoc.doc_id, { topic: topic || null, n: count }, apiKey)
      setQuiz(result)
      setAnswers(result.questions.map(() => null))
      setSubmitted(false)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  function choose(questionIndex, optionIndex) {
    setAnswers((current) => current.map((a, i) => (i === questionIndex ? optionIndex : a)))
  }

  function retry() {
    setAnswers(quiz.questions.map(() => null))
    setSubmitted(false)
  }

  const score = quiz ? quiz.questions.filter((q, i) => answers[i] === q.correct_index).length : 0
  const allAnswered = answers.length > 0 && answers.every((a) => a !== null)

  return (
    <div className="tool">
      <p className="muted">
        Multiple-choice practice questions from <strong>{activeDoc.filename}</strong>.
      </p>
      <form className="tool-form" onSubmit={makeQuiz}>
        <label className="grow">
          Focus on a topic (optional)
          <input
            value={topic}
            onChange={(e) => setTopic(e.target.value)}
            placeholder="e.g. cell membrane"
          />
        </label>
        <label>
          Questions
          <select value={count} onChange={(e) => setCount(Number(e.target.value))}>
            {[3, 4, 5, 6, 7, 8, 9, 10].map((n) => (
              <option key={n}>{n}</option>
            ))}
          </select>
        </label>
        <button className="primary" disabled={loading}>
          {loading ? 'Writing your quiz...' : quiz ? 'New quiz' : 'Generate quiz'}
        </button>
      </form>
      {error && <p className="error">{error}</p>}

      {quiz && (
        <div className="quiz">
          {quiz.questions.map((q, qi) => {
            const correct = answers[qi] === q.correct_index
            return (
              <fieldset
                key={qi}
                className={submitted ? (correct ? 'question right' : 'question wrong') : 'question'}
              >
                <legend>
                  {qi + 1}. {q.question}
                </legend>
                {q.options.map((option, oi) => (
                  <label
                    key={oi}
                    className={submitted && oi === q.correct_index ? 'option answer' : 'option'}
                  >
                    <input
                      type="radio"
                      name={`q${qi}`}
                      checked={answers[qi] === oi}
                      disabled={submitted}
                      onChange={() => choose(qi, oi)}
                    />
                    {option}
                  </label>
                ))}
                {submitted && (
                  <p className="explanation">
                    {correct ? 'Correct! ' : `Not quite. The answer is "${q.options[q.correct_index]}". `}
                    {q.explanation}
                  </p>
                )}
              </fieldset>
            )
          })}

          {submitted ? (
            <div className="score">
              <span>
                Your score: <strong>{score} / {quiz.questions.length}</strong>
                {score === quiz.questions.length && ' 🎉'}
              </span>
              <button onClick={retry}>Try again</button>
            </div>
          ) : (
            <div className="score">
              <button
                className="primary"
                disabled={!allAnswered}
                onClick={() => setSubmitted(true)}
              >
                Check answers
              </button>
              {!allAnswered && <span className="muted small">Answer every question first.</span>}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

import React, { useEffect, useRef, useState } from "react";
import "bootstrap/dist/css/bootstrap.min.css";
import ReactMarkdown from "react-markdown";
import "./App.css";
import athenaiAvatarSvg from "./assets/athenai-avatar.svg?raw";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8001";
const SESSION_STORAGE_KEY = "athenai.session.v1";
const SESSION_TTL_MS = Number(import.meta.env.VITE_SESSION_TTL_HOURS || 12) * 60 * 60 * 1000;

function logClientError(operation, error, details = {}) {
  console.error("[AthenAI]", operation, {
    ...details,
    message: error?.message,
    name: error?.name,
  });
}

const starterMessages = [
  {
    sender: "ai",
    text: "AthenAI is your intelligent learning companion for studying, comprehension, and academic success. Ask questions about your channel of videos like:",
    examples: [
      "Explain this lecture",
      "Quiz me",
      "Summarize chapter 3",
      "Find where the professor explained mitosis",
      "What are key takeaways",
    ],
  },
];

function loadStoredSession() {
  try {
    const stored = JSON.parse(window.localStorage.getItem(SESSION_STORAGE_KEY) || "null");
    if (!stored || stored.expiresAt <= Date.now()) {
      window.localStorage.removeItem(SESSION_STORAGE_KEY);
      return null;
    }
    return stored;
  } catch (error) {
    logClientError("loadStoredSession failed", error);
    window.localStorage.removeItem(SESSION_STORAGE_KEY);
    return null;
  }
}

function isStarterOnly(history) {
  return history.length === starterMessages.length && history[0]?.text === starterMessages[0].text;
}

function TokenUsage({ usage }) {
  if (!usage) return null;

  return (
    <div className="token-usage">
      {usage.label}: {usage.tokens.toLocaleString()} tokens
      {usage.total ? ` • Total: ${usage.total.toLocaleString()}` : ""}
      {usage.maxNewTokens ? ` • Limit: ${usage.maxNewTokens.toLocaleString()}` : ""}
      {usage.retriedForQuality ? " • repaired format" : ""}
      {usage.hitTokenLimit ? " • hit output limit" : ""}
      {usage.estimated ? " • estimated" : ""}
    </div>
  );
}

function ChatAvatar({ sender, thinking = false }) {
  if (sender === "ai") {
    return (
      <div className={`avatar avatar-ai ${thinking ? "is-thinking" : ""}`} aria-label="AthenAI">
        <span className="avatar-ai-art" aria-hidden="true">
          <span className="avatar-ai-svg" dangerouslySetInnerHTML={{ __html: athenaiAvatarSvg }} />
          {thinking && (
            <svg className="athenai-blink" viewBox="0 0 666.92 760.88" focusable="false">
              <ellipse className="athenai-eyelid" cx="210.55" cy="367.96" rx="71.12" ry="71.12" />
              <ellipse className="athenai-eyelid" cx="459.36" cy="367.1" rx="70.61" ry="70.61" />
            </svg>
          )}
        </span>
      </div>
    );
  }

  return (
    <div className="avatar avatar-user" aria-label="You">
      <img src="/user-avatar.png" alt="" />
    </div>
  );
}

function App() {
  const storedSession = useRef(loadStoredSession());
  const shouldRestoreSession = useRef(Boolean(storedSession.current?.sessionId));
  const [chatHistory, setChatHistory] = useState(
    storedSession.current?.chatHistory?.length ? storedSession.current.chatHistory : starterMessages,
  );
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState(storedSession.current?.sessionId || null);
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [sending, setSending] = useState(false);
  const [status, setStatus] = useState("");
  const fileInputRef = useRef(null);

  useEffect(() => {
    if (!sessionId) return;
    window.localStorage.setItem(
      SESSION_STORAGE_KEY,
      JSON.stringify({
        sessionId,
        chatHistory,
        expiresAt: Date.now() + SESSION_TTL_MS,
      }),
    );
  }, [chatHistory, sessionId]);

  useEffect(() => {
    if (!sessionId || !shouldRestoreSession.current || !isStarterOnly(chatHistory)) return;
    shouldRestoreSession.current = false;

    let cancelled = false;
    fetch(`${API_BASE_URL}/session/${encodeURIComponent(sessionId)}`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (cancelled || !data?.chat_history?.length) return;
        setChatHistory([starterMessages[0], ...data.chat_history]);
        setStatus(
          data.documents?.length
            ? `${data.documents.length} uploaded file${data.documents.length === 1 ? "" : "s"} restored for retrieval.`
            : "",
        );
      })
      .catch((error) => {
        logClientError("restoreSession failed", error, { sessionId });
      });

    return () => {
      cancelled = true;
    };
  }, [chatHistory, sessionId]);

  const handleSend = async () => {
    const prompt = input.trim();
    if (!prompt || sending) return;

    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), 240000);

    setInput("");
    setSending(true);
    setStatus("");
    setChatHistory((history) => [...history, { sender: "user", text: prompt }]);

    try {
      const res = await fetch(`${API_BASE_URL}/chat/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, session_id: sessionId }),
        signal: controller.signal,
      });

      if (!res.ok) throw new Error("Chat request failed");

      const data = await res.json();
      if (data.session_id) setSessionId(data.session_id);
      setChatHistory((history) => {
        const nextHistory = [...history];
        const usage = data.usage;
        if (usage?.prompt_tokens != null) {
          for (let index = nextHistory.length - 1; index >= 0; index -= 1) {
            if (nextHistory[index].sender === "user" && nextHistory[index].text === prompt) {
              nextHistory[index] = {
                ...nextHistory[index],
                tokenUsage: {
                  label: "Request",
                  tokens: usage.prompt_tokens,
                  estimated: usage.estimated,
                },
              };
              break;
            }
          }
        }

        nextHistory.push({
          sender: "ai",
          text: data.response || "I did not receive a response.",
          sources: data.sources || [],
          tokenUsage:
            usage?.completion_tokens != null
              ? {
                  label: "Answer",
                  tokens: usage.completion_tokens,
                  total: usage.total_tokens,
                  maxNewTokens: usage.max_new_tokens,
                  retriedForQuality: usage.retried_for_quality,
                  hitTokenLimit: usage.hit_token_limit,
                  estimated: usage.estimated,
                }
              : null,
        });

        return nextHistory;
      });
    } catch (error) {
      logClientError("chat request failed", error, {
        sessionId,
        promptChars: prompt.length,
        aborted: error.name === "AbortError",
      });
      setStatus(
        error.name === "AbortError"
          ? "The model is still busy after 2 minutes. Try a shorter question or restart the API with fewer output tokens."
          : "Could not reach the chat service. Make sure the AthenAI API is running.",
      );
    } finally {
      window.clearTimeout(timeoutId);
      setSending(false);
    }
  };

  const uploadFiles = async (selectedFiles) => {
    if (!selectedFiles.length || uploading) return;

    setUploading(true);
    setStatus("");

    const formData = new FormData();
    selectedFiles.forEach((file) => formData.append("files", file));
    if (sessionId) formData.append("session_id", sessionId);

    try {
      const res = await fetch(`${API_BASE_URL}/upload/`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) throw new Error("Upload failed");

      const data = await res.json();
      setSessionId(data.session_id);
      setFiles([]);
      const addedChunks = data.added_chunk_count ?? data.chunk_count ?? 0;
      const skippedCount = data.skipped_files?.length || 0;
      const chunkLabel = addedChunks === 1 ? "source chunk" : "source chunks";
      const skippedLabel = skippedCount
        ? ` ${skippedCount} file${skippedCount === 1 ? " was" : "s were"} skipped because no readable text was found.`
        : "";
      setStatus(`${addedChunks} ${chunkLabel} indexed for retrieval.${skippedLabel}`);
      if (fileInputRef.current) fileInputRef.current.value = "";
    } catch (error) {
      logClientError("upload request failed", error, {
        sessionId,
        fileCount: selectedFiles.length,
        filenames: selectedFiles.map((file) => file.name),
      });
      setStatus("Upload failed. Check that the API is running and accepts these files.");
    } finally {
      setUploading(false);
    }
  };

  const handleFileChange = (event) => {
    const selectedFiles = Array.from(event.target.files);
    setFiles(selectedFiles);
    uploadFiles(selectedFiles);
  };

  const handleClearChat = () => {
    if (sending) return;
    const confirmed = window.confirm("Clear this chat and start a new session?");
    if (!confirmed) return;

    window.localStorage.removeItem(SESSION_STORAGE_KEY);
    shouldRestoreSession.current = false;
    setChatHistory(starterMessages);
    setSessionId(null);
    setStatus("Chat cleared. Start a new conversation when you are ready.");
    setInput("");
    setFiles([]);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const openFilePicker = () => {
    if (!uploading) fileInputRef.current?.click();
  };

  return (
    <main className="phoenix-chat-shell">
      <section className="chat-panel">
        <section className="conversation-card">
          <header className="conversation-header">
            <div>
              <p className="eyebrow mb-1">Conversation</p>
              <h2 className="brand-heading mb-0">
                <span className="brand-athen">Athen</span>
                <span className="brand-ai">AI</span>
              </h2>
            </div>
            <input
              id="source-files"
              type="file"
              multiple
              accept=".txt,.md,.csv,.json,.dfxp,.srt,.vtt,.pdf,.docx,.pptx,.zip"
              onChange={handleFileChange}
              ref={fileInputRef}
              className="file-upload-input"
            />
            <div className="header-actions">
              <button
                type="button"
                className={`status-pill ${sessionId ? "active" : ""}`}
                onClick={openFilePicker}
                disabled={uploading}
                aria-describedby="upload-files-tooltip"
              >
                <span className="status-pill-icon" aria-hidden="true">
                  <svg viewBox="0 0 24 24" focusable="false">
                    <path d="M12 3 7 8h3v6h4V8h3l-5-5Z" />
                    <path d="M5 14v5h14v-5h-2v3H7v-3H5Z" />
                  </svg>
                </span>
                {uploading ? `Uploading ${files.length || ""}`.trim() : "Upload Additional Files"}
                <span id="upload-files-tooltip" className="upload-tooltip" role="tooltip">
                  Upload your own notes or other materials to help AthenAI answer questions. Examples:
                  text, Word, PowerPoint, PDF.
                </span>
              </button>
              <button
                type="button"
                className="clear-chat-button"
                onClick={handleClearChat}
                disabled={sending || isStarterOnly(chatHistory)}
                aria-label="Clear chat"
                aria-describedby="clear-chat-tooltip"
              >
                <span className="clear-chat-icon" aria-hidden="true">
                  <svg viewBox="0 0 24 24" focusable="false">
                    <path d="M9 3h6l1 2h4v2H4V5h4l1-2Z" />
                    <path d="M6 9h12l-1 12H7L6 9Zm4 2v8h2v-8h-2Zm4 0v8h2v-8h-2Z" />
                  </svg>
                </span>
                <span id="clear-chat-tooltip" className="upload-tooltip clear-chat-tooltip" role="tooltip">
                  Clear this chat and start a fresh session.
                </span>
              </button>
            </div>
          </header>

          <div className="message-list" aria-live="polite">
            {chatHistory.map((msg, index) => (
              <article key={`${msg.sender}-${index}`} className={`message-row ${msg.sender}`}>
                <ChatAvatar sender={msg.sender} />
                <div className="message-bubble">
                  <span className="message-name">{msg.sender === "user" ? "You" : "AthenAI"}</span>
                  <div className="message-markdown">
                    <ReactMarkdown>{msg.text}</ReactMarkdown>
                  </div>
                  <TokenUsage usage={msg.tokenUsage} />
                  {msg.examples?.length > 0 && (
                    <ul className="prompt-examples">
                      {msg.examples.map((example) => (
                        <li key={example}>"{example}"</li>
                      ))}
                    </ul>
                  )}
                  {msg.sources?.length > 0 && (
                    <div className="source-list" aria-label="Retrieved sources">
                      {msg.sources.map((source, sourceIndex) => (
                        <details key={source.id || `${source.filename}-${sourceIndex}`}>
                          <summary>
                            [{sourceIndex + 1}] {source.filename}
                          </summary>
                          <p>{source.preview}</p>
                        </details>
                      ))}
                    </div>
                  )}
                </div>
              </article>
            ))}
            {sending && (
              <article className="message-row ai">
                <ChatAvatar sender="ai" thinking />
                <div className="message-bubble typing" aria-label="Thinking">
                  Thinking
                  <span className="typing-dots" aria-hidden="true">
                    <span>.</span>
                    <span>.</span>
                    <span>.</span>
                  </span>
                </div>
              </article>
            )}
          </div>

          {status && <div className="chat-status">{status}</div>}

          <footer className="composer">
            <input
              type="text"
              className="form-control"
              placeholder="Ask about your uploaded material..."
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") handleSend();
              }}
            />
            <button className="btn btn-phoenix-send" onClick={handleSend} disabled={!input.trim() || sending}>
              {sending ? "Sending" : "Send"}
            </button>
          </footer>
        </section>
      </section>
    </main>
  );
}

export default App;

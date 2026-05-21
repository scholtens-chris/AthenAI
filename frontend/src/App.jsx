import React, { useEffect, useRef, useState } from "react";
import "bootstrap/dist/css/bootstrap.min.css";
import ReactMarkdown from "react-markdown";
import "./App.css";
import athenaiAvatarSvg from "./assets/athenai-avatar.svg?raw";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8001";
const SESSION_STORAGE_KEY = "athenai.session.v1";
const SAFETY_NOTICE_STORAGE_KEY = "athenai.safetyNotice.dismissed.v1";
const SESSION_TTL_MS = Number(import.meta.env.VITE_SESSION_TTL_HOURS || 12) * 60 * 60 * 1000;
const MEDIASITE_ID_PATTERN = "[0-9a-f]{32,34}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}";
const MEDIASITE_CHANNEL_SEGMENT_PATTERN = "[A-Za-z0-9._~-]+";
const GUID_PATTERN = new RegExp(MEDIASITE_ID_PATTERN, "gi");
const STARTER_TEXT =
  "AthenAI is your intelligent learning companion for studying, comprehension, and academic success. Ask questions about your video presentation or your channel of videos like:";
const LEGACY_STARTER_TEXTS = [
  "AthenAI is your intelligent learning companion for studying, comprehension, and academic success. Ask questions about your channel of videos like:",
  "AthenAI is your intelligent learning companion for studying, comprehension, and academic success. Ask questions about your course materials like:",
];

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
    text: STARTER_TEXT,
    examples: [
      "Explain this lecture",
      "Summarize this",
      "Quiz me",
      "Create flashcards",
      "Make me a study guide",
      "Summarize chapter 3",
      "Find where the professor explained mitosis",
      "What are key takeaways",
    ],
  },
];

export function detectMediasiteContextFromUrl(href = window.location.href) {
  let url;
  try {
    url = new URL(href, window.location.origin);
  } catch (_error) {
    return null;
  }

  const sourceUrlParamNames = ["source_url", "sourceUrl", "parent_url", "parentUrl", "page_url", "pageUrl", "mediasite_url", "mediasiteUrl"];
  for (const name of sourceUrlParamNames) {
    const value = url.searchParams.get(name);
    if (!value) continue;

    const context = detectMediasiteContextFromUrl(value);
    if (context) return { ...context, source: `${name}-query` };
  }

  const explicitMappings = [
    ["presentation", ["presentation_id", "presentationId", "presentation", "pid"]],
    ["channel", ["channel_id", "channelId", "catalog_id", "catalogId", "catalog", "cid"]],
  ];

  for (const [type, names] of explicitMappings) {
    for (const name of names) {
      const value = url.searchParams.get(name);
      const match = value?.match(GUID_PATTERN)?.[0];
      if (match) {
        return { type, id: match, source: "query" };
      }
    }
  }

  const decodedHref = decodeURIComponent(url.href);
  const decodedPath = decodeURIComponent(url.pathname);
  const channelWatchMatch = decodedPath.match(
    new RegExp(`/channel/(${MEDIASITE_CHANNEL_SEGMENT_PATTERN})/watch/(${MEDIASITE_ID_PATTERN})`, "i"),
  );
  if (channelWatchMatch) {
    return {
      type: "presentation",
      id: channelWatchMatch[2],
      channelId: channelWatchMatch[1],
      source: "channel-watch-url",
    };
  }

  const channelMatch = decodedPath.match(
    new RegExp(`/channel/(${MEDIASITE_ID_PATTERN})(?:/|$)`, "i"),
  );
  if (channelMatch) {
    return { type: "channel", id: channelMatch[1], source: "channel-url" };
  }

  const playMatch = decodedPath.match(
    new RegExp(`/play/(${MEDIASITE_ID_PATTERN})(?:/|$)`, "i"),
  );
  if (playMatch) {
    return { type: "presentation", id: playMatch[1], source: "play-url" };
  }

  const matches = [...decodedHref.matchAll(GUID_PATTERN)].map((match) => match[0]);
  if (!matches.length) return null;

  const lowerHref = decodedHref.toLowerCase();
  const type =
    /\b(channel|channels|catalog|catalogs|showcasechannel|showcasechannels|playlist|playlists)\b/.test(lowerHref)
      ? "channel"
      : "presentation";

  return { type, id: matches[matches.length - 1], source: "url" };
}

function collectEmbedCandidateUrls() {
  const candidates = [];
  const addCandidate = (href) => {
    if (href && !candidates.includes(href)) candidates.push(href);
  };

  addCandidate(window.location.href);

  try {
    if (window.parent && window.parent !== window) {
      addCandidate(window.parent.location.href);
    }
  } catch (_error) {
    // Cross-origin parent URLs are not readable from an iframe; document.referrer is the browser-safe fallback.
  }

  addCandidate(document.referrer);

  return candidates;
}

export function detectMediasiteContext() {
  for (const href of collectEmbedCandidateUrls()) {
    const context = detectMediasiteContextFromUrl(href);
    if (context) return context;
  }

  return null;
}

function contextStorageKey(context) {
  return context ? `${SESSION_STORAGE_KEY}.${context.type}.${context.id}` : SESSION_STORAGE_KEY;
}

function parseManualMediasiteContext(value) {
  const trimmed = value.trim();
  if (!trimmed) return null;

  const prefixedMatch = trimmed.match(/^(channel|presentation)\s*:\s*(.+)$/i);
  if (prefixedMatch) {
    return {
      type: prefixedMatch[1].toLowerCase(),
      id: prefixedMatch[2].trim(),
      source: "manual",
    };
  }

  const looksLikeUrlOrPath =
    /^[a-z][a-z\d+.-]*:/i.test(trimmed) || trimmed.includes("/") || trimmed.includes("?") || trimmed.includes("&");
  if (looksLikeUrlOrPath) {
    const detectedContext = detectMediasiteContextFromUrl(trimmed);
    if (detectedContext) return { ...detectedContext, source: "manual-url" };
  }

  return {
    type: "presentation",
    id: trimmed,
    source: "manual",
  };
}

async function apiErrorMessage(response, fallback) {
  try {
    const data = await response.json();
    const detail = data?.detail;
    if (typeof detail === "string") return detail;
    if (typeof detail?.message === "string") return detail.message;
    if (detail) return JSON.stringify(detail);
  } catch (_error) {
    // Fall through to the caller's generic message when the API returns no JSON body.
  }

  return fallback;
}

function loadStoredSession(storageKey = SESSION_STORAGE_KEY) {
  try {
    const stored = JSON.parse(window.localStorage.getItem(storageKey) || "null");
    if (!stored || stored.expiresAt <= Date.now()) {
      window.localStorage.removeItem(storageKey);
      return null;
    }
    return stored;
  } catch (error) {
    logClientError("loadStoredSession failed", error);
    window.localStorage.removeItem(storageKey);
    return null;
  }
}

function isStarterOnly(history) {
  return history.length === starterMessages.length && isStarterMessage(history[0]);
}

function isStarterMessage(message) {
  return message?.sender === "ai" && [STARTER_TEXT, ...LEGACY_STARTER_TEXTS].includes(message.text);
}

function normalizeChatHistory(history) {
  if (!history?.length) return starterMessages;
  if (!isStarterMessage(history[0])) return history;
  return [starterMessages[0], ...history.slice(1)];
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
  const mediasiteContext = useRef(detectMediasiteContext());
  const storageKey = useRef(contextStorageKey(mediasiteContext.current));
  const storedSession = useRef(loadStoredSession(storageKey.current));
  const shouldRestoreSession = useRef(Boolean(storedSession.current?.sessionId));
  const attemptedMediasiteImport = useRef(false);
  const [chatHistory, setChatHistory] = useState(
    normalizeChatHistory(storedSession.current?.chatHistory),
  );
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState(storedSession.current?.sessionId || null);
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [debugImporting, setDebugImporting] = useState(false);
  const [sending, setSending] = useState(false);
  const [status, setStatus] = useState("");
  const [showSafetyNotice, setShowSafetyNotice] = useState(
    () => window.localStorage.getItem(SAFETY_NOTICE_STORAGE_KEY) !== "true",
  );
  const fileInputRef = useRef(null);

  useEffect(() => {
    if (!sessionId) return;
    window.localStorage.setItem(
      storageKey.current,
      JSON.stringify({
        sessionId,
        chatHistory,
        mediasiteContext: mediasiteContext.current,
        expiresAt: Date.now() + SESSION_TTL_MS,
      }),
    );
  }, [chatHistory, sessionId]);

  const importMediasiteContext = async (context, { manual = false, signal } = {}) => {
    const endpoint = context.type === "channel" ? "import-channel" : "import-presentation";
    const body =
      context.type === "channel"
        ? { session_id: sessionId, channel_id: context.id, resource_type: "MediasiteChannels" }
        : { session_id: sessionId, presentation_id: context.id };

    setStatus(
      context.type === "channel"
        ? "Loading Mediasite channel captions, OCR, and slide details..."
        : "Loading Mediasite presentation captions, OCR, and slide details...",
    );

    try {
      const res = await fetch(`${API_BASE_URL}/mediasite/${endpoint}/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal,
      });

      if (!res.ok) throw new Error(await apiErrorMessage(res, "Mediasite import failed"));

      const data = await res.json();
      if (data.session_id) setSessionId(data.session_id);
      const importedCount = data.imported?.length;
      const indexedCount = importedCount ?? data.indexed_files?.length ?? 0;
      const chunks = data.added_chunk_count ?? data.chunk_count ?? 0;
      setStatus(
        context.type === "channel"
          ? `${indexedCount} Mediasite video${indexedCount === 1 ? "" : "s"} loaded (${chunks} source chunk${chunks === 1 ? "" : "s"}).`
          : `Mediasite presentation loaded (${chunks} source chunk${chunks === 1 ? "" : "s"}).`,
      );
    } catch (error) {
      if (error.name === "AbortError") return;
      logClientError("mediasite import failed", error, { context, manual });
      setStatus(
        manual
          ? `Debug import failed: ${error.message}`
          : "Could not automatically load Mediasite content. You can still upload files manually.",
      );
    }
  };

  useEffect(() => {
    const context = mediasiteContext.current;
    if (!context || attemptedMediasiteImport.current || storedSession.current?.sessionId) return;
    attemptedMediasiteImport.current = true;

    const controller = new AbortController();
    importMediasiteContext(context, { signal: controller.signal });

    return () => controller.abort();
  }, [sessionId]);

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

  const handleDebugImport = async () => {
    if (debugImporting) return;

    const value = window.prompt(
      "Enter a Mediasite ID or URL. Use channel:<id> for a channel, presentation:<id> for a presentation.",
    );
    if (value === null) return;

    let context = parseManualMediasiteContext(value);
    if (!context?.id) {
      setStatus("Debug import cancelled because no ID was entered.");
      return;
    }

    const hasExplicitType = /^(channel|presentation)\s*:/i.test(value.trim()) || context.source !== "manual";
    if (!hasExplicitType && window.confirm("Import this ID as a channel? Cancel imports it as a presentation.")) {
      context = { ...context, type: "channel" };
    }

    setDebugImporting(true);
    try {
      await importMediasiteContext(context, { manual: true });
    } finally {
      setDebugImporting(false);
    }
  };

  const handleClearChat = () => {
    if (sending) return;
    const confirmed = window.confirm("Clear this chat and start a new session?");
    if (!confirmed) return;

    window.localStorage.removeItem(storageKey.current);
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

  const dismissSafetyNotice = () => {
    window.localStorage.setItem(SAFETY_NOTICE_STORAGE_KEY, "true");
    setShowSafetyNotice(false);
  };

  return (
    <main className="phoenix-chat-shell">
      <section className="chat-panel">
        <section className="conversation-card">
          <header className="conversation-header">
            <div className="brand-lockup">
              <span className="header-avatar" aria-hidden="true">
                <span className="avatar-ai-art">
                  <span className="avatar-ai-svg" dangerouslySetInnerHTML={{ __html: athenaiAvatarSvg }} />
                </span>
              </span>
              <div className="brand-copy">
                <h2 className="brand-heading mb-0">
                  <span className="brand-athen">Athen</span>
                  <span className="brand-ai">AI</span>
                  <span className="beta-pill">Beta</span>
                </h2>
                <p className="eyebrow mb-1">Study Agent</p>
              </div>
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
                className="status-pill debug-pill"
                onClick={handleDebugImport}
                disabled={debugImporting}
                aria-describedby="debug-import-tooltip"
              >
                <span className="status-pill-icon" aria-hidden="true">
                  <svg viewBox="0 0 24 24" focusable="false">
                    <path d="M4 4h16v4H4V4Zm2 2v1h12V6H6Z" />
                    <path d="M4 10h16v10H4V10Zm2 2v6h12v-6H6Z" />
                    <path d="M8 14h8v2H8v-2Z" />
                  </svg>
                </span>
                {debugImporting ? "Loading Debug ID" : "DEBUG: CHANNEL/PRESENTATION ID"}
                <span id="debug-import-tooltip" className="upload-tooltip" role="tooltip">
                  Manually import a Mediasite channel or presentation ID for API testing.
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

          {showSafetyNotice && (
            <div className="safety-notice" role="note" aria-label="AthenAI beta notice">
              <div className="safety-notice-copy">
                AthenAI can make mistakes or produce incomplete results. Review all outputs before use.
              </div>
              <button type="button" className="safety-notice-dismiss" onClick={dismissSafetyNotice}>
                Dismiss
              </button>
            </div>
          )}

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

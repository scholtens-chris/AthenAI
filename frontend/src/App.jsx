import React, { useRef, useState } from "react";
import "bootstrap/dist/css/bootstrap.min.css";
import "./App.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8001";

const starterMessages = [
  {
    sender: "ai",
    text: "Upload your study files, then ask a question about the material.",
  },
];

function ChatAvatar({ sender }) {
  if (sender === "ai") {
    return (
      <div className="avatar avatar-ai" aria-label="AthenAI">
        <img src="/athenai-avatar.png" alt="" />
      </div>
    );
  }

  return <div className="avatar avatar-user">You</div>;
}

function App() {
  const [chatHistory, setChatHistory] = useState(starterMessages);
  const [input, setInput] = useState("");
  const [sessionId, setSessionId] = useState(null);
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [sending, setSending] = useState(false);
  const [status, setStatus] = useState("");
  const fileInputRef = useRef(null);

  const handleSend = async () => {
    const prompt = input.trim();
    if (!prompt || sending) return;

    const controller = new AbortController();
    const timeoutId = window.setTimeout(() => controller.abort(), 120000);

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
      setChatHistory((history) => [
        ...history,
        { sender: "ai", text: data.response || "I did not receive a response." },
      ]);
    } catch (error) {
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

  const handleFileChange = (event) => {
    setFiles(Array.from(event.target.files));
    setStatus("");
  };

  const handleUpload = async () => {
    if (!files.length || uploading) return;

    setUploading(true);
    setStatus("");

    const formData = new FormData();
    files.forEach((file) => formData.append("files", file));
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
      setStatus(`${files.length} file${files.length === 1 ? "" : "s"} ready for chat.`);
      if (fileInputRef.current) fileInputRef.current.value = "";
    } catch (error) {
      setStatus("Upload failed. Check that the API is running and accepts these files.");
    } finally {
      setUploading(false);
    }
  };

  return (
    <main className="phoenix-chat-shell">
      <section className="chat-panel">
        <aside className="chat-sidebar">
          <div>
            <p className="eyebrow mb-2">Study Chat</p>
            <h1>AthenAI</h1>
            <p className="sidebar-copy mb-0">
              Ask focused questions against uploaded class notes, transcripts, and study files.
            </p>
          </div>

          <div className="upload-box">
            <label className="form-label fw-semibold" htmlFor="study-files">
              Study files
            </label>
            <input
              id="study-files"
              type="file"
              multiple
              accept=".txt,.json,.dfxp,.srt,.vtt,.zip"
              onChange={handleFileChange}
              ref={fileInputRef}
              className="form-control"
            />
            <div className="d-flex align-items-center justify-content-between gap-3 mt-3">
              <span className="small text-body-secondary text-truncate">
                {files.length ? `${files.length} selected` : sessionId ? "Session active" : "No files selected"}
              </span>
              <button className="btn btn-phoenix-primary" onClick={handleUpload} disabled={!files.length || uploading}>
                {uploading ? "Uploading" : "Upload"}
              </button>
            </div>
          </div>
        </aside>

        <section className="conversation-card">
          <header className="conversation-header">
            <div>
              <p className="eyebrow mb-1">Conversation</p>
              <h2 className="mb-0">AthenAI</h2>
            </div>
            <span className={`status-pill ${sessionId ? "active" : ""}`}>
              {sessionId ? "Files indexed" : "Waiting for files"}
            </span>
          </header>

          <div className="message-list" aria-live="polite">
            {chatHistory.map((msg, index) => (
              <article key={`${msg.sender}-${index}`} className={`message-row ${msg.sender}`}>
                <ChatAvatar sender={msg.sender} />
                <div className="message-bubble">
                  <span className="message-name">{msg.sender === "user" ? "You" : "AthenAI"}</span>
                  <p>{msg.text}</p>
                </div>
              </article>
            ))}
            {sending && (
              <article className="message-row ai">
                <ChatAvatar sender="ai" />
                <div className="message-bubble typing">Thinking...</div>
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

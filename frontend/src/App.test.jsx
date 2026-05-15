import React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import "@testing-library/jest-dom/vitest";

import App from "./App.jsx";


function jsonResponse(body, ok = true) {
  return {
    ok,
    json: vi.fn().mockResolvedValue(body),
  };
}


describe("AthenAI app", () => {
  beforeEach(() => {
    window.localStorage.clear();
    vi.spyOn(window, "fetch");
  });

  afterEach(() => {
    cleanup();
    window.localStorage.clear();
    vi.restoreAllMocks();
    vi.useRealTimers();
  });

  it("renders the starter conversation and keeps send disabled for empty input", () => {
    render(<App />);

    expect(screen.getByRole("heading", { name: /AthenAI/i })).toBeInTheDocument();
    expect(screen.getByText(/intelligent learning companion/i)).toBeInTheDocument();
    expect(screen.getByText('"Quiz me"')).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /send/i })).toBeDisabled();
  });

  it("uploads selected files and reports indexed and skipped files", async () => {
    window.fetch.mockResolvedValueOnce(
      jsonResponse({
        session_id: "session-1",
        added_chunk_count: 3,
        chunk_count: 3,
        skipped_files: ["empty.txt"],
      }),
    );

    render(<App />);
    const input = document.querySelector("#source-files");
    const file = new File(["hello"], "notes.txt", { type: "text/plain" });

    fireEvent.change(input, { target: { files: [file] } });

    await waitFor(() => {
      expect(screen.getByText(/3 source chunks indexed for retrieval/i)).toBeInTheDocument();
    });
    expect(screen.getByText(/1 file was skipped/i)).toBeInTheDocument();
    expect(window.fetch).toHaveBeenCalledWith("http://127.0.0.1:8001/upload/", {
      method: "POST",
      body: expect.any(FormData),
    });
  });

  it("sends chat with the upload session, renders usage, markdown, and sources", async () => {
    const user = userEvent.setup();
    window.fetch
      .mockResolvedValueOnce(
        jsonResponse({
          session_id: "session-1",
          added_chunk_count: 1,
          skipped_files: [],
        }),
      )
      .mockResolvedValueOnce(
        jsonResponse({
          response: "**Mitosis** divides cells.",
          usage: {
            prompt_tokens: 12,
            completion_tokens: 7,
            total_tokens: 19,
            max_new_tokens: 128,
            retried_for_quality: true,
            hit_token_limit: true,
            estimated: false,
          },
          sources: [{ id: "s1", filename: "notes.txt", preview: "Mitosis source" }],
        }),
      );

    render(<App />);

    fireEvent.change(document.querySelector("#source-files"), {
      target: { files: [new File(["mitosis"], "notes.txt", { type: "text/plain" })] },
    });
    await screen.findByText(/1 source chunk indexed/i);

    await user.type(screen.getByPlaceholderText(/ask about your uploaded material/i), "Explain mitosis");
    await user.click(screen.getByRole("button", { name: /send/i }));

    await screen.findByText("Mitosis");
    expect(screen.getByText(/Request: 12 tokens/i)).toBeInTheDocument();
    expect(screen.getByText(/Answer: 7 tokens/i)).toBeInTheDocument();
    expect(screen.getByText(/Total: 19/i)).toBeInTheDocument();
    expect(screen.getByText(/repaired format/i)).toBeInTheDocument();
    expect(screen.getByText(/hit output limit/i)).toBeInTheDocument();

    const details = screen.getByLabelText("Retrieved sources");
    expect(within(details).getByText(/\[1\] notes.txt/i)).toBeInTheDocument();
    expect(window.fetch.mock.calls[1][1].body).toBe(
      JSON.stringify({ prompt: "Explain mitosis", session_id: "session-1" }),
    );
    expect(JSON.parse(window.localStorage.getItem("athenai.session.v1")).sessionId).toBe("session-1");
  });

  it("shows the animated thinking state while a chat request is pending", async () => {
    const user = userEvent.setup();
    window.fetch.mockImplementationOnce(() => new Promise(() => {}));

    render(<App />);

    await user.type(screen.getByPlaceholderText(/ask about your uploaded material/i), "Explain mitosis");
    await user.click(screen.getByRole("button", { name: /send/i }));

    expect(await screen.findByLabelText("Thinking")).toHaveClass("typing");
    expect(screen.getAllByLabelText("AthenAI").at(-1)).toHaveClass("is-thinking");
  });

  it("restores a recent session after the UI reloads", async () => {
    window.localStorage.setItem(
      "athenai.session.v1",
      JSON.stringify({
        sessionId: "session-1",
        chatHistory: [
          {
            sender: "ai",
            text: "AthenAI is your intelligent learning companion for studying, comprehension, and academic success. Ask questions about your channel of videos like:",
            examples: ["Quiz me"],
          },
        ],
        expiresAt: Date.now() + 60_000,
      }),
    );
    window.fetch.mockResolvedValueOnce(
      jsonResponse({
        session_id: "session-1",
        documents: [{ id: "d1", filename: "notes.txt" }],
        chat_history: [
          { sender: "user", text: "Explain mitosis" },
          { sender: "ai", text: "Mitosis divides cells." },
        ],
      }),
    );

    render(<App />);

    await screen.findByText("Mitosis divides cells.");
    expect(screen.getByText(/1 uploaded file restored/i)).toBeInTheDocument();
    expect(window.fetch).toHaveBeenCalledWith("http://127.0.0.1:8001/session/session-1");
  });

  it("drops expired browser session state", () => {
    window.localStorage.setItem(
      "athenai.session.v1",
      JSON.stringify({ sessionId: "old", chatHistory: [{ sender: "user", text: "old" }], expiresAt: Date.now() - 1 }),
    );

    render(<App />);

    expect(screen.queryByText("old")).not.toBeInTheDocument();
    expect(window.fetch).not.toHaveBeenCalled();
  });

  it("clears a persisted chat only after confirmation", async () => {
    const user = userEvent.setup();
    vi.spyOn(window, "confirm").mockReturnValue(true);
    window.localStorage.setItem(
      "athenai.session.v1",
      JSON.stringify({
        sessionId: "session-1",
        chatHistory: [
          {
            sender: "ai",
            text: "AthenAI is your intelligent learning companion for studying, comprehension, and academic success. Ask questions about your channel of videos like:",
            examples: ["Quiz me"],
          },
          { sender: "user", text: "Keep this until I clear it" },
          { sender: "ai", text: "Stored answer" },
        ],
        expiresAt: Date.now() + 60_000,
      }),
    );

    render(<App />);
    expect(screen.getByText("Stored answer")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /clear chat/i }));

    expect(window.confirm).toHaveBeenCalledWith("Clear this chat and start a new session?");
    expect(screen.queryByText("Stored answer")).not.toBeInTheDocument();
    expect(screen.getByText(/intelligent learning companion/i)).toBeInTheDocument();
    expect(screen.getByText(/Chat cleared/i)).toBeInTheDocument();
    expect(window.localStorage.getItem("athenai.session.v1")).toBeNull();
    expect(screen.getByRole("button", { name: /clear chat/i })).toBeDisabled();
  });

  it("reports failed uploads and failed chat requests", async () => {
    const user = userEvent.setup();
    window.fetch.mockResolvedValueOnce(jsonResponse({}, false)).mockResolvedValueOnce(jsonResponse({}, false));

    render(<App />);
    fireEvent.change(document.querySelector("#source-files"), {
      target: { files: [new File(["bad"], "bad.txt", { type: "text/plain" })] },
    });
    await screen.findByText(/Upload failed/i);

    await user.type(screen.getByPlaceholderText(/ask about your uploaded material/i), "Hello");
    await user.click(screen.getByRole("button", { name: /send/i }));
    await screen.findByText(/Could not reach the chat service/i);
  });

  it("shows the timeout message when a chat request is aborted", async () => {
    const user = userEvent.setup();
    window.fetch.mockRejectedValueOnce(new DOMException("aborted", "AbortError"));

    render(<App />);
    await user.type(screen.getByPlaceholderText(/ask about your uploaded material/i), "Long question");
    await user.click(screen.getByRole("button", { name: /send/i }));

    await screen.findByText(/model is still busy/i);
  });
});

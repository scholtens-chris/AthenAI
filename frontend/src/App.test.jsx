import React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import "@testing-library/jest-dom/vitest";

import App, { detectMediasiteContext, detectMediasiteContextFromUrl } from "./App.jsx";


function jsonResponse(body, ok = true) {
  return {
    ok,
    json: vi.fn().mockResolvedValue(body),
  };
}


describe("AthenAI app", () => {
  beforeEach(() => {
    window.localStorage.clear();
    window.history.pushState({}, "", "/");
    Object.defineProperty(document, "referrer", { value: "", configurable: true });
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
    expect(screen.getByText("Beta")).toBeInTheDocument();
    expect(screen.getByText(/intelligent learning companion/i)).toBeInTheDocument();
    expect(screen.getByText('"Quiz me"')).toBeInTheDocument();
    expect(screen.getByText('"Create flashcards"')).toBeInTheDocument();
    expect(screen.getByText('"Make me a study guide"')).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /send/i })).toBeDisabled();
  });

  it("shows and dismisses the beta safety notice", async () => {
    const user = userEvent.setup();

    render(<App />);

    expect(screen.getByRole("note", { name: /AthenAI beta notice/i })).toHaveTextContent(
      /AthenAI can make mistakes or produce incomplete results\. Review all outputs before use\./i,
    );

    await user.click(screen.getByRole("button", { name: /dismiss/i }));

    expect(screen.queryByRole("note", { name: /AthenAI beta notice/i })).not.toBeInTheDocument();
    expect(window.localStorage.getItem("athenai.safetyNotice.dismissed.v1")).toBe("true");
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
            text: "AthenAI is your intelligent learning companion for studying, comprehension, and academic success. Ask questions about your video presentation or your channel of videos like:",
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

  it("updates stale starter copy from browser session state", () => {
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
          { sender: "user", text: "Keep my previous question" },
        ],
        expiresAt: Date.now() + 60_000,
      }),
    );

    render(<App />);

    expect(screen.getByText(/video presentation or your channel of videos like:/i)).toBeInTheDocument();
    expect(screen.queryByText(/Ask questions about your channel of videos like:/i)).not.toBeInTheDocument();
    expect(screen.getByText("Keep my previous question")).toBeInTheDocument();
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

  it("detects Mediasite context from presentation and channel URLs", () => {
    expect(
      detectMediasiteContextFromUrl(
        "https://preview.sofodev.com/mediasite/play/31e188dbc9414c5cbd486b5a0fa7b0331d",
      ),
    ).toEqual({
      type: "presentation",
      id: "31e188dbc9414c5cbd486b5a0fa7b0331d",
      source: "play-url",
    });

    expect(
      detectMediasiteContextFromUrl(
        "https://preview.sofodev.com/mediasite/Channel/61d2ef6f70194d3183aa7fa07c2e809f5f",
      ),
    ).toEqual({
      type: "channel",
      id: "61d2ef6f70194d3183aa7fa07c2e809f5f",
      source: "channel-url",
    });

    expect(detectMediasiteContextFromUrl("https://preview.sofodev.com/mediasite/Channel/wardiere")).toBeNull();

    expect(
      detectMediasiteContextFromUrl(
        "https://preview.sofodev.com/mediasite/Channel/61d2ef6f70194d3183aa7fa07c2e809f5f/watch/31e188dbc9414c5cbd486b5a0fa7b0331d?sortBy=most-recent",
      ),
    ).toEqual({
      type: "presentation",
      id: "31e188dbc9414c5cbd486b5a0fa7b0331d",
      channelId: "61d2ef6f70194d3183aa7fa07c2e809f5f",
      source: "channel-watch-url",
    });

    expect(
      detectMediasiteContextFromUrl(
        "https://preview.sofodev.com/mediasite/Channel/wardiere/watch/31e188dbc9414c5cbd486b5a0fa7b0331d",
      ),
    ).toEqual({
      type: "presentation",
      id: "31e188dbc9414c5cbd486b5a0fa7b0331d",
      channelId: "wardiere",
      source: "channel-watch-url",
    });

    expect(
      detectMediasiteContextFromUrl(
        "https://example.edu/AthenAI?presentationId=11111111-2222-3333-4444-555555555555",
      ),
    ).toEqual({
      type: "presentation",
      id: "11111111-2222-3333-4444-555555555555",
      source: "query",
    });

    expect(
      detectMediasiteContextFromUrl(
        "https://athenai.local/?embed=1&source_url=https%3A%2F%2Fpreview.sofodev.com%2Fmediasite%2FChannel%2F61d2ef6f70194d3183aa7fa07c2e809f5f",
      ),
    ).toEqual({
      type: "channel",
      id: "61d2ef6f70194d3183aa7fa07c2e809f5f",
      source: "source_url-query",
    });
  });

  it("detects Mediasite context from an embedding page referrer", () => {
    Object.defineProperty(document, "referrer", {
      value:
        "https://preview.sofodev.com/mediasite/Channel/61d2ef6f70194d3183aa7fa07c2e809f5f/watch/31e188dbc9414c5cbd486b5a0fa7b0331d",
      configurable: true,
    });

    expect(detectMediasiteContext()).toEqual({
      type: "presentation",
      id: "31e188dbc9414c5cbd486b5a0fa7b0331d",
      channelId: "61d2ef6f70194d3183aa7fa07c2e809f5f",
      source: "channel-watch-url",
    });
  });

  it("auto-imports a Mediasite presentation from the embedding page URL", async () => {
    const presentationId = "31e188dbc9414c5cbd486b5a0fa7b0331d";
    Object.defineProperty(document, "referrer", {
      value: `https://preview.sofodev.com/mediasite/play/${presentationId}`,
      configurable: true,
    });
    window.fetch.mockResolvedValueOnce(
      jsonResponse({
        session_id: "session-player",
        indexed_files: [{ filename: "presentation.txt" }],
        added_chunk_count: 3,
      }),
    );

    render(<App />);

    await screen.findByText(/Mediasite presentation loaded/i);
    expect(window.fetch).toHaveBeenCalledWith("http://127.0.0.1:8001/mediasite/import-presentation/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: null,
        presentation_id: presentationId,
      }),
      signal: expect.any(AbortSignal),
    });
    expect(
      JSON.parse(window.localStorage.getItem(`athenai.session.v1.presentation.${presentationId}`)).sessionId,
    ).toBe("session-player");
  });

  it("auto-imports a Mediasite channel from the page URL", async () => {
    const channelId = "abcdefabcdefabcdefabcdefabcdefab";
    window.history.pushState({}, "", `/Mediasite/Channel/${channelId}`);
    window.fetch.mockResolvedValueOnce(
      jsonResponse({
        session_id: "session-channel",
        imported: [{ presentation_id: "p1" }, { presentation_id: "p2" }],
        added_chunk_count: 5,
      }),
    );

    render(<App />);

    await screen.findByText(/2 Mediasite videos loaded/i);
    expect(window.fetch).toHaveBeenCalledWith("http://127.0.0.1:8001/mediasite/import-channel/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: null,
        channel_id: channelId,
        resource_type: "MediasiteChannels",
      }),
      signal: expect.any(AbortSignal),
    });
    expect(JSON.parse(window.localStorage.getItem(`athenai.session.v1.channel.${channelId}`)).sessionId).toBe(
      "session-channel",
    );
  });

  it("manually imports a debug channel ID from the header button", async () => {
    const user = userEvent.setup();
    const channelId = "abcdefabcdefabcdefabcdefabcdefab";
    vi.spyOn(window, "prompt").mockReturnValue(channelId);
    vi.spyOn(window, "confirm").mockReturnValue(true);
    window.fetch.mockResolvedValueOnce(
      jsonResponse({
        session_id: "session-debug",
        imported: [{ presentation_id: "p1" }],
        added_chunk_count: 4,
      }),
    );

    render(<App />);

    await user.click(screen.getByRole("button", { name: /DEBUG: CHANNEL\/PRESENTATION ID/i }));

    await screen.findByText(/1 Mediasite video loaded/i);
    expect(window.confirm).toHaveBeenCalledWith("Import this ID as a channel? Cancel imports it as a presentation.");
    expect(window.fetch).toHaveBeenCalledWith("http://127.0.0.1:8001/mediasite/import-channel/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: null,
        channel_id: channelId,
        resource_type: "MediasiteChannels",
      }),
      signal: undefined,
    });
  });

  it("shows backend detail when a debug import fails", async () => {
    const user = userEvent.setup();
    vi.spyOn(window, "prompt").mockReturnValue("presentation:abcdefabcdefabcdefabcdefabcdefab");
    window.fetch.mockResolvedValueOnce(jsonResponse({ detail: "Mediasite base URL is required." }, false));

    render(<App />);

    await user.click(screen.getByRole("button", { name: /DEBUG: CHANNEL\/PRESENTATION ID/i }));

    await screen.findByText(/Debug import failed: Mediasite base URL is required/i);
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
            text: "AthenAI is your intelligent learning companion for studying, comprehension, and academic success. Ask questions about your video presentation or your channel of videos like:",
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

const $ = (s) => document.querySelector(s);
const state = {
  documents: [],
  sessions: [],
  selected: new Set(),
  session: localStorage.getItem("folio.session"),
  busy: false,
  health: null,
};
const node = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};
function notify(message) {
  $("#notification").textContent = message;
  $("#notification").hidden = !message;
}
async function api(path, options = {}) {
  const token = sessionStorage.getItem("folio.token");
  const headers = {
    ...(options.body && !(options.body instanceof FormData)
      ? { "Content-Type": "application/json" }
      : {}),
    ...options.headers,
  };
  if (token) headers.Authorization = `Bearer ${token}`;
  let response;
  try {
    response = await fetch(`/api${path}`, { ...options, headers });
  } catch {
    throw Error("Cannot reach Folio. Check that the server is running.");
  }
  const result = await response.json();
  if (!response.ok)
    throw Error(
      typeof result.detail === "string"
        ? result.detail
        : "The request could not be completed.",
    );
  return result;
}
async function refresh() {
  const [docs, sessions, health] = await Promise.all([
    api("/documents"),
    api("/sessions"),
    api("/health"),
  ]);
  state.documents = docs;
  state.sessions = sessions;
  state.health = health;
  state.selected = new Set(
    [...state.selected].filter((id) => docs.some((d) => d.id === id)),
  );
  renderDocuments();
  renderSessions();
  $("#model-label").textContent = health.llm_configured
    ? `${health.provider} · ${health.model.split("/").pop()}`
    : "Evidence preview · no LLM";
}
function renderDocuments() {
  const container = $("#documents");
  container.replaceChildren();
  $("#doc-count").textContent = state.documents.length;
  $("#composer-docs").textContent = state.documents.length
    ? `${state.documents.length} documents in your library`
    : "No documents yet";
  $("#scope-label").textContent = state.selected.size
    ? `Searching ${state.selected.size} selected document${state.selected.size === 1 ? "" : "s"}`
    : "All documents · Select files in your library to narrow the search";
  for (const doc of state.documents) {
    const row = node("div", "document"),
      box = node("input"),
      label = node("label"),
      title = node("strong", "", doc.name),
      meta = node("small", "", `${doc.pages} pages · ${doc.chunks} passages`),
      remove = node("button", "remove", "×");
    box.type = "checkbox";
    box.id = `doc-${doc.id}`;
    box.checked = state.selected.has(doc.id);
    label.htmlFor = box.id;
    box.addEventListener("change", () => {
      box.checked ? state.selected.add(doc.id) : state.selected.delete(doc.id);
      renderDocuments();
    });
    remove.setAttribute("aria-label", `Remove ${doc.name}`);
    remove.disabled = state.busy;
    remove.addEventListener("click", async () => {
      if (
        !confirm(
          `Remove ${doc.name} from the library? Existing conversation records will retain their previous excerpts.`,
        )
      )
        return;
      try {
        await api(`/documents/${doc.id}`, { method: "DELETE" });
        await refresh();
      } catch (e) {
        notify(e.message);
      }
    });
    label.append(title, meta);
    row.append(box, label, remove);
    container.append(row);
  }
}
function renderSessions() {
  const container = $("#sessions");
  container.replaceChildren();
  for (const session of state.sessions) {
    const button = node(
      "button",
      session.id === state.session ? "active" : "",
      session.title,
    );
    button.disabled = state.busy;
    button.addEventListener("click", () =>
      loadSession(session.id).catch((e) => notify(e.message)),
    );
    container.append(button);
  }
}
async function newSession() {
  const session = await api("/sessions", { method: "POST" });
  state.session = session.id;
  localStorage.setItem("folio.session", session.id);
  $("#messages").replaceChildren();
  $("#welcome").hidden = false;
  $("#conversation-title").textContent = "Document research";
  await refresh();
}
async function loadSession(id) {
  if (state.busy) return;
  const data = await api(`/sessions/${id}`);
  state.session = id;
  localStorage.setItem("folio.session", id);
  $("#messages").replaceChildren();
  $("#welcome").hidden = !!data.turns.length;
  $("#conversation-title").textContent = data.session.title;
  for (const turn of data.turns) {
    renderUser(turn.question);
    renderAnswer(turn.response);
  }
  renderSessions();
}
function renderUser(text) {
  const wrapper = node("article", "message user");
  wrapper.append(node("div", "bubble", text));
  $("#messages").append(wrapper);
  $("#welcome").hidden = true;
}
function showEvidence(sources, focus) {
  const container = $("#evidence-content");
  container.replaceChildren();
  const ordered = focus
    ? [...sources].sort((a, b) => (b.number === focus) - (a.number === focus))
    : sources;
  for (const source of ordered) {
    const section = node("section", "source-detail");
    section.append(
      node(
        "div",
        "source-meta",
        `SOURCE ${source.number} · PAGE ${source.page} · ${source.cited ? "CITED" : "RETRIEVED"}`,
      ),
      node("h3", "", source.name),
      node("p", "", source.text),
    );
    const link = node("a", "", "Open PDF at this page ↗");
    link.href = `/api/documents/${encodeURIComponent(source.document_id)}/pdf#page=${source.page}`;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.addEventListener("click", async (event) => {
      if (!sessionStorage.getItem("folio.token")) return;
      event.preventDefault();
      try {
        const response = await fetch(link.href.split("#")[0], {
          headers: {
            Authorization: `Bearer ${sessionStorage.getItem("folio.token")}`,
          },
        });
        if (!response.ok) throw Error("This PDF is no longer available.");
        const url = URL.createObjectURL(await response.blob());
        window.open(`${url}#page=${source.page}`, "_blank", "noopener");
        setTimeout(() => URL.revokeObjectURL(url), 60000);
      } catch (e) {
        notify(e.message);
      }
    });
    section.append(link);
    container.append(section);
  }
  $("#evidence-panel").hidden = false;
  document.body.classList.add("evidence-open");
}
function renderAnswer(result) {
  const article = node("article", "message assistant"),
    label = node("div", "assistant-label");
  label.append(node("span", "mini-brand", "f."), node("span", "", "Folio"));
  article.append(label);
  if (result.status !== "answered")
    article.append(
      node("div", "answer-status", result.status.replaceAll("_", " ")),
    );
  const answer = node("div", "answer");
  if (result.claims.length) {
    for (const claim of result.claims) {
      const p = node("p", "", claim.text);
      for (const source of claim.sources) {
        const b = node("button", "citation", source);
        b.setAttribute("aria-label", `Show source ${source}`);
        b.addEventListener("click", () => showEvidence(result.sources, source));
        p.append(" ", b);
      }
      answer.append(p);
    }
    if (result.note && result.status !== "answered")
      answer.append(node("div", "answer-note", result.note));
  } else answer.textContent = result.answer;
  article.append(answer);
  if (result.sources.length) {
    const chips = node("div", "source-actions"),
      cited = result.sources.filter((s) => s.cited);
    for (const source of (cited.length ? cited : result.sources).slice(0, 4)) {
      const b = node(
        "button",
        "source-chip",
        `[${source.number}] ${source.name} · p. ${source.page}`,
      );
      b.addEventListener("click", () =>
        showEvidence(result.sources, source.number),
      );
      chips.append(b);
    }
    const all = node(
      "button",
      "source-chip",
      `View ${result.sources.length} passages ↗`,
    );
    all.addEventListener("click", () => showEvidence(result.sources));
    chips.append(all);
    article.append(chips);
  }
  const trace = node("details", "trace");
  trace.append(
    node(
      "summary",
      "",
      `${(result.timings.total_ms / 1000).toFixed(1)}s · ${result.embedding_backend} · View retrieval details`,
    ),
  );
  const dl = node("dl");
  for (const [key, value] of [
    ["Search query", result.standalone_question],
    ["Model", result.model],
    ["Retrieval", result.retrieval],
    ["Rewrite", `${result.timings.rewrite_ms} ms`],
    ["Search", `${result.timings.retrieval_ms} ms`],
    ["Generation", `${result.timings.generation_ms} ms`],
  ])
    dl.append(node("dt", "", key), node("dd", "", value));
  for (const step of result.trace || []) {
    const text =
      step.stage === "assess_evidence"
        ? step.sufficient
          ? "Sufficient evidence"
          : "Searching for missing evidence"
        : step.stage === "corrective_search"
          ? step.query
          : step.stage === "grounding_check"
            ? `${step.removed_claims} unsupported claims removed`
            : step.stage === "regenerate"
              ? "One bounded revision"
              : step.stage === "decompose_query"
                ? step.queries.join(" / ")
                : `${step.passages} passages`;
    dl.append(
      node("dt", "", step.stage.replaceAll("_", " ")),
      node("dd", "", text),
    );
  }
  trace.append(dl);
  article.append(trace);
  $("#messages").append(article);
}
function busy(value) {
  state.busy = value;
  $("#send").disabled = value;
  $("#demo").disabled = value;
  $("#new-chat").disabled = value;
  $("#files").disabled = value;
  renderSessions();
  renderDocuments();
}
async function sendQuestion(question) {
  if (state.busy || !question.trim()) return;
  if (!state.documents.length) {
    notify("Add PDFs or load the sample workspace first.");
    return;
  }
  notify("");
  busy(true);
  let loading;
  try {
    if (!state.session) await newSession();
    renderUser(question);
    $("#question").value = "";
    loading = node("div", "loading", "Reading your documents");
    $("#messages").append(loading);
    loading.scrollIntoView({ behavior: "smooth", block: "center" });
    const result = await api(`/sessions/${state.session}/chat`, {
      method: "POST",
      body: JSON.stringify({
        question,
        document_ids: state.selected.size ? [...state.selected] : null,
      }),
    });
    loading.remove();
    renderAnswer(result);
    await refresh();
    $("#conversation-title").textContent =
      state.sessions.find((s) => s.id === state.session)?.title ||
      "Document research";
  } catch (e) {
    if (loading) loading.remove();
    notify(e.message);
    $("#question").value = question;
  } finally {
    busy(false);
    $("#question").focus();
    $("#chat-form").scrollIntoView({ behavior: "smooth", block: "end" });
  }
}
async function upload(files) {
  if (state.busy || !files.length) return;
  notify("");
  busy(true);
  $("#upload-status").textContent = "Reading and indexing PDFs…";
  const data = new FormData();
  for (const file of files) data.append("files", file);
  try {
    const result = await api("/documents", { method: "POST", body: data });
    await refresh();
    $("#upload-status").textContent =
      `${result.documents.length} PDF${result.documents.length === 1 ? "" : "s"} ready`;
    const warnings = result.documents.flatMap((d) =>
      (d.warnings || []).map((w) => `${d.name}: ${w}`),
    );
    if (result.errors.length || warnings.length)
      notify(
        [
          ...result.errors.map((e) => `${e.name}: ${e.error}`),
          ...warnings,
        ].join(" "),
      );
  } catch (e) {
    notify(e.message);
    $("#upload-status").textContent = "Upload failed";
  } finally {
    busy(false);
    $("#files").value = "";
  }
}
$("#chat-form").addEventListener("submit", (e) => {
  e.preventDefault();
  sendQuestion($("#question").value);
});
$("#question").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendQuestion(e.target.value);
  }
});
$("#new-chat").addEventListener("click", () =>
  newSession().catch((e) => notify(e.message)),
);
$("#files").addEventListener("change", (e) => upload(e.target.files));
$("#dropzone").addEventListener("dragover", (e) => {
  e.preventDefault();
  $("#dropzone").classList.add("dragover");
});
$("#dropzone").addEventListener("dragleave", () =>
  $("#dropzone").classList.remove("dragover"),
);
$("#dropzone").addEventListener("drop", (e) => {
  e.preventDefault();
  $("#dropzone").classList.remove("dragover");
  upload(e.dataTransfer.files);
});
$("#demo").addEventListener("click", async () => {
  busy(true);
  notify("");
  try {
    await api("/demo", { method: "POST" });
    await refresh();
    $("#upload-status").textContent = "Sample workspace ready";
  } catch (e) {
    notify(e.message);
  } finally {
    busy(false);
  }
});
$("#public-demo").addEventListener("click", async () => {
  if (state.busy) return;
  busy(true);
  notify("");
  try {
    const result = await api("/demo?corpus=public", { method: "POST" });
    if (!result.documents.length)
      throw Error(
        "Download the public reports first: python -m scripts.fetch_public_samples",
      );
    await refresh();
    $("#upload-status").textContent = "NIST reports ready";
  } catch (e) {
    notify(e.message);
  } finally {
    busy(false);
  }
});
document.querySelectorAll(".suggestion").forEach((b) =>
  b.addEventListener("click", () => {
    if (!state.documents.length) {
      notify("Click “Explore sample workspace” first to load the sample PDFs.");
      return;
    }
    sendQuestion(b.dataset.question);
  }),
);
$("#close-evidence").addEventListener("click", () => {
  $("#evidence-panel").hidden = true;
  document.body.classList.remove("evidence-open");
});
$("#settings-button").addEventListener("click", () =>
  $("#settings").showModal(),
);
$("#save-token").addEventListener("click", () => {
  sessionStorage.setItem("folio.token", $("#access-token").value);
  refresh()
    .then(() => notify(""))
    .catch((e) => notify(e.message));
});
(async () => {
  try {
    await refresh();
    if (state.session && state.sessions.some((s) => s.id === state.session))
      await loadSession(state.session);
    else state.session = null;
  } catch (e) {
    notify(e.message);
  }
})();

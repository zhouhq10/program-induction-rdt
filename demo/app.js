// app.js — UI for the melody program induction demo.  Inference runs in
// worker.js; this file handles input, runs, rendering, audio and the tutorial.

// ----- Notes: 1–6 in the model, shown as letters (low → high) -----
const NOTE_NAMES = ["C", "E", "F", "G", "A", "B"];
const NOTE_FREQS = [261.63, 329.63, 349.23, 392.0, 440.0, 493.88];
const noteName = (n) => NOTE_NAMES[n - 1] ?? "?";

const EXAMPLES = {
  "Repeats": "B A G F\nB A G F\nB A G F",
  "Up and down": "C E F G A B A G F E C",
  "Blocks": "F F F F A A A A C C C C",
};
const MODEL_DESC = {
  PCFG: "No library",
  AG: "Global library",
  HAG: "Local + global library",
};
const SWEEP_BETAS = [0.25, 0.5, 1, 2, 4];
const MODEL_COLOR = { PCFG: "var(--m-pcfg)", AG: "var(--m-ag)", HAG: "var(--m-hag)" };

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = (x, d = 2) => {
  const s = Number(x).toFixed(d);
  return s.includes(".") ? s.replace(/\.?0+$/, "") : s;
};

// ----- Audio -----
let audioCtx, playing = [];
const midiFreq = (m) => 440 * 2 ** ((m - 69) / 12);

// Play frequencies one after another (null is a rest), stopping anything still playing.
// The recorder voice is a sine with a little vibrato and a longer sustain.
function playFreqs(freqs, { recorder = false } = {}) {
  audioCtx ??= new AudioContext();
  for (const node of playing) try { node.stop(); } catch {}
  playing = [];
  const step = recorder ? 0.3 : 0.28;
  const start = audioCtx.currentTime + 0.05;
  freqs.forEach((f, i) => {
    if (f == null) return;
    const t = start + i * step, end = t + step - 0.01;
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = recorder ? "sine" : "triangle";
    osc.frequency.value = f;
    gain.gain.setValueAtTime(0, t);
    gain.gain.linearRampToValueAtTime(0.25, t + 0.02);
    if (recorder) {
      const lfo = audioCtx.createOscillator();
      const depth = audioCtx.createGain();
      lfo.frequency.value = 5.5;
      depth.gain.value = f * 0.004;
      lfo.connect(depth).connect(osc.frequency);
      lfo.start(t);
      lfo.stop(end);
      playing.push(lfo);
      gain.gain.setValueAtTime(0.22, t + step * 0.75);
    }
    gain.gain.exponentialRampToValueAtTime(0.001, end - 0.01);
    osc.connect(gain).connect(audioCtx.destination);
    osc.start(t);
    osc.stop(end);
    playing.push(osc);
  });
}
const playNotes = (notes) => playFreqs(notes.map((n) => NOTE_FREQS[n - 1]));
const playButton = (notes, label = "Play") =>
  `<button class="play" data-notes="${notes.join(" ")}" aria-label="${label}" title="${label}">▶</button>`;
document.addEventListener("click", (e) => {
  const b = e.target.closest("[data-notes]");
  if (b) playNotes(b.dataset.notes.split(" ").map(Number));
  // data-midi: MIDI pitches, "-" for a rest
  const m = e.target.closest("[data-midi]");
  if (m) playFreqs(m.dataset.midi.split(" ").map((x) => (x === "-" ? null : midiFreq(Number(x)))), { recorder: true });
});

// ----- Rendering notes and programs -----
const noteHtml = (n, cls = "") =>
  `<span class="note n${n} ${cls}" style="--p:${n}" title="${noteName(n)}">${noteName(n)}</span>`;
const notesHtml = (notes, contour = true) =>
  `<span class="notes${contour ? " contour" : ""}">${notes.map((n) => noteHtml(n)).join("")}</span>`;

// Parse a bracket term like "[K,[up,note_1_3],count_2]" into nested arrays
function parseTerm(term) {
  const tokens = term.match(/\[|\]|[^\[\],]+/g);
  let i = 0;
  const parse = () => {
    const t = tokens[i++];
    if (t !== "[") return t.trim();
    const node = [];
    while (tokens[i] !== "]") {
      node.push(parse());
      if (tokens[i] === ",") i++;
    }
    i++;
    return node;
  };
  return parse();
}

// Render a term in function form, e.g. up(C F, 2), with notes as chips
function prettyTerm(term, noteFn = noteHtml) {
  const isRouter = (x) => typeof x === "string" && /^[BCSK]+$/.test(x);
  const split = (node) => (node.length === 3 && isRouter(node[0]) ? [node[1], node[2]] : [node[0], node[1]]);
  const render = (node) => {
    if (typeof node === "string") {
      if (node.startsWith("note_")) {
        const notes = node.split("_").slice(1).map(Number);
        return `<span class="notes" style="display:inline-flex">${notes.map((n) => noteFn(n)).join("")}</span>`;
      }
      if (node.startsWith("count_")) return node.slice(6);
      return `<span class="fn">${esc(node)}</span>`;
    }
    if (node.length < 2) return esc(JSON.stringify(node));
    let [f, x] = split(node);
    const args = [x];
    while (Array.isArray(f) && f.length >= 2) {
      [f, x] = split(f);
      args.unshift(x);
    }
    if (f === "I") return render(args[0]);
    return `<span class="fn">${esc(f)}</span>(${args.map(render).join(", ")})`;
  };
  try {
    return render(parseTerm(term));
  } catch {
    return esc(term);
  }
}

function progHtml(term, noteFn) {
  return `<div class="prog">${prettyTerm(term, noteFn)}<code class="raw"${showRaw() ? "" : " hidden"}>${esc(term)}</code></div>`;
}
const showRaw = () => $("show-raw").checked;

// ----- Input parsing: letters or 1–6 -----
function parseInput(text) {
  const lines = text.split("\n").map((l) => l.trim()).filter(Boolean);
  if (!lines.length) throw new Error("Enter at least one segment.");
  return lines.map((line) => {
    const tokens = line.toUpperCase().split(/[\s,]+/).filter(Boolean);
    const notes = tokens.map((t) => (/^[1-6]$/.test(t) ? Number(t) : NOTE_NAMES.indexOf(t) + 1));
    const bad = tokens.find((t, i) => notes[i] < 1);
    if (bad) throw new Error(`"${bad}" is not a note — use C E F G A B or 1–6.`);
    if (notes.length > 40) throw new Error("Segments can have at most 40 notes.");
    return notes;
  });
}

// ----- Parameters -----
function buildParams() {
  document.querySelectorAll(".param").forEach((el) => {
    const { param, min, max, step, value, label, hint } = el.dataset;
    el.innerHTML = `<div class="param-head"><label for="p-${param}">${label}${hint ? ` <span class="hint">${hint}</span>` : ""}</label>
      <output id="o-${param}" for="p-${param}">${value}</output></div>
      <input type="range" id="p-${param}" min="${min}" max="${max}" step="${step}" value="${value}">`;
    const input = el.querySelector("input");
    input.addEventListener("input", () => { $(`o-${param}`).textContent = input.value; });
  });
}
const currentModel = () => document.querySelector('input[name="model"]:checked').value;
function currentParams() {
  const p = { model: currentModel() };
  document.querySelectorAll(".param").forEach((el) => {
    p[el.dataset.param] = Number(el.querySelector("input").value);
  });
  p.lossless = $("p-lossless").checked;
  p.selection = document.querySelector('input[name="selection"]:checked').value;
  return p;
}
// With lossless compression distortion is always 0, so β cannot change which
// programs are best — disable it to make that clear
function updateLosslessUI() {
  const on = $("p-lossless").checked;
  $("p-beta").disabled = on;
  $("sweep").disabled = on || !!current;
  $("beta-note").hidden = !on;
}

function updateModelUI() {
  const model = currentModel();
  $("model-desc").textContent = MODEL_DESC[model];
  document.querySelectorAll("[data-models]").forEach((el) => {
    el.hidden = !el.dataset.models.split(" ").includes(model);
  });
}
function paramSummary(p) {
  const parts = [p.lossless ? "lossless" : `β ${fmt(p.beta)}`];
  if (p.selection === "sample") parts.push("sampled");
  if (p.model !== "PCFG") parts.push(`α ${fmt(p.global_alpha)}`, `d ${fmt(p.global_d)}`);
  if (p.model === "HAG") parts.push(`local α ${fmt(p.local_alpha)}`, `d ${fmt(p.local_d)}`);
  return parts.join(" · ");
}

// ----- Runs -----
// Both tabs share one worker and one queue.  Explore runs are kept in `runs`;
// Bad recorder runs (kind "recorder") are kept in `recRuns`.
const runs = [];
const recRuns = [];
const queue = [];
let worker, ready = false, current = null, selected = null, libStep = null, nextId = 1;

const findRun = (id) => runs.find((r) => r.id === id) ?? recRuns.find((r) => r.id === id);
// Drop a run that was stopped or failed before finishing
function discard(run) {
  run.status = "stopped";
  const list = run.kind === "recorder" ? recRuns : runs;
  if (list.includes(run)) list.splice(list.indexOf(run), 1);
}

function setStatus(message, { busy = false, error = false } = {}) {
  for (const el of [$("status"), $("rec-status")]) {
    el.classList.toggle("error", error);
    el.innerHTML = (busy ? '<span class="spinner" aria-hidden="true"></span>' : "") + esc(message);
  }
}
function setBusy(on) {
  $("run").disabled = on;
  $("sweep").disabled = on || $("p-lossless").checked;
  $("stop").disabled = !on;
  $("rec-run").disabled = on;
  $("rec-stop").disabled = !on;
}

function startWorker() {
  ready = false;
  worker = new Worker("worker.js");
  worker.onmessage = ({ data }) => {
    const run = findRun(data.runId);
    if (data.type === "status") {
      const label = current?.kind === "recorder" ? current.label : `Run ${current?.id}`;
      if (current) setStatus(`${queue.length || current.kind === "recorder" ? `${label} · ` : ""}${data.message}`, { busy: true });
      else setStatus(data.message, { busy: true });
    } else if (data.type === "ready") {
      ready = true;
      if (!current) setStatus("Ready.");
    } else if (data.type === "result" && run) {
      run.results.push(data.result);
      if (run.kind === "recorder") renderRecorder();
      else if (selected === run) renderSelected();
    } else if (data.type === "done" && run) {
      run.status = "done";
      run.seconds = (Date.now() - run.start) / 1000;
      run.rate = run.results.reduce((s, r) => s + r.rate, 0);
      run.distortion = run.results.reduce((s, r) => s + r.distortion, 0);
      current = null;
      if (run.kind === "recorder") {
        recCacheRun(run);
        renderRecorder();
      } else {
        renderRuns();
        if (selected === run) renderSelected();
      }
      nextRun();
    } else if (data.type === "error") {
      for (const r of [run, ...queue]) if (r) discard(r);
      current = null;
      queue.length = 0;
      if (!runs.includes(selected)) selected = runs[runs.length - 1] ?? null;
      renderSelected();
      setBusy(false);
      setStatus(data.message, { error: true });
      renderRuns();
      renderRecorder();
    }
  };
}

function enqueue(paramsList) {
  let melodies;
  try {
    melodies = parseInput($("melodies").value);
  } catch (e) {
    setStatus(e.message, { error: true });
    return;
  }
  const added = paramsList.map((params) => addRun(params, melodies));
  runs.push(...added);
  selected = added[0];
  libStep = null;
  renderRuns();
  renderSelected();
  if (!current) nextRun();
}

function addRun(params, melodies, kind = "explore") {
  const run = { id: nextId++, kind, params, melodies, results: [], status: "queued" };
  queue.push(run);
  return run;
}

function nextRun() {
  const run = queue.shift();
  if (!run) {
    setBusy(false);
    setStatus("Done.");
    return;
  }
  current = run;
  run.status = "running";
  run.start = Date.now();
  setBusy(true);
  setStatus(ready ? "Starting…" : "Loading the Python runtime (first run only)…", { busy: true });
  if (run.kind === "recorder") renderRecorder();
  else renderRuns();
  worker.postMessage({
    type: "run", runId: run.id,
    text: run.melodies.map((m) => m.join(" ")).join("\n"),
    params: run.params,
  });
}

function stop() {
  worker.terminate();
  for (const r of [current, ...queue]) if (r) discard(r);
  queue.length = 0;
  current = null;
  if (!runs.includes(selected)) selected = runs[runs.length - 1] ?? null;
  setBusy(false);
  setStatus("Stopped. Restarting the Python runtime…", { busy: true });
  renderRuns();
  renderSelected();
  renderRecorder();
  startWorker();
}

// ----- Run list and rate–distortion plot -----
function renderRuns() {
  if (!runs.length) return;
  $("run-list").innerHTML = `<table>
    <thead><tr><th>#</th><th>Model</th><th>Parameters</th><th class="num">Rate</th><th class="num">Dist.</th></tr></thead>
    <tbody>${runs.map((r) => `<tr class="run-row${r === selected ? " selected" : ""}" data-run="${r.id}">
      <td>${r.id}</td>
      <td><span class="dot" style="background:${MODEL_COLOR[r.params.model]}"></span>${r.params.model}</td>
      <td>${paramSummary(r.params)}</td>
      <td class="num">${r.status === "done" ? fmt(r.rate, 1) : r.status === "running" ? "…" : "queued"}</td>
      <td class="num">${r.status === "done" ? fmt(r.distortion, 0) : ""}</td></tr>`).join("")}</tbody></table>`;
  renderPlot();
}

// Runs are only comparable on the same melody; curves connect runs that differ only in β
const melodyKey = (r) => r.melodies.map((m) => m.join(" ")).join("|");
const curveKey = (r) => {
  const { beta, ...rest } = r.params;
  return JSON.stringify(rest);
};

function renderPlot() {
  const key = selected ? melodyKey(selected) : null;
  const done = runs.filter((r) => r.status === "done" && melodyKey(r) === key);
  if (!done.length) { $("rd-plot").innerHTML = ""; return; }
  const W = 300, H = 200, L = 36, B = 30, T = 14, R = 10;
  const maxX = Math.max(...done.map((r) => r.rate)) * 1.1 || 1;
  const maxY = Math.max(...done.map((r) => r.distortion), 1) * 1.15;
  const x = (v) => L + (v / maxX) * (W - L - R);
  const y = (v) => H - B - (v / maxY) * (H - B - T);
  const ticks = (max) => [0, max / 2, max].map((v) => Math.round(v));
  const curves = {};
  done.forEach((r) => (curves[curveKey(r)] ??= []).push(r));
  const lines = Object.values(curves).map((rs) => {
    const pts = rs.slice().sort((a, b) => a.params.beta - b.params.beta).map((r) => `${x(r.rate)},${y(r.distortion)}`).join(" ");
    return rs.length > 1 ? `<polyline points="${pts}" fill="none" stroke="${MODEL_COLOR[rs[0].params.model]}" stroke-width="1" stroke-dasharray="3 3" opacity="0.6"/>` : "";
  }).join("");
  $("rd-plot").innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Rate–distortion of each run">
    <line class="axis" x1="${L}" y1="${H - B}" x2="${W - R}" y2="${H - B}"/>
    <line class="axis" x1="${L}" y1="${T}" x2="${L}" y2="${H - B}"/>
    ${ticks(maxX).map((v) => `<text x="${x(v)}" y="${H - B + 12}" text-anchor="middle">${v}</text>`).join("")}
    ${ticks(maxY).map((v) => `<text x="${L - 6}" y="${y(v) + 3}" text-anchor="end">${v}</text>`).join("")}
    <text x="${(L + W - R) / 2}" y="${H - 3}" text-anchor="middle">rate (nats)</text>
    <text x="10" y="${(T + H - B) / 2}" text-anchor="middle" transform="rotate(-90 10 ${(T + H - B) / 2})">distortion</text>
    ${lines}
    ${done.map((r) => `<circle data-run="${r.id}" class="${r === selected ? "selected" : ""}" cx="${x(r.rate)}" cy="${y(r.distortion)}" r="5" fill="${MODEL_COLOR[r.params.model]}">
      <title>Run ${r.id} · ${r.params.model} · ${paramSummary(r.params)}\nrate ${fmt(r.rate, 1)} · distortion ${fmt(r.distortion, 0)}</title></circle>`).join("")}
  </svg>`;
}

document.addEventListener("click", (e) => {
  const el = e.target.closest("[data-run]");
  if (!el) return;
  selected = runs.find((r) => r.id === Number(el.dataset.run));
  libStep = null;
  renderRuns();
  renderSelected();
});

// ----- Selected run: melodies and library -----
function renderSelected() {
  const run = selected;
  if (!run) {
    $("results").innerHTML = "";
    return;
  }
  $("results-title").textContent = `Run ${run.id} · ${run.params.model}`;
  $("results").innerHTML = run.melodies.map((melody, i) => {
    const r = run.results[i];
    if (!r) {
      return `<div class="panel"><div class="melody-head"><h2>${playButton(melody)} Segment ${i + 1}</h2>
        <span class="stats">${run.status === "queued" ? "queued" : i === run.results.length ? "compressing…" : "waiting"}</span></div>
        ${notesHtml(melody)}</div>`;
    }
    let pos = 0;
    const pieces = r.programs.map((p) => {
      const out = p.output.split(" ").map(Number);
      const target = r.melody.slice(pos, pos + out.length);
      pos += out.length;
      const wrong = out.some((n, k) => n !== target[k]);
      return `<div class="segment">
        <div style="display:flex;gap:6px;align-items:flex-start">${playButton(target, "Play these notes")}${notesHtml(target)}</div>
        ${wrong ? `<div class="notes out-row" style="margin-left:30px">${out.map((n, k) =>
          noteHtml(n, "out" + (n !== target[k] ? " wrong" : ""))).join("")}</div>` : ""}
        ${progHtml(p.program)}
        <div class="seg-meta">log p ${fmt(p.log_prob)}${p.distortion ? ` · ${p.distortion} wrong` : ""}</div>
      </div>`;
    }).join("");
    return `<div class="panel">
      <div class="melody-head"><h2>${playButton(r.melody)} Segment ${i + 1}</h2>
        <div class="stats">rate <b>${fmt(r.rate, 1)}</b> · distortion <b>${fmt(r.distortion, 0)}</b></div></div>
      <div class="segments">${pieces}</div></div>`;
  }).join("");
  renderLibrary();
}

const libTable = (lib) => !lib.length ? '<p class="empty">No programs cached yet.</p>' : `<table>
  <thead><tr><th>Program</th><th class="num">Uses</th><th class="num">log p</th></tr></thead>
  <tbody>${lib.map((p) => `<tr><td>${progHtml(p.program)}<div class="seg-meta">${esc(p.type.replace("->", " → "))}</div></td>
    <td class="num">${p.count}</td><td class="num">${fmt(p.log_prob)}</td></tr>`).join("")}</tbody></table>`;

// HAG: local library after each segment, plus the global library after the melody.
// AG: only the global library, which is updated once the melody is done.
function renderLibrary() {
  const run = selected;
  const setEmpty = (msg) => {
    $("library-title").textContent = "Library";
    $("lib-step").innerHTML = "";
    $("library").innerHTML = `<p class="empty">${msg}</p>`;
  };
  if (!run || !run.results.length) return setEmpty("Cached programs appear here.");
  if (run.params.model === "PCFG") return setEmpty("PCFG has no library.");

  const n = run.results.length;
  const global = run.results[n - 1].global_library;
  const steps = run.params.model === "HAG" ? run.results.map((_, i) => String(i + 1)) : [];
  if (global) steps.push("global");
  const step = steps.includes(libStep) ? libStep : steps[steps.length - 1];

  $("lib-step").innerHTML = steps.length ? '<span class="hint" style="font-size:12px;align-self:center">' +
    `${run.params.model === "HAG" ? "after segment" : ""}</span>` +
    steps.map((s) => `<button data-step="${s}" aria-pressed="${s === step}">${s}</button>`).join("") : "";
  if (step === "global") {
    $("library-title").textContent = "Global library";
    $("library").innerHTML = libTable(global);
  } else if (step) {
    $("library-title").textContent = "Local library";
    $("library").innerHTML = libTable(run.results[Number(step) - 1].library);
  } else {
    $("library-title").textContent = "Global library";
    $("library").innerHTML = '<p class="empty">Updated after the last segment.</p>';
  }
}
$("lib-step").addEventListener("click", (e) => {
  const b = e.target.closest("[data-step]");
  if (!b) return;
  libStep = b.dataset.step;
  renderLibrary();
});

$("show-raw").addEventListener("change", () => {
  document.querySelectorAll("#view-explore .raw").forEach((el) => { el.hidden = !showRaw(); });
});

// ----- Bad recorder: seven notes, played with their octaves -----
const REC_NAMES = ["C", "D", "E", "F", "G", "A", "B"];
const REC_SEMIS = [0, 2, 4, 5, 7, 9, 11];
// Same colour per letter as in Explore; D is the extra note
const REC_CLASS = ["n1", "n7", "n2", "n3", "n4", "n5", "n6"];
const REC_EXAMPLE = "C E G C' G E C\nC E G C' G E C\nD F A D' A F D\nC E G C' C";
let recData = null;   // recorder.json: players, fixed params, preset tunes, precomputed results
let recShow = null;   // on stage: {name, phrases, players: [{player, results, run}]}

// One phrase per line; each note has a pitch class 1–7 (what the model sees)
// and a MIDI pitch (what is played).  ' and , move a note an octave up or down.
function parseTune(text) {
  const lines = text.split("\n").map((l) => l.trim()).filter(Boolean);
  if (!lines.length) throw new Error("Enter at least one phrase.");
  return lines.map((line) => {
    const phrase = line.split(/\s+/).map((tok) => {
      if (/[#♯♭]|^[A-Ga-g]b/.test(tok)) throw new Error(`"${tok}": no sharps or flats — transpose the tune to C major.`);
      const m = tok.match(/^([A-Ga-g]|[1-7])([',]*)$/);
      if (!m) throw new Error(`"${tok}" is not a note — use C D E F G A B, with ' or , for octaves.`);
      const pc = /\d/.test(m[1]) ? Number(m[1]) : REC_NAMES.indexOf(m[1].toUpperCase()) + 1;
      const octave = [...m[2]].reduce((o, c) => o + (c === "'" ? 1 : -1), 0);
      return { pc, midi: 60 + 12 * octave + REC_SEMIS[pc - 1] };
    });
    if (phrase.length > 40) throw new Error("Phrases can have at most 40 notes.");
    return phrase;
  });
}
// Must match cache_key in recorder.py
const recCacheKey = (pcs, beta) => pcs.map((p) => p.join(" ")).join("|") + "@" + beta;

// The model only outputs pitch classes; play each one in the octave closest to
// the note it should have been, so a wrong note sounds a little off, not far off
function nearestMidi(pc, target) {
  const base = 60 + REC_SEMIS[pc - 1];
  return base + 12 * Math.round((target - base) / 12);
}
// Position on the staff (diatonic steps from middle C), for the contour
function staffStep(midi) {
  const octave = Math.floor((midi - 60) / 12);
  const semi = midi - 60 - 12 * octave;
  return 7 * octave + REC_SEMIS.findIndex((s) => s >= semi);
}

const recNote = (pc, { midi = null, cls = "", title = REC_NAMES[pc - 1] } = {}) =>
  `<span class="note ${REC_CLASS[pc - 1]} ${cls}"${midi == null ? "" : ` style="--p:${staffStep(midi)}"`}
    title="${esc(title)}">${REC_NAMES[pc - 1]}</span>`;
const recPhrase = (notes, label) =>
  `<button class="rec-phrase notes" data-midi="${notes.map((n) => n.midi).join(" ")}" title="${label}" aria-label="${label}">
    ${notes.map((n) => recNote(n.pc, n)).join("")}</button>`;
const recPlay = (phrases, label) =>
  `<button class="play" data-midi="${phrases.map((p) => p.map((n) => n.midi).join(" ")).join(" - ")}"
    aria-label="${label}" title="${label}">▶</button>`;

// What a player plays for each phrase: the programs' outputs, aligned with the tune
// piece by piece as in Explore, with wrong notes marked
function playedPhrases(phrases, results) {
  return results.map((r, i) => {
    const tune = phrases[i];
    const played = [];
    let pos = 0;
    for (const p of r.programs) {
      const out = p.output.split(" ").map(Number);
      out.forEach((pc, k) => {
        const want = tune[Math.min(pos + k, tune.length - 1)];
        const wrong = pos + k >= tune.length || pc !== want.pc;
        played.push({
          pc, midi: nearestMidi(pc, want.midi), cls: wrong ? "miss" : "",
          title: wrong ? (pos + k >= tune.length ? `${REC_NAMES[pc - 1]} (extra)` : `${REC_NAMES[pc - 1]} (should be ${REC_NAMES[want.pc - 1]})`) : REC_NAMES[pc - 1],
        });
      });
      pos += out.length;
    }
    return played;
  });
}

function renderRecorder() {
  const stage = $("rec-stage");
  if (!recShow) return;
  const { name, phrases, players } = recShow;
  const steps = phrases.flat().map((n) => staffStep(n.midi));
  const nNotes = steps.length;
  // Contour range, with room for wrong notes slightly outside the tune
  stage.style.setProperty("--top", Math.max(...steps) + 1);
  stage.style.setProperty("--bottom", Math.min(...steps) - 1);

  const stats = players.map((p) => p.results ?? (p.run?.status === "done" ? p.run.results : null));
  const maxRate = Math.max(1, ...stats.filter(Boolean).map((rs) => rs.reduce((s, r) => s + r.rate, 0)));

  const tuneCard = `<section class="panel">
    <div class="melody-head"><h2>${recPlay(phrases, `Play ${name}`)} ${esc(name)}</h2>
      <div class="stats">${phrases.length} phrases · ${nNotes} notes</div></div>
    <div class="rec-phrases">${phrases.map((p, i) => recPhrase(p, `Play phrase ${i + 1}`)).join("")}</div>
  </section>`;

  const cards = players.map((p, idx) => {
    const { player } = p;
    const results = stats[idx];
    const head = (right) => `<div class="melody-head"><h2>${results ? recPlay(playedPhrases(phrases, results), `Play ${player.name}`) : ""}
      ${esc(player.name)} <span class="hint">β ${fmt(player.beta)}</span></h2>${right}</div>
      <div class="rec-desc">${esc(player.desc)}</div>`;
    if (!results) {
      const run = p.run;
      const state = !run || run.status === "stopped" ? "Not played"
        : run.status === "queued" ? "Waiting…"
        : `Practising phrase ${Math.min(run.results.length + 1, phrases.length)} of ${phrases.length}…`;
      const busy = run && (run.status === "queued" || run.status === "running");
      return `<section class="panel rec-player">${head(`<div class="stats">${busy ? '<span class="spinner" aria-hidden="true"></span>' : ""}${state}</div>`)}</section>`;
    }
    const played = playedPhrases(phrases, results);
    const rate = results.reduce((s, r) => s + r.rate, 0);
    const wrong = results.reduce((s, r) => s + r.distortion, 0);
    return `<section class="panel rec-player">
      ${head(`<div class="stats">program <b>${fmt(rate, 1)}</b> nats · <b>${fmt(wrong, 0)}</b> wrong of ${nNotes}</div>`)}
      <div class="rec-bars" aria-hidden="true">
        <span>Program length</span><div class="bar"><div style="width:${(100 * rate) / maxRate}%"></div></div>
        <span>Wrong notes</span><div class="bar bad"><div style="width:${Math.min(100, (100 * wrong) / nNotes)}%"></div></div>
      </div>
      <div class="rec-phrases">${played.map((ph, i) => recPhrase(ph, `Play phrase ${i + 1}`)).join("")}</div>
      <details class="rec-progs"><summary>Programs</summary>
        ${results.map((r, i) => `<div class="rec-prog-row"><span class="hint">${i + 1}</span><div>${r.programs.map((pr) =>
          progHtml(pr.program, (n) => recNote(n))).join("")}</div></div>`).join("")}
      </details>
    </section>`;
  }).join("");
  stage.innerHTML = tuneCard + `<div class="rec-grid">${cards}</div>`;
  stage.querySelectorAll(".rec-progs .raw").forEach((el) => el.remove());
}

// Put a tune on stage: presets are precomputed, anything else runs in the worker
function performTune(name, text) {
  let phrases;
  try {
    phrases = parseTune(text);
  } catch (e) {
    setStatus(e.message, { error: true });
    return;
  }
  const pcs = phrases.map((p) => p.map((n) => n.pc));
  recShow = {
    name, phrases,
    players: recData.players.map((player) => ({ player, results: recData.cache[recCacheKey(pcs, player.beta)] ?? null, run: null })),
  };
  const missing = recShow.players.filter((p) => !p.results);
  for (const p of missing) {
    p.run = addRun({ ...recData.params, beta: p.player.beta }, pcs, "recorder");
    p.run.label = p.player.name;
    recRuns.push(p.run);
  }
  document.querySelectorAll("#rec-tunes .chip").forEach((b) => b.setAttribute("aria-pressed", b.textContent === name));
  renderRecorder();
  if (missing.length && !current) nextRun();
}
// Keep live results so the same tune plays back instantly next time
function recCacheRun(run) {
  recData.cache[recCacheKey(run.melodies, run.params.beta)] = run.results;
}

async function initRecorder() {
  $("rec-input").value = REC_EXAMPLE;
  try {
    recData = await (await fetch("recorder.json")).json();
  } catch {
    $("rec-stage").innerHTML = '<p class="empty">recorder.json is missing — run <code>python demo/recorder.py</code>.</p>';
    $("rec-run").disabled = true;
    return;
  }
  for (const tune of recData.tunes) {
    const b = document.createElement("button");
    b.className = "chip";
    b.textContent = tune.name;
    b.onclick = () => performTune(tune.name, tune.text);
    $("rec-tunes").appendChild(b);
  }
  $("rec-run").onclick = () => performTune("Your tune", $("rec-input").value);
  $("rec-stop").onclick = stop;
  performTune(recData.tunes[0].name, recData.tunes[0].text);
}

// ----- Tutorial -----
const PRIMITIVES = [
  ["memorize(notes)", "Store the notes exactly as they are.", "[K,memorize,note_1_3_2]", [1, 3, 2]],
  ["repeat(notes, k)", "Play the notes k times.", "[K,[repeat,note_1_3],count_3]", [1, 3, 1, 3, 1, 3]],
  ["reverse(notes)", "Play the notes forward, then backward (the turning note is shared).", "[K,reverse,note_1_2_3]", [1, 2, 3, 2, 1]],
  ["up(notes, k)", "Add k copies, each one step higher than the last.", "[K,[up,note_1_3],count_2]", [1, 3, 2, 4, 3, 5]],
  ["down(notes, k)", "Add k copies, each one step lower than the last.", "[K,[down,note_5_6],count_2]", [5, 6, 4, 5, 3, 4]],
  ["ranges(notes, s, k)", "Add k copies, each s steps higher than the last.", "[K,[[ranges,note_1],count_2],count_3]", [1, 3, 5, 1]],
  ["concatenate(a, b)", "Play a, then b.", "[K,[concatenate,note_1_2],note_6_5]", [1, 2, 6, 5]],
];
const COMPOSED = [
  ["Down from G twice, then mirror it.", "[B,reverse,[K,[down,note_4],count_2]]", [4, 3, 2, 3, 4]],
  ["Go down from E four times, then repeat that run once more, three steps higher.",
    "[CK,[C,[B,ranges,[K,[down,note_2],count_4]],count_3],count_1]", [2, 1, 6, 5, 4, 5, 4, 3, 2, 1]],
];

function renderTutorial() {
  $("tut-notes").innerHTML = NOTE_NAMES.map((name, i) =>
    `<div class="tut-note"><button class="note n${i + 1}" data-notes="${i + 1}" aria-label="Play ${name}">${name}</button>${i + 1}</div>`).join("");
  const row = (desc, term, out, name) => `<tr>
    ${name ? `<td><code>${esc(name)}</code></td>` : ""}<td>${desc}</td>
    <td>${progHtml(term)}<code class="raw">${esc(term)}</code></td>
    <td><div style="display:flex;gap:6px;align-items:center">${playButton(out)}${notesHtml(out, false)}</div></td></tr>`;
  $("prim-table").innerHTML = PRIMITIVES.map(([name, desc, term, out]) => row(desc, term, out, name)).join("");
  $("compose-table").innerHTML = COMPOSED.map(([desc, term, out]) => row(desc, term, out)).join("");
  document.querySelectorAll(".tut-example").forEach((el) => {
    const melody = el.dataset.melody.split(" ").map(Number);
    el.innerHTML = `<div style="display:flex;gap:8px;align-items:flex-start">${playButton(melody)}${notesHtml(melody)}</div>
      <div class="segments" style="margin-top:10px">${el.dataset.programs.split("|").map((t) =>
        `<div class="segment">${progHtml(t)}</div>`).join("")}</div>`;
  });
  // Raw terms are always shown in the tutorial tables
  document.querySelectorAll("#view-tutorial .prog .raw").forEach((el) => el.remove());
}

const TABS = ["explore", "tutorial", "recorder"];
function showTab(name) {
  for (const t of TABS) {
    $(`tab-${t}`).setAttribute("aria-selected", t === name);
    $(`view-${t}`).hidden = t !== name;
  }
  history.replaceState(null, "", name === "explore" ? location.pathname : `#${name}`);
}
$("tab-explore").onclick = () => showTab("explore");
$("tab-tutorial").onclick = () => showTab("tutorial");
$("tab-recorder").onclick = () => showTab("recorder");
$("try-it").onclick = () => {
  $("melodies").value = "C E F C E F C E F G A B";
  showTab("explore");
  enqueue([currentParams()]);
};

// ----- Init -----
buildParams();
for (const [name, text] of Object.entries(EXAMPLES)) {
  const b = document.createElement("button");
  b.className = "chip";
  b.textContent = name;
  b.onclick = () => { $("melodies").value = text; };
  $("examples").appendChild(b);
}
$("melodies").value = EXAMPLES["Repeats"];
document.querySelectorAll('input[name="model"]').forEach((el) => el.addEventListener("change", updateModelUI));
updateModelUI();
$("p-lossless").addEventListener("change", updateLosslessUI);
updateLosslessUI();
$("run").onclick = () => enqueue([currentParams()]);
$("sweep").onclick = () => enqueue(SWEEP_BETAS.map((beta) => ({ ...currentParams(), beta })));
$("stop").onclick = stop;
renderTutorial();
initRecorder();
if (TABS.includes(location.hash.slice(1))) showTab(location.hash.slice(1));
startWorker();

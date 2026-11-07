// Pulse demo UI. Plain JS: polls /api/state twice a second and renders what changed.
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = (v, d = 0) => (v === null || v === undefined || Number.isNaN(v) ? "–" : Number(v).toFixed(d));
// the four predictors (pulse/predictor/predictors/). INPUTS = what each one reads from a hint; the running predictor's
// own GET /info overrides it. Old run names (gbt, ngram_action) still get a label.
const PREDICTORS = {
  lightgbm: "LightGBM (next action) · main",
  markov: "Markov chain (order 2)",
  speculator: "LLM speculator",
  pulse_v3: "pulse-v3 (text + tool graph)",
};
const PRED_INPUTS = {
  lightgbm: ["trajectory", "signals"], markov: ["trajectory"],
  speculator: ["trajectory", "history"], pulse_v3: ["trajectory", "history"],
};
const LOOKAHEAD_OK = new Set(["lightgbm", "markov"]);
const OLD_NAMES = { gbt: "LightGBM (gbt)", ngram_action: "n-gram (ngram_action)", lr_action: "Logistic regression (removed)",
                    ensemble: "Trajectory ensemble (removed)" };
const predLabel = (k) => PREDICTORS[k] || OLD_NAMES[k] || k;
const inputBadges = (ins) => (ins || []).map((i) => `<span class="badge" title="this predictor reads the hint's ${esc(i)}">${esc(i)}</span>`).join(" ");
const HELDOUT = [6, 20, 22, 26, 30];
const CONFIG_COLOR = { C0: "#eb6834", C1: "#2a78d6", C3: "#1baf7a" };
const EXTRA = ["#eda100", "#e87ba4", "#7a5af5", "#008300"];

async function api(path, body) {
  const r = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.detail || r.statusText);
  return j;
}
function toast(msg, ms = 3500) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.remove("hidden");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.add("hidden"), ms);
}
function seg(id, onChange) {
  const el = document.getElementById(id);
  const get = () => $("button.on", el)?.dataset.v ?? $("button.on", el)?.dataset.mode;
  el.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b || b.disabled) return;
    $$("button", el).forEach((x) => x.classList.toggle("on", x === b));
    onChange && onChange(get());
  });
  return { get, set: (v) => $$("button", el).forEach((x) => x.classList.toggle("on", (x.dataset.v ?? x.dataset.mode) === String(v))) };
}

// ------------------------------------------------------------------ tabs
$("#tabs").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  $$("#tabs button").forEach((x) => x.classList.toggle("on", x === b));
  $$(".tab").forEach((t) => t.classList.toggle("on", t.id === "tab-" + b.dataset.tab));
  if (b.dataset.tab === "results") loadResults();
  if (b.dataset.tab === "mon") renderMonTab();
  if (b.dataset.tab === "models") loadModels();
  if (b.dataset.tab === "setup") loadSetup();
});
const showTab = (name) => $(`#tabs button[data-tab="${name}"]`).click();

// ------------------------------------------------------------------ setup (data, models, keys, clusters, cloud)
let SETUP = null;
const pill = (ok, text, title = "") => `<span class="pill ${ok === true ? "ok" : ok === false ? "bad" : "wait"}" title="${esc(title)}">${esc(text)}</span>`;
async function loadSetup() {
  try { SETUP = await api("/api/setup"); } catch (e) { toast(e.message, 6000); return; }
  const s = SETUP, d = s.data, m = s.models, env = s.env;
  $("#sProject").innerHTML =
    pill(d.trajectories === d.of, `tau-bench data ${d.trajectories}/${d.of}`, "data/tau-bench (Download data)") +
    pill(d.schemas > 0, `tool schemas ${d.schemas}`) +
    pill(m.action, "lightgbm + markov model", "models/action_model.pkl") +
    pill(m.action_retail, "retail lightgbm + markov", "models/action_model_retail.pkl (Train retail lightgbm + markov)") +
    pill(m.pulse_v3, "pulse-v3 model", "models/pulse_v3/predictor.pkl (Train pulse-v3, or the S3 registry)") +
    pill(m.embed_server, "embedding server", "llama.cpp + nomic-embed-text on :8091 (pulse-v3)") +
    pill(env.llm_ready, env.llm_ready ? "LLM keys set" : "no LLM keys", "live runs, the AI customer, the speculator");
  $("#sEnvState").textContent = Object.entries(env.values).filter(([k, v]) => v && k !== "GROQ_API_KEY").map(([k, v]) => `${k}: ${v}`)
    .concat(env.groq_keys ? [`Groq keys: ${env.groq_keys}`] : []).join(" · ") || "nothing set yet";
  $("#sGroqCount").textContent = env.groq_keys ? `(${env.groq_keys} saved; add more below)` : "(none saved)";
  $("#sTools").innerHTML = Object.entries(s.tools).map(([t, ok]) => pill(ok, t)).join("");
  $("#sKind").textContent = s.kind.exists
    ? "cluster pulse-kn exists" + (s.kind.retail_tools ? ` · ${s.kind.retail_tools} retail tools` : "") : "no cluster";
  $("#sKind").className = "pill " + (s.kind.exists ? "ok" : "");
  $("#sEks").textContent = (s.eks.account ? `AWS ${s.eks.account}` : "not logged in to AWS") +
    (s.eks.context ? " · cluster context found" : "") + (s.images.ecr ? ` · image ${s.images.ecr.split(":").pop()}` : "");
  $("#sEks").className = "pill " + (s.eks.context ? "ok" : "");
  $("#sAccount").textContent = s.eks.account ? `AWS account ${s.eks.account}` : "";
  // the model registry URIs in use (typed here, filled in from Terraform, or from the environment the UI started with)
  const reg = s.registry || {};
  for (const [id, v] of [["#sRegPulse", reg.pulse_v3], ["#sRegAction", reg.action]])
    if (document.activeElement !== $(id)) $(id).value = v || "";
  $("#sRegState").textContent = reg.pulse_v3 || reg.action ? "in use (Models tab)" : "not set: the predictors use local model files";
  $("#sClusterPred").textContent = s.cluster.knative ? `${s.cluster.context} · predictor ${s.cluster.predictor || "not deployed"}`
    : "switch the top bar to Knative";
  if (!$("#sPredictor").options.length) $("#sPredictor").innerHTML = s.predictors.map((p) => `<option>${p}</option>`).join("");
}
function setupParams(btn) {
  return { region: $("#sRegion").value.trim(), terraform: $("#sTerraform").checked, target: btn.dataset.target,
           predictor: $("#sPredictor").value, theta: Number($("#sTheta").value), github_repo: $("#sRepo").value.trim(),
           family: $("#sFamily").value, version: $("#sVersion").value.trim(), revision: $("#sRevision").value.trim() };
}
$("#tab-setup").addEventListener("click", async (e) => {
  const b = e.target.closest("button[data-act]");
  if (!b) return;
  const act = b.dataset.act, info = (SETUP && SETUP.actions[act]) || {};
  if (info.confirm && !confirm(info.confirm)) return;
  try {
    await api("/api/setup/run", { action: act, params: setupParams(b) });
    runAfter = 0;
    $("#xLog").textContent = "";
    $("#sLog").textContent = "";
    toast(`${info.title || act}: started (log below)`);
  } catch (err) { toast(err.message, 8000); }
});
$("#sRefresh").onclick = loadSetup;
$("#sStop").onclick = () => api("/api/run/stop", {});
// Groq keys: any number; each new one is saved under the next free name (GROQ_API_KEY, GROQ_API_KEY_1, ...)
$("#sGroqAdd").onclick = () => {
  const i = document.createElement("input");
  Object.assign(i, { type: "password", className: "groq-key", autocomplete: "off", placeholder: "gsk_…" });
  $("#sGroq").appendChild(i);
  i.focus();
};
$("#sEnvSave").onclick = async () => {
  const values = {};
  $$("#sEnv [data-env]").forEach((el) => { if (el.value) values[el.dataset.env] = el.value; });
  const groq_keys = $$("#sGroq .groq-key").map((el) => el.value.trim()).filter(Boolean);
  try {
    await api("/api/setup/env", { values, groq_keys });
    $$("#sEnv input[type=password]").forEach((el) => (el.value = ""));
    $$("#sGroq .groq-key").slice(1).forEach((el) => el.remove());      // back to one empty row
    toast("Saved to .env. Restart the predictor (or the UI) for new keys to take effect.", 6000);
    loadSetup();
  } catch (err) { toast(err.message, 6000); }
};
// the model registry: saved to .env and used at once (no restart)
$("#sRegSave").onclick = async () => {
  const values = {};
  $$("[data-reg]").forEach((el) => { if (el.value.trim()) values[el.dataset.reg] = el.value.trim(); });
  if (!Object.keys(values).length) return toast("Type at least one registry URI (s3://…), or use Fill in from Terraform");
  try {
    await api("/api/setup/env", { values });
    toast("Model registry saved and in use: see the Models tab", 5000);
    loadSetup();
    loadModels();
  } catch (err) { toast(err.message, 7000); }
};

// ------------------------------------------------------------------ cluster switch
const clusterSeg = seg("clusterSeg", async (mode) => {
  try {
    await api("/api/cluster", { mode });
    toast(mode === "sim" ? "Using the simulated cluster" : "Using the Knative cluster");
  } catch (err) {
    toast(err.message, 6000);
    clusterSeg.set(state.cluster?.mode || "sim");
  }
});
// airline (14 tools) or retail (16): the tools experiment runs use on the real cluster
let shownDomain = null;
const domainSeg = seg("domainSeg", async (domain) => {
  try {
    const i = await api("/api/domain", { domain });
    toast(`${domain}: ${i.detail}`, 5000);
    shownDomain = domain;
    domainForm(domain);
  } catch (err) {
    toast(err.message, 6000);
    domainSeg.set(state.cluster?.domain || "airline");
  }
});
// what the Experiments form offers for the domain: its task presets; retail has no live agent, speculator or run pod
function domainForm(domain) {
  const retail = domain === "retail";
  $$("#xTasksPreset option[data-domain]").forEach((o) => { o.hidden = o.disabled = o.dataset.domain !== domain; });
  const sel = $("#xTasksPreset");
  if (sel.selectedOptions[0]?.disabled) { sel.value = retail ? "r-heldout" : "heldout"; sel.onchange(); }
  for (const [id, off] of [["#xMode button[data-v=live]", retail], ["#xWherePod", retail]]) $(id).disabled = off;
  if (retail && xMode.get() === "live") { xMode.set("replay"); expForm(); }
  if (retail && xWhere.get() === "cluster") xWhere.set("laptop");
  const spec = $("#xPred option[value=speculator]");
  spec.disabled = retail;
  if (retail && $("#xPred").value === "speculator") { $("#xPred").value = "lightgbm"; expForm(); }
}
$("#detectBtn").onclick = async () => {
  try {
    const i = await api("/api/cluster/detect", {});
    toast(`${i.context || "no context"}: ${i.detail}`, 5000);
  } catch (e) { toast(e.message); }
};

// ------------------------------------------------------------------ predictor card
const predModel = $("#predModel");
predModel.innerHTML = Object.entries(PREDICTORS).map(([k, v]) => `<option value="${k}">${v} · reads ${PRED_INPUTS[k].join(" + ")}</option>`).join("");
$("#theta").oninput = () => ($("#thetaVal").textContent = Number($("#theta").value).toFixed(2));
const policySeg = seg("policySeg", () => policyForm());
$("#lookChk").onchange = () => policyForm();
predModel.onchange = () => policyForm();
function policyForm() {
  // lookahead needs a second model trained for the step after the customer's reply (lightgbm, markov)
  const look = LOOKAHEAD_OK.has(predModel.value);
  $("#lookChk").disabled = !look;
  if (!look) $("#lookChk").checked = false;
  $("#predInputs").innerHTML = inputBadges(PRED_INPUTS[predModel.value]);
  $$(".pol-theta").forEach((e) => e.classList.toggle("hidden", policySeg.get() !== "theta"));
  $$(".pol-topk").forEach((e) => e.classList.toggle("hidden", policySeg.get() !== "topk"));
  $$(".pol-look").forEach((e) => e.classList.toggle("hidden", !$("#lookChk").checked));
}
// what the Predictor card asks for (pulse/predictor/service.py: PREWARM_POLICY, TOP_K, P_MIN, LOOKAHEAD, THETA_NEXT)
const cardPredictor = () => ({
  model: predModel.value, theta: Number($("#theta").value),
  policy: { policy: policySeg.get(), top_k: Number($("#topK").value), p_min: Number($("#pMin").value),
            lookahead: $("#lookChk").checked, theta_next: Number($("#thetaNext").value) },
});
$("#predStart").onclick = async () => {
  try {
    await api("/api/predictor/start", cardPredictor());
    toast("Starting the predictor…");
  } catch (e) { toast(e.message, 6000); }
};
// Does the running predictor already use what the card shows? (a change only takes effect after a restart)
function predictorMatchesCard() {
  const p = state.services?.predictor, c = cardPredictor();
  if (!p || !p.running) return false;
  const a = p.policy || {}, b = c.policy;
  if (p.model !== c.model || (a.policy || "theta") !== b.policy || !!a.lookahead !== b.lookahead) return false;
  if (b.policy === "topk" && (a.top_k !== b.top_k || Math.abs(a.p_min - b.p_min) > 1e-9)) return false;
  if (b.policy === "theta" && Math.abs(p.theta - c.theta) > 1e-9) return false;
  if (b.lookahead && Math.abs(a.theta_next - b.theta_next) > 1e-9) return false;
  return true;
}
const busyNow = () => ["running", "waiting", "starting"].includes(state.session?.state) || state.run?.state === "running";
let applyTimer = null, applyPending = false;
async function applyCard() {
  applyPending = false;
  if (predictorMatchesCard()) return;
  if (busyNow()) { applyPending = true; return; }         // applied when the episode / run ends (poll loop)
  try {
    await api("/api/predictor/start", cardPredictor());
    toast("Applying the new predictor settings: the predictor restarts (about a minute)");
  } catch (e) { toast(e.message, 6000); }
}
function scheduleApply() { clearTimeout(applyTimer); applyTimer = setTimeout(applyCard, 900); }
predModel.addEventListener("change", scheduleApply);
$("#theta").addEventListener("change", scheduleApply);
["#topK", "#pMin", "#thetaNext"].forEach((id) => $(id).addEventListener("change", scheduleApply));
$("#lookChk").addEventListener("change", scheduleApply);
$("#policySeg").addEventListener("click", (e) => { if (e.target.closest("button")) scheduleApply(); });
$("#predStop").onclick = async () => { await api("/api/predictor/stop", {}); };

// ------------------------------------------------------------------ simulated cluster controls
const minScaleSeg = seg("minScaleSeg", async (v) => {
  await api("/api/sim/min_scale", { min_scale: Number(v) });
  toast(v === "1" ? "C1: every tool keeps one warm pod" : "Scale to zero: idle tools go cold after 60 s");
});
$("#simReset").onclick = async () => {
  try { await api("/api/sim/reset", {}); toast("All simulated tools back to zero pods"); } catch (e) { toast(e.message); }
};

// ------------------------------------------------------------------ demo form
let TASKS = [];
const kindSeg = seg("kindSeg", () => demoForm());
const userSeg = seg("userSeg", () => demoForm());
const predSeg = seg("predSeg");
function demoForm() {
  const kind = kindSeg.get(), replay = kind === "replay", recorded = kind === "recorded";
  $$(".live-only").forEach((e) => e.classList.toggle("hidden", kind !== "live"));
  $$(".replay-only").forEach((e) => e.classList.toggle("hidden", !replay));
  $$(".ai-only").forEach((e) => e.classList.toggle("hidden", kind !== "live" || userSeg.get() !== "ai"));
  $$(".recorded-only").forEach((e) => e.classList.toggle("hidden", !recorded));
  $$(".not-recorded").forEach((e) => e.classList.toggle("hidden", recorded));
  if (recorded) { loadRecordings(); $("#taskGoal").classList.add("hidden"); } else fillTasks();
}
// ---- recorded live runs (ui/.state/recordings): replayed with their exact messages, timing and tool calls
let RECS = [];
async function loadRecordings(select) {
  try { RECS = await api("/api/recordings"); } catch (e) { return toast(e.message); }
  const cur = select || $("#recSel").value;
  $("#recSel").innerHTML = RECS.length
    ? RECS.map((r) => `<option value="${esc(r.id)}">${esc(r.created.slice(5, 16))} · task ${esc(r.task_id ?? "free")} · ` +
        `${esc((r.agent_model || "").replace(/^.*\//, ""))} · customer ${esc(r.user)} · recorded as ${esc(r.config)} · ` +
        `${r.tool_steps} tool steps, ${r.live_calls} calls · ${secs(r.live_e2e_s)}${r.state !== "done" ? " (" + esc(r.state) + ")" : ""}</option>`).join("")
    : `<option value="">No live runs recorded yet: run a Live agent episode first</option>`;
  if (RECS.some((r) => r.id === cur)) $("#recSel").value = cur;
}
// replay: which recorded runs the "Show tasks" filter allows (fan-out = the widest group of calls fired at once)
const runAllowed = (r) => {
  const f = $("#taskFilter").value;
  return f === "fanout" ? r.fanout > 1 : f === "plain" || f === "heldout-plain" ? r.fanout === 1 : true;
};
function fillTasks() {
  const replay = kindSeg.get() === "replay", f = $("#taskFilter").value;
  const cur = $("#taskSel").value;
  const list = TASKS.filter((t) => !replay || (t.recorded && (!f.startsWith("heldout") || t.heldout) && t.runs.some(runAllowed)));
  const human = !replay && userSeg.get() === "human";
  const tag = (t) => (t.heldout ? " · held-out" : "") +
    (replay && t.recorded ? ` · fan-out in ${t.recorded - t.plain_runs} of ${t.recorded} runs` : "");
  $("#taskSel").innerHTML = (human ? `<option value="">No task: free conversation</option>` : "") +
    list.map((t) => `<option value="${t.id}">Task ${t.id}${tag(t)} — ${esc(t.instruction.slice(0, 60))}…</option>`).join("");
  if ([...$("#taskSel").options].some((o) => o.value === cur)) $("#taskSel").value = cur;
  else if (list.some((t) => t.id === 30)) $("#taskSel").value = "30";
  taskChanged();
}
function taskChanged() {
  const t = TASKS.find((x) => String(x.id) === $("#taskSel").value);
  const g = $("#taskGoal");
  if (!t) { g.classList.add("hidden"); return; }
  const who = kindSeg.get() === "live" && userSeg.get() === "human" ? "Your goal as the customer" : "Customer's goal";
  g.innerHTML = `<b>${who}:</b> ${esc(t.instruction)}<br><span class="muted small">Expected actions: ${esc(t.gold.join(" → ") || "none")}</span>`;
  g.classList.remove("hidden");
  // value = the run's index among the task's recordings (the replay's "recorded run #"), even when filtered
  const runs = (t.runs || []).map((r, i) => ({ ...r, i })).filter(runAllowed);
  $("#trialSel").innerHTML = runs.map((r) => `<option value="${r.i}">run #${r.i + 1} · ` +
      `${r.fanout > 1 ? "fan-out ×" + r.fanout : "no fan-out"} · ${r.calls} calls, ${r.tool_steps} tool steps, ` +
      `${r.turns} customer turns · ${esc(r.model || "")}</option>`).join("") ||
    `<option value="0">recorded run #1</option>`;
}
$("#taskSel").onchange = taskChanged;
$("#taskFilter").onchange = fillTasks;
$("#userDelay").oninput = () => ($("#delayVal").textContent = $("#userDelay").value + " s");

const demoCfg = (predictorOn) => kindSeg.get() === "recorded" ? {
  kind: "recorded", recording: $("#recSel").value, predictor: predictorOn,
  task_id: String((RECS.find((r) => r.id === $("#recSel").value) || {}).task_id ?? ""),
} : ({
  kind: kindSeg.get(), user: userSeg.get(), task_id: $("#taskSel").value,
  agent_model: $("#agentModel").value, user_model: $("#userModel").value, user_delay: Number($("#userDelay").value),
  trial: Number($("#trialSel").value || 0), speed: Number($("#speedSel").value), fanout: $("#fanoutChk").checked,
  predictor: predictorOn, seed: 252,
});
async function startEpisode(cfg) {
  if (cfg.kind === "live" && cfg.user === "ai" && !cfg.task_id) throw new Error("The AI customer needs a task");
  if (cfg.kind === "recorded" && !cfg.recording) throw new Error("Record a live run first (Live agent → Start episode)");
  cfg.bg_agents = state.cluster && state.cluster.mode === "knative" ? Number($("#bgAgents").value || 0) : 0;
  if (cfg.predictor && !predictorMatchesCard()) {        // C3 always runs what the Predictor card shows
    await api("/api/predictor/start", cardPredictor());
    toast("Starting the predictor with the card's settings first (about a minute)…");
  }
  await api("/api/session/start", cfg);
}
$("#startBtn").onclick = async () => {
  $("#startBtn").disabled = true;
  try { await startEpisode(demoCfg(predSeg.get() === "on")); } catch (e) { toast(e.message, 6000); }
  finally { $("#startBtn").disabled = false; }
};
$("#stopBtn").onclick = () => { race.abort = true; api("/api/session/stop", {}); };

// ------------------------------------------------------------------ C0 then C3, same episode, both from cold
const race = { running: false, abort: false };
let lastRecording = null;
async function raceRecording(id) {
  kindSeg.set("recorded");
  demoForm();
  await loadRecordings(id);
  $("#recSel").value = id;
  $("#raceCard").scrollIntoView({ behavior: "smooth", block: "start" });
  $("#raceBtn").click();
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function waitEpisodeEnd(prevId) {
  for (;;) {                                         // a new episode started, then it ended
    await sleep(500);
    if (race.abort) throw new Error("stopped");
    const s = state.session;
    if (s && s.id !== prevId && ["done", "failed", "stopped"].includes(s.state)) return s;
  }
}
async function prepareCold(label) {
  const r = await api("/api/demo/prepare", {});
  if (state.cluster?.mode !== "knative") return;
  const t0 = Date.now();                               // real cluster: wait until every tool has scaled to zero
  for (;;) {
    const left = (state.pods || []).reduce((n, p) => n + p.ready + p.starting, 0);
    if (left === 0) return;
    if (Date.now() - t0 > 6 * 60 * 1000) throw new Error("tools did not scale to zero within 6 min");
    $("#raceStatus").textContent = `${label}: waiting for every tool to go cold (${left} pod${left > 1 ? "s" : ""} left, ` +
      `${Math.round((Date.now() - t0) / 1000)} s; idle window 60 s + scale-down-delay) · database ${r.db}`;
    if (race.abort) throw new Error("stopped");
    await sleep(1000);
  }
}
$("#raceBtn").onclick = async () => {
  if (race.running) return;
  if (kindSeg.get() === "live")
    return toast("For a fair C0 vs C3, race a replay: record this live run first, then 'Replay a live run' (or use the button at the end of the episode)", 7000);
  race.running = true; race.abort = false;
  $("#raceBtn").disabled = true;
  try {
    for (const [on, label] of [[false, "C0 (no prewarming)"], [true, "C3 (predictor)"]]) {
      $("#raceStatus").textContent = `${label}: preparing a cold start…`;
      await prepareCold(label);
      $("#raceStatus").textContent = `${label}: running…`;
      const prev = state.session?.id;
      await startEpisode(demoCfg(on));
      const s = await waitEpisodeEnd(prev);
      if (s.state !== "done") throw new Error(`${label} ended: ${s.state}`);
    }
    $("#raceStatus").textContent = "Done: both runs below, and in Episodes so far.";
  } catch (e) {
    $("#raceStatus").textContent = "C0 vs C3: " + e.message;
  } finally {
    race.running = false;
    $("#raceBtn").disabled = false;
  }
};
$("#replyForm").onsubmit = async (e) => {
  e.preventDefault();
  const text = $("#replyText").value;
  $("#replyText").value = "";
  $("#replyForm").classList.add("hidden");
  try { await api("/api/session/reply", { text }); } catch (err) { toast(err.message); }
};

// ------------------------------------------------------------------ timeline
let sessionAfter = 0;
let stepEls = {};
let thinkingEl = null;
function resetTimeline() {
  sessionAfter = 0;
  stepEls = {};
  thinkingEl = null;
  $("#timeline").innerHTML = "";
}
function add(html, cls = "") {
  const tl = $("#timeline");
  const atBottom = tl.scrollHeight - tl.scrollTop - tl.clientHeight < 80;
  const d = document.createElement("div");
  d.className = cls;
  d.innerHTML = html;
  tl.appendChild(d);
  if (atBottom) tl.scrollTop = tl.scrollHeight;
  return d;
}
function stepEl(step) {
  if (!stepEls[step]) {
    stepEls[step] = add(`<div class="step-head"><span>LLM step ${step}</span><span class="sh"></span></div>`, "step");
  }
  return stepEls[step];
}
function clearThinking() { if (thinkingEl) { thinkingEl.remove(); thinkingEl = null; } }
function renderEvent(ev) {
  switch (ev.type) {
    case "info": add(esc(ev.text), "sys"); break;
    case "error": clearThinking(); add("✗ " + esc(ev.text), "sys err"); break;
    case "goal": break;
    case "customer":
      clearThinking();
      add(`<div class="who">Customer · ${ev.t}s</div>${esc(ev.text)}`, "msg customer"); break;
    case "hint": {
      const p = document.createElement("div");
      p.className = "pred";
      p.dataset.step = ev.step;
      p.innerHTML = `<span class="lbl">Predictor${ev.kind === "reply" ? " (reply step)" : ""}</span><span class="muted">asking…</span>`;
      stepEl(ev.step).appendChild(p);
      break;
    }
    case "agent": {
      clearThinking();
      add(`<div class="who">Agent${ev.think_s != null ? ` · thought ${ev.think_s}s` : ""} · ${ev.t}s</div>${esc(ev.text)}`, "msg agent");
      break;
    }
    case "typing":
      clearThinking();
      thinkingEl = add("customer is typing", "thinking cust");
      break;
    case "thinking":
      clearThinking();
      thinkingEl = add(`agent thinking (step ${ev.step}${ev.think_s ? `, ${ev.think_s}s` : ""})`, "thinking");
      break;
    case "prediction": {
      const el = stepEl(ev.step);
      let p = el.querySelector(".pred:not(.done)");
      if (!p) { p = document.createElement("div"); p.className = "pred"; el.insertBefore(p, el.querySelector(".call")); }
      p.classList.add("done");
      if (ev.error) {
        p.innerHTML = `<span class="lbl">Predictor</span><span class="muted">no answer: ${esc(ev.error)}</span>`;
      } else {
        const warm = new Set(ev.prewarm.map((x) => x.tool));
        p.innerHTML = `<span class="lbl">Predictor${ev.kind === "reply" ? " (reply step)" : ""}</span>` +
          ev.top.map((x) => `<span class="chip ${warm.has(x.tool) ? "warm" : ""}">${esc(x.tool)} <span class="p">${x.p.toFixed(2)}</span></span>`).join("") +
          `<span class="muted">→ ${ev.prewarm.length ? "prewarm " + ev.prewarm.map((x) => esc(x.tool) + (x.k > 1 ? " ×" + x.k : "")).join(", ") : "nothing to prewarm"} · ${ev.rtt_ms} ms</span>`;
      }
      break;
    }
    case "agent_calls": {
      clearThinking();
      const el = stepEl(ev.step);
      $(".sh", el).textContent = `agent called ${ev.tools.length > 1 ? ev.tools.length + " tools at once" : ev.tools[0] || "?"}` +
        (ev.think_s != null ? ` after thinking ${ev.think_s}s` : "");
      break;
    }
    case "call": {
      const el = stepEl(ev.step);
      const c = document.createElement("div");
      c.className = "call" + (ev.cold ? " cold" : "");
      c.innerHTML = `<span class="name">${esc(ev.tool)}</span>
        <span class="meta">${ev.parallel > 1 ? `<span class="badge par">∥ ${ev.parallel}</span>` : ""}
          ${ev.prewarmed ? `<span class="badge hit">prewarmed ✓</span>` : ""}
          ${ev.wait_ms > 0 ? `<span class="muted small" title="time this call waited for a pod to start">waited ${ev.wait_ms} ms${ev.prewarmed ? " (prewarm landed late)" : ""}</span>` : ""}
          <span class="badge ${ev.cold ? "cold" : "warm"}">${ev.cold ? "cold" : "warm"}</span>
          <b>${fmt(ev.latency_ms)} ms</b></span>
        <span class="args">${esc(JSON.stringify(ev.args))}</span>
        <span class="res">↳ ${esc(ev.result)}</span>`;
      el.appendChild(c);
      break;
    }
    case "waiting_for_user":
      clearThinking();
      add("Your turn: reply as the customer below", "sys");
      $("#replyForm").classList.remove("hidden");
      $("#replyText").focus();
      break;
    case "recorded":
      lastRecording = ev.recording;
      break;
    case "end": {
      clearThinking();
      $("#replyForm").classList.add("hidden");
      const s = ev.summary;
      if (s.kind === "live" && s.recording) {         // offer the fair comparison of exactly this conversation
        const b = add(`<button class="primary">↻ Replay this exact run: C0 vs C3</button>
          <span class="muted small">same customer messages, think and reply times and tool calls; only prewarming differs</span>`, "replay-offer");
        b.querySelector("button").onclick = () => raceRecording(s.recording);
      }
      add(`<div class="stats">${stat(s.calls, "tool calls")}${stat(s.cold, "cold calls")}${stat(s.prewarmed_hits, "prewarmed")}` +
        `${stat(fmt(s.mean_ms), "mean latency ms")}${stat(fmt(s.cold_wait_s, 1), "s lost to cold starts")}` +
        `${s.pod_seconds != null ? stat(fmt(s.pod_seconds), "pod-seconds") : ""}${stat(s.hints_answered + "/" + s.hints_sent, "hints answered")}</div>`, "step");
      break;
    }
  }
}
const stat = (v, k) => `<div class="stat"><div class="v">${v ?? "–"}</div><div class="k">${k}</div></div>`;

// ------------------------------------------------------------------ experiments tab
const xMode = seg("xMode", () => expForm());
const xConfig = seg("xConfig", () => expForm());
const xWhere = seg("xWhere");
$("#xPred").innerHTML = Object.entries(PREDICTORS).map(([k, v]) => `<option value="${k}">${v}</option>`).join("");
$("#xTheta").oninput = () => ($("#xThetaVal").textContent = Number($("#xTheta").value).toFixed(2));
$("#xTasksPreset").onchange = () => {
  const v = $("#xTasksPreset").value;
  $("#xTasksCustomWrap").classList.toggle("hidden", v !== "custom");
  if (v === "heldout-seq") $("#xFanout").checked = false;      // the replay fires every call on its own
  else if (v === "heldout" || v === "fanout") $("#xFanout").checked = true;
};
function expForm() {
  const mode = xMode.get(), live = mode === "live", load = mode === "load";
  const target = $("#lTarget").value, engine = $("#lEngine").value;
  if (load && target === "episodes" && engine === "asyncio") $("#lEngine").value = "locust";   // a Locust-only test
  $$(".replay-x").forEach((e) => e.classList.toggle("hidden", mode !== "replay"));
  $$(".live-x").forEach((e) => e.classList.toggle("hidden", !live));
  $$(".not-load").forEach((e) => e.classList.toggle("hidden", load));
  $$(".load-x").forEach((e) => e.classList.toggle("hidden", !load));
  $$(".l-locust").forEach((e) => e.classList.toggle("hidden", !load || $("#lEngine").value !== "locust"));
  $$(".l-async-pred").forEach((e) => e.classList.toggle("hidden", !load || $("#lEngine").value !== "asyncio" || target !== "predictor"));
  $$(".l-async-tools").forEach((e) => e.classList.toggle("hidden", !load || $("#lEngine").value !== "asyncio" || target !== "tools"));
  $("#xStart").textContent = load ? "▶ Start load test" : "▶ Start run";
  // the predictor test always runs a predictor (C0/C1 then only label the cluster state)
  const c3 = xConfig.get() === "C3" || (load && target === "predictor");
  $$(".c3-only").forEach((e) => e.classList.toggle("hidden", !c3));
  $$(".x-theta").forEach((e) => e.classList.toggle("hidden", !c3 || $("#xPolicy").value !== "theta"));
  $$(".x-topk").forEach((e) => e.classList.toggle("hidden", !c3 || $("#xPolicy").value !== "topk"));
  const lookOk = LOOKAHEAD_OK.has($("#xPred").value);
  $("#xLook").disabled = !lookOk;
  if (!lookOk) $("#xLook").checked = false;
  $$(".x-look").forEach((e) => e.classList.toggle("hidden", !c3 || !$("#xLook").checked));
}
$("#xPred").addEventListener("change", () => expForm());
$("#xPolicy").onchange = () => expForm();
$("#xLook").onchange = () => expForm();
$("#lTarget").onchange = $("#lEngine").onchange = () => expForm();
function expTasks() {
  const v = $("#xTasksPreset").value;
  if (v === "heldout" || v === "heldout-seq") return HELDOUT;
  if (v === "r-heldout") return state.cluster?.heldout || null;     // null: no retail model trained yet
  if (v === "fanout") return [2, 3, 28, 30, 31, 33, 34];
  if (v === "30") return [30];
  if (v === "all") return [];
  return $("#xTasksCustom").value.split(/[\s,]+/).filter(Boolean).map(Number);
}
let runAfter = 0;
$("#xStart").onclick = async () => {
  const live = xMode.get() === "live";
  const spec = {
    mode: xMode.get(), config: xConfig.get(), predictor: $("#xPred").value, theta: Number($("#xTheta").value),
    where: xWhere.get(), tasks: expTasks(), label: $("#xLabel").value.trim() || "ui",
    episodes: live ? null : ($("#xEpisodes").value || null), seed: live ? null : $("#xSeed").value,
    user_turn_median: live ? null : $("#xUserTurn").value, fanout: $("#xFanout").checked,
    agent_model: $("#xAgent").value, repeats: live ? $("#xRepeats").value : null,
    user_delay_median: live ? $("#xUserDelay").value : null,
    policy: { policy: $("#xPolicy").value, top_k: Number($("#xTopK").value), p_min: Number($("#xPMin").value),
              lookahead: $("#xLook").checked, theta_next: Number($("#xThetaNext").value) },
    // load tests (loadtest/)
    target: $("#lTarget").value, engine: $("#lEngine").value, stages: $("#lStages").value.trim(),
    interactive: $("#lWeb").checked, rates: $("#lRates").value, duration: Number($("#lDuration").value),
    burst: $("#lBurst").value, pause: Number($("#lPause").value),
  };
  if (live && !spec.tasks.length) return toast("Pick tasks for a live run");
  if (spec.tasks === null || (spec.tasks.length === 0 && $("#xTasksPreset").value === "r-heldout"))
    return toast("No held-out retail tasks yet: Setup -> Train retail lightgbm + markov writes them (or pick All tasks)", 7000);
  try {
    const r = await api("/api/run/start", spec);
    runAfter = 0;
    $("#xLog").textContent = "";
    $("#xSummaryCard").classList.add("hidden");
    toast("Run started: " + r.steps.filter((s) => !s.startsWith("__")).join(" → "), 5000);
  } catch (e) { toast(e.message, 7000); }
};
$("#xStop").onclick = () => api("/api/run/stop", {});

function renderRun(run) {
  if (!run) return;
  const gate = run.lines.filter((l) => l.startsWith("GATE:")).pop();     // retraining: the release gate's verdict
  if (gate) { $("#gVerdict").textContent = gate; loadModels(); }
  if (run.spec && run.spec.mode === "retrain" && run.state === "failed") $("#gVerdict").textContent = "training failed: see Run log";
  if (run.lines.length) {
    for (const log of [$("#xLog"), $("#sLog")]) {          // the Experiments tab's Run log and the Setup tab's log
      log.textContent += (log.textContent && !log.textContent.endsWith("\n") ? "\n" : "") + run.lines.join("\n");
      if ($("#xFollow").checked) log.scrollTop = log.scrollHeight;
    }
    runAfter = run.n;
  }
  const running = run.state === "running";
  const setupSecs = Math.round((run.ended || state.now) - run.started);
  $("#sState").textContent = `${run.spec.mode === "setup" ? run.spec.action : run.spec.mode} · ${run.state} · ` +
    `${Math.floor(setupSecs / 60)}m ${setupSecs % 60}s`;
  $("#sStop").classList.toggle("hidden", !running);
  if (run.spec.mode === "setup" && !running && renderRun._last !== run.started) {    // a setup job ended: new status
    renderRun._last = run.started;
    loadSetup();
  }
  $("#xStart").classList.toggle("hidden", running);
  $("#xStop").classList.toggle("hidden", !running);
  const secs = Math.round((run.ended || state.now) - run.started);
  $("#xState").textContent = `${run.state} · ${Math.floor(secs / 60)}m ${secs % 60}s` + (run.exit_code != null ? ` · exit ${run.exit_code}` : "");
  if (run.summary && run.summary.kind === "load") {
    const s = run.summary;
    $("#xSummaryCard").classList.remove("hidden");
    $("#xResultDir").textContent = run.result_dir || "";
    $("#xStats").innerHTML = stat(s.calls, "requests") + stat(s.failures, "failed") + stat(fmt(s.rps, 1), "requests/s") +
      stat(s.peak_load, "peak load") + stat(fmt(s.p50), "p50 ms") + stat(fmt(s.p95), "p95 ms") + stat(fmt(s.p99), "p99 ms") +
      (s.cold_share != null ? stat(fmt(s.cold_share * 100, 1) + "%", "cold (> 1 s)") : "");
  } else if (run.summary) {
    const s = run.summary;
    $("#xSummaryCard").classList.remove("hidden");
    $("#xResultDir").textContent = run.result_dir || "";
    $("#xStats").innerHTML = stat(s.calls, "tool calls") + stat(s.cold, "cold calls") + stat(fmt(s.cold_share * 100, 1) + "%", "cold share") +
      stat(fmt(s.mean), "mean ms") + stat(fmt(s.p50), "p50 ms") + stat(fmt(s.p95), "p95 ms") + stat(fmt(s.p99), "p99 ms") +
      stat(fmt(s.pod_seconds), "pod-seconds") + (s.hints ? stat(`${s.hints.answered}/${s.hints.sent}`, "hints answered") : "");
  }
}

// ------------------------------------------------------------------ results tab
let RESULTS = { runs: [], compares: [] };
async function loadResults() {
  try {
    RESULTS = await api("/api/results");
    renderResults();
  } catch (e) { toast(e.message); }
}
$("#rRefresh").onclick = loadResults;
$("#rWhere").onchange = renderResults;
function renderResults() {
  const w = $("#rWhere").value;
  const runs = RESULTS.runs.filter((r) => !w || r.where === w);
  const wheres = [...new Set(RESULTS.runs.map((r) => r.where))];
  $("#rWhere").innerHTML = `<option value="">all folders</option>` + wheres.map((x) => `<option ${x === w ? "selected" : ""}>${esc(x)}</option>`).join("");
  $("#rTbl tbody").innerHTML = runs.map((r) => `<tr>
      <td><input type="checkbox" data-path="${esc(r.path)}"></td>
      <td class="name" title="${esc(r.path)}">${esc(r.name)}</td><td class="muted">${esc(r.where)}</td>
      <td>${esc(r.config)}${r.kind === "load" ? ` <span class="pill">${esc(r.mode)} · ${esc(r.engine)}</span>` : ""}</td><td>${esc(r.predictor || "")}</td>
      <td>${r.calls ?? ""}</td><td>${r.kind === "load" ? (r.failures ?? "") : (r.cold ?? "")}</td><td>${fmt(r.mean)}</td><td>${fmt(r.p50)}</td><td>${fmt(r.p95)}</td><td>${fmt(r.p99)}</td>
      <td>${r.kind === "load" ? `${fmt(r.rps, 1)}/s @ ${r.peak_load ?? "–"}` : fmt(r.pod_seconds)}</td><td><a href="/api/results/zip?path=${encodeURIComponent(r.path)}" title="Download this run folder as a zip">zip</a></td>
      <td><a href="#" data-graf="${esc(r.path)}" title="This run's time window in the Monitoring tab (needs the cluster's Prometheus to still hold it)">Grafana</a></td></tr>`).join("") ||
    `<tr><td colspan="14" class="muted">No runs.</td></tr>`;
  $$("#rTbl input").forEach((c) => (c.onchange = selChanged));
  $$("#rTbl a[data-graf]").forEach((a) => (a.onclick = (e) => { e.preventDefault(); grafanaForRun(a.dataset.graf); }));
  selChanged();
  $("#rGallery").innerHTML = RESULTS.compares.map(galleryHtml).join("") || `<div class="empty">No comparison plots yet.</div>`;
}
const galleryHtml = (c) => `<div class="set"><h4>${esc(c.name)}</h4><div class="imgs">` +
  c.pngs.map((p) => `<a href="/files/${esc(c.path)}/${esc(p)}" target="_blank"><img loading="lazy" src="/files/${esc(c.path)}/${esc(p)}" alt="${esc(p)}"></a>`).join("") + "</div></div>";
function selected() { return $$("#rTbl input:checked").map((c) => c.dataset.path); }
function selChanged() {
  const n = selected().length;
  $("#rCompare").disabled = n < 1;
  $("#rPlot").disabled = n < 1;
  $("#rState").textContent = n ? `${n} selected` : "";
}
$("#rCompare").onclick = () => {
  const runs = selected().map((p) => RESULTS.runs.find((r) => r.path === p));
  const used = {};
  const color = (r) => {         // C0 orange, C1 blue, C3 with the main predictor (lightgbm) green, others extra colours
    const k = r.config + (r.predictor || "");
    if (!used[k]) used[k] = r.config !== "C3" || !r.predictor || r.predictor === "gbt" || r.predictor === "lightgbm"
      ? CONFIG_COLOR[r.config] || EXTRA[0] : EXTRA[Object.keys(used).length % EXTRA.length];
    return used[k];
  };
  const load = runs.every((r) => r.kind === "load");
  if (!load && runs.some((r) => r.kind === "load")) return toast("Compare either experiment runs or load runs, not both");
  const label = (r) => `${r.config}${r.predictor ? " " + r.predictor : ""}${load ? ` · ${r.target} · ${r.engine}` : ""}`;
  const metrics = load
    ? [["failures", "Failed requests"], ["mean", "Mean latency (ms)"], ["p95", "p95 (ms)"], ["p99", "p99 (ms)"], ["rps", "Requests/s"]]
    : [["cold", "Cold calls"], ["mean", "Mean latency (ms)"], ["p95", "p95 (ms)"], ["p99", "p99 (ms)"], ["pod_seconds", "Pod-seconds"]];
  $("#rBars").innerHTML = metrics.map(([k, title]) => {
    const max = Math.max(...runs.map((r) => r[k] || 0), 1);
    return `<div class="bar-group"><h4>${title}</h4>` + runs.map((r) => `<div class="bar-row" title="${esc(r.name)}">
      <span class="lbl">${esc(label(r))}</span><div class="bar" style="width:${(100 * (r[k] || 0)) / max}%;background:${color(r)}"></div>
      <span class="val">${fmt(r[k])}</span></div>`).join("") + "</div>";
  }).join("");
  $("#rCompareCard").classList.remove("hidden");
};
$("#rPlot").onclick = async () => {
  $("#rState").textContent = "making plots…";
  try {
    const c = await api("/api/results/plot", { paths: selected() });
    RESULTS.compares.unshift(c);
    $("#rGallery").innerHTML = RESULTS.compares.map(galleryHtml).join("");
    $("#rState").textContent = "saved to " + c.path;
  } catch (e) { $("#rState").textContent = ""; toast(e.message, 8000); }
};

// ------------------------------------------------------------------ side panel renders
let eventsAfter = 0;
const flashed = {};
function renderPods(pods) {
  $("#pods").innerHTML = (pods || []).map((p) => {
    const n = p.ready + p.starting;
    const dots = Array.from({ length: p.ready }, (_, i) => `<span class="dot ${i < p.busy ? "busy" : "ready"}"></span>`).join("") +
      Array.from({ length: p.starting }, () => `<span class="dot starting"></span>`).join("") || `<span class="dot none"></span>`;
    return `<div class="pod ${n ? "" : "zero"} ${flashed[p.tool] > Date.now() ? "flash" : ""}"><span>${esc(p.tool)}</span><span class="dots">${dots}</span></div>`;
  }).join("") || `<div class="empty small">No pod data (is the cluster reachable?)</div>`;
}
function renderEvents(evs) {
  if (!evs.length) return;
  const box = $("#events");
  if (box.querySelector(".empty")) box.innerHTML = "";
  for (const e of evs) {
    const t = e.time ? (typeof e.time === "number" ? new Date(e.time * 1000) : new Date(e.time)) : new Date();
    const when = t.toLocaleTimeString([], { hour12: false });
    let what;
    if (e.event === "warmed") {
      flashed[e.tool] = Date.now() + 1500;
      what = `prewarm <span class="tool">${esc(e.tool)}</span>${e.desired_replicas > 1 ? " ×" + e.desired_replicas : ""}` +
        ` · ${e.was_cold ? "was cold → starting" : "already warm, window reset"}` + (e.probability != null ? ` · P ${Number(e.probability).toFixed(2)}` : "");
    } else if (e.event === "min_scale") {
      what = e.min_scale ? "every tool keeps a warm pod (C1)" : "scale to zero (idle tools go cold after 60 s)";
    } else if (e.event === "scaled_down" || e.event === "evicted") {
      what = `<span class="tool">${esc(e.tool)}</span> scaled down (idle 60 s)`;
    } else {
      what = `${esc(e.event)} <span class="tool">${esc(e.tool || "")}</span> ${esc(e.error || "")}`;
    }
    const d = document.createElement("div");
    d.className = "ev";
    d.innerHTML = `<span class="t">${when}</span><span>${what}</span>`;
    box.prepend(d);
  }
  while (box.children.length > 120) box.lastChild.remove();
}
function renderServices(sv) {
  const p = sv.predictor;
  const pill = $("#predPill");
  if (!p) { pill.textContent = "stopped"; pill.className = "pill"; $("#predInfo").textContent = ""; }
  else if (!p.running) { pill.textContent = "exited"; pill.className = "pill bad"; $("#predInfo").textContent = (p.log_tail || []).join(" ").slice(-240); }
  else if (!p.ready) { pill.textContent = "loading…"; pill.className = "pill wait"; $("#predInfo").textContent = predLabel(p.model); }
  else {
    pill.textContent = "ready"; pill.className = "pill ok";
    const h = p.health || {};
    const asked = p.policy || {};
    // lookahead asked for a predictor whose /info says it can't: it runs the policy without it
    const ignored = p.lookahead_supported === false && asked.lookahead;
    $("#policyNote").classList.toggle("hidden", !ignored);
    const changed = !predictorMatchesCard();
    $("#predInfo").innerHTML = (changed ? `<b class="warn-text">Running: ${esc(p.policy_label)}. The card's new settings are ` +
        `${applyPending || busyNow() ? "applied when the current episode ends" : "being applied…"}</b><br>` : "") +
      `${esc(predLabel(p.model))} ${inputBadges(p.inputs || PRED_INPUTS[p.model])} · ` +
      `${esc(ignored ? p.policy_label.replace(/ \+ lookahead.*/, "") : p.policy_label)} · CRDs ${esc(p.crd)}` +
      (sv.controller ? ` · controller ${sv.controller.deployment ? "(deploy/" + esc(sv.controller.deployment) + ") " : ""}${sv.controller.running ? "running" : "not running"}` : "") +
      (h.crd && h.crd.written != null ? ` · ${h.crd.written} written` : "") +
      (p.serving_version ? ` · model ${esc(p.serving_version)}` : "") +
      (MON.figures && MON.figures.predictor_p95_ms != null
        ? ` · <span title="Prometheus, last 2 min (NFR-4: p95 < 50 ms)" class="${MON.figures.predictor_p95_ms > 50 ? "warn-text" : ""}">p95 ${fmt(MON.figures.predictor_p95_ms, 1)} ms</span>` : "");
  }
}

// ------------------------------------------------------------------ deployment: code version, GitOps (Argo CD)
function renderDeployment(d) {
  if (!d) return;
  $("#depPill").textContent = d.image ? `image ${d.image}` : (d.commit ? `commit ${d.commit}` : "");
  $("#depPill").classList.toggle("hidden", !d.commit && !d.image);
  const apps = d.apps || [];
  const ok = apps.filter((a) => a.sync === "Synced" && a.health === "Healthy").length;
  const pill = $("#gitopsPill");
  pill.classList.toggle("hidden", !d.argocd);
  pill.textContent = `GitOps ${ok}/${apps.length} synced`;
  pill.className = "pill " + (ok === apps.length ? "ok" : "wait") + (d.argocd ? "" : " hidden");
  pill.title = apps.map((a) => `${a.name}: ${a.sync} / ${a.health}${a.revision ? " @ " + a.revision : ""}`).join("\n");
  $("#argoLink").classList.toggle("hidden", !d.argocd_url);
  if (d.argocd_url) $("#argoLink").href = d.argocd_url;
}

// ------------------------------------------------------------------ models (the S3 registry: one tab per family)
// per family: the predictors it serves, the held-out numbers shown (the first one gates a release), how it trains
const FAMILIES = {
  pulse_v3: { predictors: ["pulse_v3"], how: "pulse.pulse_v3.train: train → evaluate on held-out tasks → publish",
              cols: [["warm@3", "held-out share of next-step calls warm with the top 3 prewarmed (the release gate)"],
                     ["hit@1", "next tool ranked first"], ["hit@3", "next tool in the top 3"]] },
  action: { predictors: ["lightgbm", "markov"], how: "pulse.predictor.train_action: train both → evaluate on the 10 held-out tasks → publish",
            cols: [["lightgbm_top1", "lightgbm: next action ranked first, held-out (the release gate)"],
                   ["lightgbm_hit@0.3", "lightgbm: share of tool calls prewarmed at θ 0.3"],
                   ["markov_top1", "markov: next action ranked first, held-out"]] },
};
const gWhere = seg("gWhere");
const gFamily = seg("gFamily", () => loadModels());
const fam = () => gFamily.get() || "pulse_v3";
let MODELS = { versions: [] };
async function loadModels() {
  $("#gNote").classList.add("hidden");
  try { MODELS = await api("/api/models?family=" + fam()); } catch (e) { MODELS = { versions: [], error: e.message }; }
  renderModels();
}
function renderModels() {
  const F = FAMILIES[fam()], gate = F.cols[0][0];
  const p = (state.services || {}).predictor;
  const serving = p && F.predictors.includes(p.model) && p.running ? p.serving_version : null;
  $("#gUri").textContent = MODELS.uri || "";
  $("#gHow").textContent = F.how;
  $("#gNote").innerHTML = esc(MODELS.error || "");
  $("#gNote").classList.toggle("hidden", !MODELS.error);
  $("#gTbl thead tr").innerHTML = "<th>Version</th><th>Created</th>" +
    F.cols.map(([k, t]) => `<th title="${esc(t)}">${esc(k)}</th>`).join("") +
    `<th title="the model it was compared with at training time (${esc(gate)})">vs</th><th>Notes</th><th></th>`;
  const mt = (m) => ((m.metrics || {}).new || {});
  $("#gTbl tbody").innerHTML = MODELS.versions.map((m) => {
    const cur = (m.metrics || {}).current || {};
    return `<tr><td class="mono">${esc(m.version)} ${m.latest ? '<span class="pill ok">latest</span>' : ""}` +
      `${m.version === serving ? ' <span class="pill wait">serving</span>' : ""}</td>
      <td class="muted">${esc((m.created || "").slice(0, 16).replace("T", " "))}</td>
      ${F.cols.map(([k], i) => `<td>${i === 0 ? "<b>" : ""}${fmt(mt(m)[k], 3)}${i === 0 ? "</b>" : ""}</td>`).join("")}
      <td class="muted small">${cur[gate] != null ? `${fmt(cur[gate], 3)} (${esc((m.metrics || {}).current_version || "")})` : "–"}</td>
      <td class="small">${esc(m.notes || "")}</td>
      <td><button class="small" data-use="${esc(m.version)}" title="Start ${esc(F.predictors.join(" / "))} with this version">Use</button>
        ${m.latest ? "" : `<button class="small ghost" data-promote="${esc(m.version)}" title="Make it the latest (also: roll back)">Promote</button>`}
        <button class="small ghost" data-pin="${esc(m.version)}" title="GitOps: pin this version for the cluster in infra/cloud/gitops/config.yaml (no commit)">Pin in Git</button></td></tr>`;
  }).join("") || `<tr><td colspan="${F.cols.length + 5}" class="muted">${MODELS.error ? "–" : "No versions yet."}</td></tr>`;
  $$("#gTbl [data-use]").forEach((b) => (b.onclick = async () => {
    try {
      const st = await api("/api/models/use", { family: fam(), version: b.dataset.use });
      predModel.value = (st.predictor || {}).model || F.predictors[0];
      policyForm();
      toast(`Starting ${predModel.value} with model ${b.dataset.use}`);
    } catch (e) { toast(e.message, 6000); }
  }));
  $$("#gTbl [data-pin]").forEach((b) => (b.onclick = async () => {
    try {
      const r = await api("/api/models/pin", { family: fam(), version: b.dataset.pin });
      toast(r.changed ? `Pinned ${b.dataset.pin} in ${r.file}. Next: ${r.next}` : `${r.file} already pins ${b.dataset.pin}`, 12000);
    } catch (e) { toast(e.message, 6000); }
  }));
  $$("#gTbl [data-promote]").forEach((b) => (b.onclick = async () => {
    if (!confirm(`Make ${b.dataset.promote} the latest model? Predictors that load "latest" pick it up when they restart.`)) return;
    try { MODELS = await api("/api/models/promote", { family: fam(), version: b.dataset.promote }); renderModels(); toast("Promoted"); }
    catch (e) { toast(e.message, 6000); }
  }));
}
$("#gRefresh").onclick = loadModels;
$("#gRetrain").onclick = async () => {
  try {
    await api("/api/models/retrain", { family: fam(), where: gWhere.get(), promote: $("#gPromote").value, notes: $("#gNotes").value.trim() });
    runAfter = 0;
    $("#xLog").textContent = "";
    $("#gVerdict").textContent = "training… (log: Experiments → Run log)";
    toast("Retraining started: the log streams in Experiments → Run log", 5000);
  } catch (e) { toast(e.message, 7000); }
};

// ------------------------------------------------------------------ monitoring (Prometheus + Grafana)
let MON = {};
let monResult = null;                       // {from, to, name}: a result run picked in the Results tab
$("#mDash").onchange = $("#mRange").onchange = () => renderMonTab();
$("#monInstall").onclick = async () => {
  if (!confirm("Install Prometheus + Grafana into the current cluster (infra/monitoring/install.sh, needs helm)?")) return;
  try {
    await api("/api/monitoring/install", {});
    runAfter = 0;
    $("#xLog").textContent = "";
    showTab("exp");
    toast("Installing monitoring: the log is in Run log", 5000);
  } catch (e) { toast(e.message, 7000); }
};
function renderMonitoring(m) {
  MON = m || {};
  const knative = state.cluster && state.cluster.mode === "knative";
  $("#monBox").classList.toggle("hidden", !knative);
  const pill = $("#monPill");
  pill.textContent = MON.reachable ? "monitoring on" : MON.installed === false ? "no monitoring" : "monitoring …";
  pill.className = "pill " + (MON.reachable ? "ok" : MON.installed === false ? "" : "wait");
  pill.title = MON.detail || "";
  $("#monLink").classList.toggle("hidden", !MON.reachable);
  $("#monInstall").classList.toggle("hidden", !(MON.installed === false && MON.can_install));
  if (!$("#mDash").options.length && MON.dashboards)
    $("#mDash").innerHTML = Object.entries(MON.dashboards).map(([k, v]) => `<option value="${k}">${esc(v)}</option>`).join("");
  if ($("#tab-mon").classList.contains("on")) renderMonTab();
}
async function grafanaForRun(path) {
  try {
    const w = await api(`/api/monitoring/link?path=${encodeURIComponent(path)}`);
    monResult = { from: w.from, to: w.to, name: path.split("/").pop() };
    const opt = $("#mRange option[value=result]");
    opt.disabled = false;
    opt.textContent = "result: " + monResult.name;
    $("#mRange").value = "result";
    showTab("mon");
  } catch (e) { toast(e.message); }
}
function monWindow() {
  // Grafana time range: a live window follows "now"; a result run is fixed. (seconds -> ms)
  const r = $("#mRange").value, run = state.run, s = state.session;
  if (r === "result" && monResult) return { from: monResult.from, to: monResult.to, live: false, label: monResult.name };
  if (r === "auto" && run && run.state === "running") return { from: Math.floor((run.started - 30) * 1000), to: "now", live: true, label: "the running experiment" };
  if (r === "auto" && s && s.t_start && ["running", "waiting", "starting"].includes(s.state))
    return { from: Math.floor((s.t_start - 30) * 1000), to: "now", live: true, label: "the running episode" };
  return { from: r === "last60" ? "now-1h" : "now-15m", to: "now", live: true, label: r === "last60" ? "last hour" : "last 15 min" };
}
function renderMonTab() {
  const note = $("#mNote"), frame = $("#mFrame");
  const knative = state.cluster && state.cluster.mode === "knative";
  const why = !knative ? "Monitoring needs the real cluster: switch the top bar to Knative. (The simulated cluster has no Prometheus.)"
    : MON.installed === false ? "Prometheus + Grafana are not installed in this cluster: use Install in the top bar, or run " +
      "<code>bash infra/monitoring/install.sh</code>." : !MON.reachable ? `Grafana is not reachable yet (${esc(MON.detail || "checking")}).` : "";
  note.innerHTML = why;
  note.classList.toggle("hidden", !why);
  frame.classList.toggle("hidden", !!why);
  if (why) return;
  const w = monWindow();
  const src = `/grafana/d/${$("#mDash").value}?orgId=1&kiosk&from=${w.from}&to=${w.to}` + (w.live ? "&refresh=10s" : "");
  $("#mOpen").href = src.replace("&kiosk", "");
  $("#mOpen").title = "Showing " + w.label;
  if (frame.getAttribute("src") !== src) frame.setAttribute("src", src);     // only reload when the window changes
}
const secs = (v) => (v == null ? "–" : `${Number(v).toFixed(1)} s`);
const prewarmLabel = (s) => (s.predictor === "none" ? "none (C0)" : `${s.predictor}${s.policy ? " · " + s.policy : ""}`);
function renderHistory(h) {
  $("#historyTbl tbody").innerHTML = h.map((s) => `<tr><td>${s.kind === "recorded" ? "replay of live run" : esc(s.kind)}${s.user && s.kind === "live" ? " · " + esc(s.user) : ""}</td>
    <td>${esc(s.task || "free")}${s.kind === "replay" && s.trial != null ? " #" + (s.trial + 1) : ""}</td><td>${esc(prewarmLabel(s))}</td><td>${esc(s.cluster)}</td>
    <td><b>${secs(s.e2e_s)}</b></td><td>${secs(s.tool_wait_s)}</td>
    <td>${s.calls}</td><td>${s.cold}</td><td>${s.prewarmed_hits}</td><td>${fmt(s.mean_ms)} ms</td><td>${fmt(s.cold_wait_s, 1)} s</td>
    <td>${fmt(s.pod_seconds)}</td><td>${s.hint_rtt_p50_ms != null ? fmt(s.hint_rtt_p50_ms, 1) + " ms" : "–"}</td></tr>`).join("") ||
    `<tr><td colspan="13" class="muted">None yet.</td></tr>`;
}

// ------------------------------------------------------------------ end-to-end clock + C0 vs C3 card
const elapsed = (s) => (s && s.t_start ? (s.t_end || state.now) - s.t_start : null);
function renderClock(s) {
  const run = s && !s.t_end && ["running", "waiting", "starting"].includes(s.state);
  $("#clock").classList.toggle("running", !!run);
  $("#clockVal").textContent = secs(elapsed(s) ?? 0);
  $("#toolWaitVal").textContent = secs(s ? s.tool_wait_s : 0);
}
// two episodes are the "same" if they replay the same recording (or, live, the same task with the same agent)
const raceKey = (x) => x.kind === "recorded" ? `recorded/${x.recording}`
  : [x.kind, x.task, x.kind === "replay" ? `${x.trial}|${x.speed}` : x.user].join("/");
function renderRace(history, s) {
  const cur = s && s.cfg ? { kind: s.cfg.kind, task: String(s.cfg.task_id || ""), trial: s.cfg.trial, speed: s.cfg.speed,
                              recording: s.cfg.recording, user: s.cfg.kind === "live" ? s.cfg.user : null,
                              predictor: s.cfg.predictor ? "x" : "none" } : null;
  const running = s && ["running", "waiting", "starting"].includes(s.state);
  const ref = running ? cur : history[0];
  const done = history.filter((h) => h.state === "done" && h.e2e_s != null && ref && raceKey(h) === raceKey({ ...ref, task: String(ref.task ?? "") }));
  const c0 = done.find((h) => h.predictor === "none"), c3 = done.find((h) => h.predictor !== "none");
  const lane = (id, sub, h, live) => {
    $(`#lane${id}`).classList.toggle("running", !!live);
    if (live) {
      $(`#e2e${id}`).textContent = secs(elapsed(s));
      $(`#sub${id}`).textContent = `running · tool wait ${secs(s.tool_wait_s)}`;
    } else if (h) {
      $(`#e2e${id}`).textContent = secs(h.e2e_s);
      $(`#sub${id}`).textContent = `tool wait ${secs(h.tool_wait_s)} · ${h.cold} cold of ${h.calls} calls`;
    } else {
      $(`#e2e${id}`).textContent = "–";
      $(`#sub${id}`).textContent = sub;
    }
  };
  const liveC0 = running && cur.predictor === "none", liveC3 = running && cur.predictor !== "none";
  lane("C0", "end-to-end", liveC0 ? null : c0, liveC0);
  lane("C3", "end-to-end", liveC3 ? null : c3, liveC3);
  $("#headC3").textContent = "C3 · " + (c3 && !liveC3 ? prewarmLabel(c3) : "predictor");
  const box = $("#raceSaving");
  if (c0 && c3 && !running) {
    const d = c0.e2e_s - c3.e2e_s, dw = c0.tool_wait_s - c3.tool_wait_s;
    box.classList.remove("hidden");
    box.classList.toggle("worse", d < 0);
    box.textContent = d >= 0
      ? `With prewarming the episode finished ${d.toFixed(1)} s faster (−${(100 * d / c0.e2e_s).toFixed(0)}%): ` +
        `waiting on tools ${c0.tool_wait_s.toFixed(1)} s → ${c3.tool_wait_s.toFixed(1)} s, cold calls ${c0.cold} → ${c3.cold}.`
      : `C3 took ${(-d).toFixed(1)} s longer end-to-end (tool wait ${dw >= 0 ? "−" : "+"}${Math.abs(dw).toFixed(1)} s).`;
  } else box.classList.add("hidden");
const rec = ref && ref.kind === "recorded" ? RECS.find((r) => r.id === ref.recording) : null;
  $("#raceNote").textContent = ref && ref.kind === "live"
    ? "Live episodes differ in LLM speed and customer replies, so end-to-end times vary by seconds between runs: compare the tool wait, or race the recording of this run (button at the end of the episode)."
    : rec
      ? `Replays of your live run ${rec.id} (${rec.created.slice(11, 16)}, task ${rec.task_id ?? "free"}): identical messages, think and reply times and tool calls, so the difference is only the waiting on tools. ` +
        `The original live run took ${secs(rec.live_e2e_s)} as ${rec.config} (tool wait ${secs(rec.live_tool_wait_s)}).`
      : "Replay: both runs have identical think and customer-reply gaps (seed 252), so the end-to-end difference is exactly the waiting on tools.";
}

// ------------------------------------------------------------------ the poll loop
let state = {};
let sessionId = null;
async function poll() {
  try {
    const s = await api(`/api/state?session_after=${sessionAfter}&run_after=${runAfter}&events_after=${eventsAfter}`);
    state = s;
    const c = s.cluster;
    clusterSeg.set(c.mode);
    if (c.domain && c.domain !== shownDomain) { shownDomain = c.domain; domainSeg.set(c.domain); domainForm(c.domain); }
    $("#ctxName").textContent = c.context ? `(${c.context.replace(/^.*@/, "")})` : "";
    $$(".sim-only").forEach((e) => e.classList.toggle("hidden", c.mode !== "sim"));
    $$(".knative-only").forEach((e) => e.classList.toggle("hidden", c.mode !== "knative"));
    minScaleSeg.set(String(s.sim_min_scale));
    const mf = (s.monitoring && s.monitoring.figures) || {};
    $("#podSec").textContent = c.mode === "sim" ? `${fmt(s.pod_seconds)} pod-s`
      : (c.context || "") + (mf.tool_pod_s_15m != null ? ` · ${fmt(mf.tool_pod_s_15m)} pod-s / 15 min (Prometheus)` : "");
    renderMonitoring(s.monitoring);
    renderDeployment(s.deployment);
    renderServices(s.services);
    renderPods(s.pods);
    if (s.events_n < eventsAfter) eventsAfter = 0;
    renderEvents(s.events);
    eventsAfter = s.events_n;
    if (s.session && s.session.id !== sessionId) {     // a new episode: start the timeline over
      sessionId = s.session.id;
      if (sessionAfter !== 0 || $("#timeline .empty")) { resetTimeline(); setTimeout(poll, 0); return; }
    }
    if (s.session) {
      s.session.events.forEach(renderEvent);
      sessionAfter = s.session.n;
      const live = ["running", "waiting", "starting"].includes(s.session.state);
      $("#startBtn").classList.toggle("hidden", live);
      $("#stopBtn").classList.toggle("hidden", !live);
      $("#sessionState").textContent = s.session.state === "waiting" ? "waiting for your reply" : s.session.state;
    }
    renderRun(s.run);
    renderHistory(s.history);
    renderClock(s.session);
    renderRace(s.history, s.session);
    if (applyPending && !busyNow()) applyCard();            // a card change made during an episode
  } catch (e) {
    $("#sessionState").textContent = "UI server not reachable";
  }
  setTimeout(poll, 500);
}

// Is the server running the same UI code as this page? An old server process (started before an update) serves the
// new page from disk but ignores new settings such as the Top-k policy. Then say so, loudly.
async function checkVersion() {
  let stale = null;
  try {
    const v = await api("/api/version");
    if (v.restart_needed) stale = "The UI server is running older code than the files on disk.";
  } catch (e) {
    stale = "The UI server is older than this page (it has no /api/version).";
  }
  const b = $("#staleBanner");
  b.classList.toggle("hidden", !stale);
  if (stale) b.innerHTML = `⚠ ${stale} New features (e.g. the Top-k policy) won't work until you restart it: ` +
    `press Ctrl+C in its terminal, then run <code>.\\agent\\.venv\\Scripts\\python.exe ui\\server.py</code> and reload this page.`;
}
setInterval(checkVersion, 30000);

(async function init() {
  checkVersion();
  try { TASKS = await api("/api/tasks"); } catch (e) { toast("Could not load tasks: " + e.message); }
  demoForm();
  expForm();
  policyForm();
  poll();
})();

"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowDownToLine, ArrowRight, ArrowUpRight, Box, Check, ChevronDown, ChevronRight, CircleHelp, Code2, Cpu, Eye, FileText, FlaskConical, ImageIcon, Layers3, LoaderCircle, MousePointer2, Pause, PenLine, Play, Radio, RotateCcw, ShieldCheck, SkipForward, Sparkles, Terminal, TriangleAlert, X } from "lucide-react";
import RobotSimulator from "@/components/RobotSimulator";
import Robot3DSimulator from "@/components/Robot3DSimulator";
import TopViewSimulator from "@/components/TopViewSimulator";
import { api } from "@/lib/api";
import { frames, Frame, HOME } from "@/lib/simulation";
import { Mode, PIPELINES, Pipeline, Result, Scene, pretty } from "@/lib/types";

const EMPTY: Frame = { positions: {}, held: null, gripper: HOME };
type Job = { status: "queued" | "running" | "complete" | "failed"; job_id?: string; pipeline?: Pipeline; result?: Result; error?: string; stage?: string; elapsed_seconds?: number };
type SceneData = { scenes: Scene[]; pipelines: { id: Pipeline; has_replay: boolean }[] };

export default function Playground() {
  const [scenes, setScenes] = useState<Scene[]>([]);
  const [sceneId, setSceneId] = useState("scene_01");
  const [pipeline, setPipeline] = useState<Pipeline>("path_a_graph");
  const [mode, setMode] = useState<Mode>("replay");
  const [instructionId, setInstructionId] = useState("scene_01_i1");
  const [instruction, setInstruction] = useState("");
  const [instructionSource, setInstructionSource] = useState<"manual" | "dataset">("dataset");
  const [result, setResult] = useState<Result | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [jobProgress, setJobProgress] = useState("Starting the pipeline");
  const [jobElapsed, setJobElapsed] = useState(0);
  const [loading, setLoading] = useState(true);
  const [connected, setConnected] = useState(false);
  const [liveEnabled, setLiveEnabled] = useState(true);
  const [tab, setTab] = useState<"robot" | "top" | "3d" | "scene">("top");
  const [showJson, setShowJson] = useState(false);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [progress, setProgress] = useState(1);
  const [animating, setAnimating] = useState(false);
  const [speed, setSpeed] = useState(1);
  const requestVersion = useRef(0);
  const progressRef = useRef(1);
  const jobTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const dialogRef = useRef<HTMLElement | null>(null);
  const scene = scenes.find(s => s.id === sceneId);
  const selectedInstruction = scene?.instructions.find(i => i.id === instructionId);
  const activeInstruction = instructionSource === "dataset" ? selectedInstruction?.text || "" : instruction;
  const timeline = useMemo(() => result ? frames(result) : [EMPTY], [result]);
  const topTimeline = useMemo(() => result ? frames(result, "top") : [EMPTY], [result]);
  const frame = timeline[Math.min(step, timeline.length - 1)];
  const previous = timeline[Math.max(0, step - 1)];
  const actions = result?.answer.actions || [];
  const decision = actions[0]?.action;
  const canPlay = result && decision !== "ask_user" && decision !== "refuse" && !frame.error;
  const activeAction = step > 0 ? actions[step - 1] : null;
  const playbackFinished = step >= timeline.length - 1 && !animating;
  const canReplay = selectedInstruction?.replays.includes(pipeline);

  useEffect(() => {
    if (!showJson) return;
    const priorFocus = document.activeElement as HTMLElement | null;
    const buttons = dialogRef.current?.querySelectorAll<HTMLButtonElement>("button:not(:disabled)");
    buttons?.[0]?.focus();
    const priorOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    function keydown(event: KeyboardEvent) {
      if (event.key === "Escape") setShowJson(false);
      if (event.key === "Tab" && buttons?.length) {
        const first = buttons[0], last = buttons[buttons.length - 1];
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
      }
    }
    document.addEventListener("keydown", keydown);
    return () => { document.body.style.overflow = priorOverflow; document.removeEventListener("keydown", keydown); priorFocus?.focus(); };
  }, [showJson]);

  function resetPlayback() {
    setPlaying(false); setAnimating(false); setStep(0); setProgress(1); progressRef.current = 1;
  }
  function clearResult() {
    requestVersion.current++; setResult(null); setError(""); resetPlayback();
  }
  function acceptResult(value: Result) { setResult(value); setTab("top"); resetPlayback(); setError(""); }
  async function connect() {
    setLoading(true); setError("");
    try {
      const [data, health] = await Promise.all([api<SceneData>("/scenes"), api<{ live_enabled: boolean }>("/health")]);
      setScenes(data.scenes); setConnected(true); setLiveEnabled(health.live_enabled);
      const first = data.scenes[0];
      if (first) {
        setSceneId(first.id); setInstructionId(first.instructions[0]?.id || "");
        if (!first.instructions.length) return;
        const saved = await api<Job>("/plans", { method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ mode: "replay", scene_id: first.id, pipeline: "path_a_graph", instruction_id: first.instructions[0].id }) });
        if (saved.result) acceptResult(saved.result);
      }
    } catch (e) { setConnected(false); setError((e as Error).message); }
    finally { setLoading(false); }
  }
  useEffect(() => {
    void connect();
    return () => { requestVersion.current++; if (jobTimer.current) clearTimeout(jobTimer.current); };
    // Initialization only; subsequent requests use the current selected scene.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!playing || !animating) return;
    let raf = 0, last = performance.now();
    function tick(now: number) {
      progressRef.current = Math.min(1, progressRef.current + (now - last) / (1900 / speed)); last = now;
      setProgress(progressRef.current);
      if (progressRef.current < 1) raf = requestAnimationFrame(tick);
      else setAnimating(false);
    }
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, animating, speed]);

  useEffect(() => {
    if (!playing || animating) return;
    if (step >= timeline.length - 1 || frame.error) { setPlaying(false); return; }
    const timer = setTimeout(() => {
      progressRef.current = 0; setProgress(0); setStep(s => s + 1); setAnimating(true);
    }, step === 0 ? 80 : 450 / speed);
    return () => clearTimeout(timer);
  }, [playing, animating, step, timeline.length, frame.error, speed]);

  async function generate() {
    setBusy(true); setError(""); resetPlayback(); setResult(null);
    setJobProgress("Starting the pipeline"); setJobElapsed(0);
    const version = ++requestVersion.current;
    try {
      let job = await api<Job>("/plans", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({
        scene_id: sceneId, pipeline, mode,
        instruction_id: instructionSource === "dataset" ? instructionId : null,
        instruction: activeInstruction, instruction_source: instructionSource,
      }) });
      if (job.job_id) {
        const jobId = job.job_id;
        job = await new Promise<Job>((resolve, reject) => {
          async function poll() {
            if (version !== requestVersion.current) return;
            try {
              const update = await api<Job>(`/jobs/${jobId}`);
              setJobProgress(update.stage || (update.status === "queued" ? "Waiting for the pipeline" : "Preparing the result"));
              setJobElapsed(update.elapsed_seconds || 0);
              if (update.status === "complete" || update.status === "failed") resolve(update);
              else jobTimer.current = setTimeout(poll, 1200);
            } catch (e) { reject(e); }
          }
          jobTimer.current = setTimeout(poll, 1000);
        });
      }
      if (version !== requestVersion.current) return;
      if (job.result) acceptResult(job.result);
      else throw new Error(`${job.pipeline || pipeline}${job.job_id ? ` · Job ${job.job_id}` : ""}: ${job.error || "No result returned."}`);
    } catch (e) { if (version === requestVersion.current) setError((e as Error).message); }
    finally { if (version === requestVersion.current) setBusy(false); }
  }

  function selectInstruction(id: string) {
    setInstructionId(id); clearResult();
  }
  function selectScene(id: string) {
    const choice = scenes.find(s => s.id === id);
    setSceneId(id); setInstructionId(choice?.instructions[0]?.id || ""); clearResult();
  }
  function play() {
    if (playbackFinished) { setStep(0); setAnimating(false); progressRef.current = 1; setProgress(1); }
    setPlaying(value => !value);
  }
  function advance() {
    setPlaying(false); setAnimating(false); setProgress(1); progressRef.current = 1;
    if (!animating) setStep(s => Math.min(timeline.length - 1, s + 1));
  }
  function download() {
    if (!result) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(result, null, 2)], { type: "application/json" }));
    const anchor = document.createElement("a"); anchor.href = url; anchor.download = `${result.scene_id}-${result.pipeline}-${result.mode}.json`; anchor.click(); URL.revokeObjectURL(url);
  }

  const stageStatus = busy ? "Planning your move…" : frame.error ? "Playback paused" : !result ? "Ready when you are" : decision === "refuse" ? "Movement refused" : decision === "ask_user" ? "A little clarification" : playing ? "Robot in motion" : playbackFinished ? "Sequence complete" : step ? "Playback paused" : "Plan ready to play";

  return <div className="app-shell">
    <header className="topbar">
      <a className="brand" href="/" aria-label="Pick Lab home"><span className="brand-icon"><Box size={23} strokeWidth={1.7} /></span><span>pick<span className="brand-light">lab</span><span className="brand-dot">.</span></span></a>
      <div className="topbar-center"><span className="tiny-dot" /> APERTUS ROBOT PLAYGROUND</div>
      <div className="topbar-right"><span className={`connection ${connected ? "online" : ""}`}><span />{connected ? "Local backend connected" : "Backend offline"}</span><span className="hackathon-tag"><FlaskConical size={14} /> Hackathon edition</span></div>
    </header>

    <main>
      <div className="page-heading"><div><div className="eyebrow"><span /> SEE IT. SAY IT. PICK IT.</div><h1>Small instructions.<br className="mobile-break" /> <span>Real possibilities.</span></h1><p>Give Apertus a task. Watch the plan come to life, one move at a time.</p></div><div className="heading-sticker"><Sparkles size={18} /><span>A little language.<br /><strong>A lot of motion.</strong></span></div></div>

      {error && <div className="error-banner" role="alert"><TriangleAlert size={18} /><span>{error}</span>{!connected && <button onClick={() => void connect()} disabled={loading}>Reconnect</button>}<button className="icon-button" onClick={() => setError("")} aria-label="Dismiss error"><X size={17} /></button></div>}

      <div className="workspace">
        <aside className="input-panel card">
          <div className="panel-title"><span className="number-badge">01</span><h2>Set the scene</h2><MousePointer2 size={17} /></div>
          <label className="field-label" htmlFor="scene">WORKSPACE</label>
          <div className="select-wrap"><ImageIcon size={16} /><select id="scene" value={sceneId} onChange={e => selectScene(e.target.value)} disabled={busy || loading}>{scenes.map(s => <option value={s.id} key={s.id}>Scene {s.id.split("_")[1]} · Robot table</option>)}</select><ChevronDown size={15} /></div>
          <div className="scene-preview">{connected ? <img src={`/api/scenes/${sceneId}/image`} alt={`Original robot table photo for ${sceneId}`} /> : <div className="preview-placeholder"><ImageIcon size={24} /><span>Connect to load scenes</span></div>}<span className="photo-badge"><Eye size={12} /> Original scene</span><span className="scene-count">{scenes.length || "—"} scenes</span></div>
          <div className="section-divider" />
          <label className="field-label" htmlFor="pipeline">THE BRAIN</label>
          <div className="select-wrap"><Cpu size={17} /><select id="pipeline" value={pipeline} onChange={e => { setPipeline(e.target.value as Pipeline); clearResult(); }} disabled={busy}>{PIPELINES.map(p => <option value={p.id} key={p.id}>{p.label}</option>)}</select><ChevronDown size={15} /></div>
          <p className="field-hint">{PIPELINES.find(p => p.id === pipeline)?.detail}</p>
          <div className="mode-toggle" aria-label="Inference mode"><button className={mode === "replay" ? "selected" : ""} disabled={busy} onClick={() => { setMode("replay"); setInstructionSource("dataset"); clearResult(); }}><Layers3 size={14} /> Replay</button><button className={mode === "live" ? "selected" : ""} disabled={busy || !liveEnabled} onClick={() => { setMode("live"); clearResult(); }}><Radio size={14} /> Live Apertus</button></div>
          <p className="mode-note">{mode === "replay" ? "Saved results. No model calls." : "Calls Apertus when you generate a plan."}</p>
          <div className="section-divider" />
          <div className="instruction-heading"><label className="field-label" htmlFor="instruction">WHAT SHOULD THE ROBOT DO?</label><Sparkles size={15} /></div>
          <div className="mode-toggle instruction-source" role="group" aria-label="Instruction source"><button aria-pressed={instructionSource === "manual"} className={instructionSource === "manual" ? "selected" : ""} disabled={busy} onClick={() => { setInstructionSource("manual"); setMode("live"); clearResult(); }}><PenLine size={14} /> Type manually</button><button aria-pressed={instructionSource === "dataset"} className={instructionSource === "dataset" ? "selected" : ""} disabled={busy} onClick={() => { setInstructionSource("dataset"); clearResult(); }}><FileText size={14} /> From dataset</button></div>
          {instructionSource === "dataset" && <>
            <p className="instruction-file"><FileText size={12} /><span>{scene?.instruction_file ? `dataset/${sceneId}/${scene.instruction_file}` : "No instruction file in this scene"}</span></p>
            <label className="sr-only" htmlFor="saved-instruction">Instruction from selected scene</label>
            <div className="select-wrap dataset-instructions"><select id="saved-instruction" value={instructionId} onChange={e => selectInstruction(e.target.value)} disabled={busy || !scene?.instructions.length}>{scene?.instructions.length ? scene.instructions.map((i, index) => <option key={i.id} value={i.id}>{index + 1}. {i.text}</option>) : <option value="">No dataset instructions available</option>}</select><ChevronDown size={14} /></div>
          </>}
          <textarea id="instruction" value={activeInstruction} readOnly={instructionSource === "dataset"} disabled={busy} onChange={e => { setInstruction(e.target.value); clearResult(); }} placeholder={instructionSource === "manual" ? "Type your instruction, e.g. pick up the red cube…" : "Select an instruction from this scene…"} rows={3} maxLength={2000} />
          <p className="instruction-help">{instructionSource === "manual" ? liveEnabled ? "Your own words. Uses Live Apertus when you generate." : "You can type a draft; live inference is disabled in this session." : `${scene?.instructions.length || 0} instructions from this scene. ${mode === "replay" ? "Load a saved result." : "Send the selected instruction to Apertus."}`}</p>
          <button className="generate-button" onClick={() => void generate()} disabled={busy || loading || !connected || (mode === "replay" ? !canReplay : !activeInstruction.trim() || !liveEnabled)}>{busy ? <LoaderCircle size={18} className="spin" /> : <Sparkles size={17} />}<span>{busy ? "Apertus is planning…" : mode === "replay" ? "Load saved plan" : "Generate a plan"}</span>{!busy && <ArrowRight size={17} />}</button>
          {mode === "replay" && !canReplay && connected && <p className="unavailable-note">No saved result for this selection. Try Path A or switch to Live.</p>}
          <div className="input-footnote"><ShieldCheck size={15} /><span>Plans are checked before playback.</span></div>
        </aside>

        <section className="simulation-panel card">
          <div className="simulation-header"><div className="panel-title"><span className="number-badge">02</span><h2>Watch it happen</h2></div><div className="view-tabs"><button aria-pressed={tab === "top"} className={tab === "top" ? "active" : ""} onClick={() => setTab("top")}><Layers3 size={14} /> Top view</button><button aria-pressed={tab === "robot"} className={tab === "robot" ? "active" : ""} onClick={() => setTab("robot")}><Box size={14} /> 2D robot</button><button aria-pressed={tab === "3d"} className={tab === "3d" ? "active" : ""} onClick={() => setTab("3d")}><Box size={14} /> 3D robot</button><button aria-pressed={tab === "scene"} className={tab === "scene" ? "active" : ""} onClick={() => setTab("scene")}><ImageIcon size={14} /> Scene</button></div></div>
          <div className="stage">
            <div className="stage-top"><div className="stage-label"><span className={`status-dot ${playing || busy ? "pulse" : ""}`} />{stageStatus}</div><span className="mode-badge">{result ? result.mode === "replay" ? "SAVED REPLAY" : "LIVE RESULT" : "PLAYGROUND"}</span></div>
            {tab === "robot" ? <RobotSimulator result={result} frame={frame} previous={previous} progress={progress} animated={animating} activeObject={activeAction?.object} /> : tab === "top" ? <TopViewSimulator result={result} timeline={topTimeline} step={step} progress={progress} animated={animating} activeObject={activeAction?.object} /> : tab === "3d" ? <Robot3DSimulator result={result} frame={frame} previous={previous} progress={progress} animated={animating} /> : <div className="photo-stage">{connected && <div className="photo-overlay"><img src={`/api/scenes/${sceneId}/image`} alt="Original scene with available object target positions" />{result?.objects.filter(o => o.pixel).map(o => <span key={o.id} className={`object-marker ${o.id === activeAction?.object ? "marker-active" : ""}`} style={{ left: `${o.pixel![0] / result.image_size[0] * 100}%`, top: `${o.pixel![1] / result.image_size[1] * 100}%` }}><span /><b>{pretty(o.id)}</b></span>)}</div>}</div>}
            {busy && <div className="planning-overlay"><div><LoaderCircle className="spin" size={30} /><strong>Turning words into moves</strong><span>{jobProgress}<br /><b>{Math.floor(jobElapsed / 60)}m {jobElapsed % 60}s elapsed</b><br />{jobElapsed >= 90 ? "Still waiting. Model requests can retry when the endpoint is busy." : "Vision and planning may take a few minutes."}<br />You can leave this page open.</span></div></div>}
            {result && (decision === "refuse" || decision === "ask_user") && <div className={`decision-overlay ${decision}`}><div>{decision === "refuse" ? <ShieldCheck size={26} /> : <CircleHelp size={26} />}<strong>{decision === "refuse" ? "Let’s keep this safe." : "One question before we move."}</strong><p>{actions[0].reason || actions[0].question}</p>{decision === "ask_user" && <span>Switch to Live and submit a clarified instruction.</span>}</div></div>}
            <div className="stage-bottom"><span><span className="legend-dot teal" /> Robot arm</span><span><span className="legend-dot coral" /> Detected objects</span><span className="stage-caption">Illustrative motion · schematic placement</span></div>
          </div>
          <div className="playback"><div className="playback-buttons"><button className="play-button" onClick={play} disabled={!canPlay || busy} aria-label={playing ? "Pause animation" : "Play animation"}>{playing ? <Pause size={17} fill="currentColor" /> : <Play size={17} fill="currentColor" />}{playing ? "Pause" : playbackFinished && result ? "Replay" : "Play"}</button><button className="icon-button" onClick={advance} disabled={!canPlay || playbackFinished || busy} title="Next action" aria-label="Next action"><SkipForward size={18} /></button><button className="icon-button" onClick={resetPlayback} disabled={!result || busy} title="Reset scene" aria-label="Reset scene"><RotateCcw size={17} /></button></div><div className="playback-progress"><div className="progress-track"><span style={{ width: `${actions.length ? Math.max(0, step - (animating ? 1 - progress : 0)) / actions.length * 100 : 0}%` }} /></div><span>{step} / {actions.length} actions</span></div><div className="speed-control"><label htmlFor="speed" className="sr-only">Playback speed</label><select id="speed" value={speed} onChange={e => setSpeed(Number(e.target.value))}><option value={.5}>0.5×</option><option value={1}>1× speed</option><option value={1.5}>1.5×</option><option value={2}>2×</option></select><ChevronDown size={13} /></div></div>
          <div className="scene-note"><CircleHelp size={14} /><span>{tab === "top" ? "Top view groups objects by the scene graph's tray membership and uses pixels for relative layout. Placement is schematic; unknown trays stay unassigned." : result?.geometry_source === "saved_pixels" ? "Replay positions come from saved actions. Objects without coordinates use schematic positions." : "Robot view simplifies the camera scene. Use Scene to inspect original target pixels."}</span></div>
        </section>

        <aside className="actions-panel card">
          <div className="panel-title"><span className="number-badge">03</span><h2>The action plan</h2><span className="action-count">{actions.length}</span></div>
          <div className="plan-status"><div className={`plan-status-icon ${decision === "refuse" || decision === "ask_user" ? "amber" : ""}`}>{decision === "refuse" ? <ShieldCheck size={19} /> : decision === "ask_user" ? <CircleHelp size={19} /> : <Check size={19} />}</div><div><strong>{!result ? "Waiting for a plan" : decision === "refuse" ? "Refused safely" : decision === "ask_user" ? "Clarification needed" : "Ready for playback"}</strong><span>{result ? PIPELINES.find(p => p.id === result.pipeline)?.label : "Start with a saved example"}</span></div></div>
          <div className="action-list" aria-live="polite">{actions.length ? actions.map((action, index) => <div className={`action-item ${step === index + 1 ? "current" : ""} ${step > index + 1 ? "done" : ""}`} key={index}><div className="action-step">{step > index + 1 ? <Check size={13} /> : String(index + 1).padStart(2, "0")}</div><div><div className="action-label">{action.action === "pick" ? <ArrowUpRight size={15} /> : action.action === "place" ? <ArrowDownToLine size={15} /> : action.action === "refuse" ? <ShieldCheck size={15} /> : <CircleHelp size={15} />}<strong>{action.action === "ask_user" ? "Ask user" : action.action[0].toUpperCase() + action.action.slice(1)}</strong>{step === index + 1 && <span className="now-tag">{playing ? "NOW" : "ACTIVE"}</span>}</div><p>{action.object ? `${action.action === "place" ? "Onto " : ""}${pretty(action.object)}` : action.question || action.reason}</p>{action.pixel && <code>[{action.pixel.join(", ")}]</code>}</div></div>) : <div className="empty-actions"><Layers3 size={28} /><p>A plan starts with<br />a little intention.</p><span>Choose an instruction to begin.</span></div>}</div>
          <div className="held-state"><Box size={16} /><span>Gripper</span><strong>{frame.held ? pretty(frame.held) : "Empty"}</strong></div>
          {frame.error && <div className="unavailable-note" role="alert">{frame.error}</div>}
          <div className="result-stats"><div><span>MODEL TIME</span><strong>{result?.elapsed_seconds != null ? `${result.elapsed_seconds.toFixed(1)}s` : "—"}</strong></div><div><span>RESULT SOURCE</span><strong>{result?.mode === "replay" ? "Saved run" : result ? "Live run" : "—"}</strong></div></div>
          <button className="json-button" onClick={() => setShowJson(true)} disabled={!result}><Code2 size={16} /> Inspect result JSON <ChevronRight size={15} /></button>
        </aside>
      </div>

      <div className="pipeline-strip"><div className="pipeline-caption"><span className="tiny-dot" /><strong>From words to movement</strong><span>Apertus thinks. Code checks. The robot follows.</span></div><div className="pipeline-flow"><span><Eye size={15} /> Perception</span><ChevronRight size={14} /><span><Cpu size={15} /> Apertus planner</span><ChevronRight size={14} /><span><ShieldCheck size={15} /> Validation</span><ChevronRight size={14} /><span className="flow-last"><Box size={15} /> Simulation</span></div></div>
      <footer><span>Built for Hack Apertus <span className="footer-dot">✦</span> Powered by curiosity.</span><span>2D / 3D action visualization · {mode === "replay" ? "No API calls in replay" : "Live inference on request"}</span></footer>
    </main>

    {showJson && result && <div className="modal-backdrop" onClick={() => setShowJson(false)}><section ref={dialogRef} className="json-modal" role="dialog" aria-modal="true" aria-labelledby="json-title" onClick={e => e.stopPropagation()}><div className="modal-heading"><h2 id="json-title"><Terminal size={18} /> Result JSON</h2><button className="icon-button" onClick={() => setShowJson(false)} aria-label="Close JSON"><X size={20} /></button></div><p>Scene geometry, original actions, and result metadata.</p><pre>{JSON.stringify(result, null, 2)}</pre><button className="generate-button" onClick={download}><ArrowDownToLine size={17} /> Download result</button></section></div>}
  </div>;
}

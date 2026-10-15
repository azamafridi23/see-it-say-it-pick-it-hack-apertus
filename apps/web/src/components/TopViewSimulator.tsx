"use client";

import { useEffect, useState } from "react";
import { Frame, interpolate, isSurface, TOP_HOME, TOP_TRAYS } from "@/lib/simulation";
import { Result, SceneObject, pretty } from "@/lib/types";

function ObjectGlyph({ object }: { object: SceneObject }) {
  const type = `${object.type} ${object.colour || ""}`.toLowerCase();
  const fill = type.includes("red") ? "#ef806f" : /yellow|banana/.test(type) ? "#edcc63" : /mug|green|cup/.test(type) ? "#67b5a1" : /mandarin|orange/.test(type) ? "#efa158" : "#bfd0da";
  if (type.includes("gear")) return <g>{Array.from({ length: 10 }, (_, i) => <rect key={i} x="-4" y="-22" width="8" height="12" rx="2" fill={fill} transform={`rotate(${i * 36})`} />)}<circle r="17" fill={fill} /><circle r="6" fill="#ffffffb0" /></g>;
  if (/mug|cup/.test(type)) return <g><rect x="12" y="-9" width="16" height="18" rx="6" fill="none" stroke={fill} strokeWidth="5" /><circle r="19" fill={fill} /><circle r="13" fill="#335f55" /><circle r="10" fill="#ffffff25" /></g>;
  if (/plate|saucer/.test(type)) return <g><circle r="24" fill="#eef0e6" stroke="#b0c5b5" strokeWidth="2" /><circle r="17" fill="none" stroke="#b0c5b5" strokeWidth="2" /></g>;
  if (/mandarin|orange|fruit/.test(type)) return <g><circle r="18" fill={fill} /><path d="M0 -15 Q10 -24 14 -16" stroke="#669c65" strokeWidth="4" fill="none" /></g>;
  if (type.includes("banana")) return <path d="M-22 -15 Q0 22 22 -12 Q15 35 -15 12 Z" fill={fill} />;
  return <g><rect x="-18" y="-18" width="36" height="36" rx="6" fill={fill} /><path d="M-10 -11 H10" stroke="#ffffff65" strokeWidth="3" strokeLinecap="round" /></g>;
}

export default function TopViewSimulator({ result, timeline, step, progress, animated, activeObject }: {
  result: Result | null; timeline: Frame[]; step: number; progress: number; animated: boolean; activeObject?: string;
}) {
  const [reducedMotion, setReducedMotion] = useState(false);
  useEffect(() => {
    const media = window.matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => setReducedMotion(media.matches);
    update(); media.addEventListener("change", update);
    return () => media.removeEventListener("change", update);
  }, []);
  const frame = timeline[Math.min(step, timeline.length - 1)];
  const previous = timeline[Math.max(0, step - 1)];
  const moving = animated && !reducedMotion && progress < 1;
  const t = moving ? progress * progress * (3 - 2 * progress) : 1;
  const gripper = frame ? interpolate(previous.gripper, frame.gripper, t) : TOP_HOME;
  const held = moving ? previous.held : frame?.held;
  const objects = result?.objects.filter(o => !isSurface(o)) || [];
  const unknown = objects.some(o => !o.in || !(o.in in TOP_TRAYS));
  return <svg viewBox="0 0 840 460" className="robot-svg top-svg" role="img" aria-label="Top view of objects in grey and black trays with animated gripper">
    <defs><pattern id="top-grid" width="24" height="24" patternUnits="userSpaceOnUse"><circle cx="1" cy="1" r="1" fill="#bdccbb" opacity=".5" /></pattern></defs>
    <rect width="840" height="460" fill="url(#top-grid)" />
    {Object.entries(TOP_TRAYS).map(([id, tray]) => <g key={id}>
      <rect x={tray.x} y={tray.y} width={tray.width} height={tray.height} rx="21" fill={id === "black_tray" ? "#344b45" : "#dfe5da"} stroke={id === "black_tray" ? "#536e62" : "#bac9b5"} strokeWidth="5" />
      <rect x={tray.x + 12} y={tray.y + 12} width={tray.width - 24} height={tray.height - 24} rx="13" fill="none" stroke={id === "black_tray" ? "#ffffff15" : "#ffffff90"} strokeWidth="2" />
      <text x={tray.x + 18} y={tray.y + tray.height + 24} fill="#708571" fontSize="11" letterSpacing="1.5">{id === "black_tray" ? "BLACK TRAY / MAT" : "GREY TRAY"}</text>
    </g>)}
    {unknown && <g><rect x="64" y="353" width="712" height="73" rx="13" fill="#ffffff60" stroke="#c3cebb" strokeDasharray="5 4" /><text x="77" y="446" fill="#859579" fontSize="10">TRAY UNKNOWN · saved results may not include tray membership</text></g>}
    {objects.map(object => {
      if (!frame?.positions[object.id]) return null;
      let position = moving ? previous.positions[object.id] : frame.positions[object.id];
      if (held === object.id) position = gripper;
      const isHeld = held === object.id || (!moving && frame.held === object.id);
      return <g key={object.id} transform={`translate(${position.x} ${position.y})`}>
        <title>{`${pretty(object.id)}${object.in ? ` · ${pretty(object.in)}` : " · tray unknown"}${isHeld ? " · held" : ""}`}</title>
        {(activeObject === object.id || isHeld) && <circle r="30" fill="#d4ecca25" stroke="#a8d79a" strokeWidth="2" strokeDasharray={isHeld ? undefined : "4 4"} />}
        <ObjectGlyph object={object} />
        <rect x="-37" y="25" width="74" height="16" rx="5" fill="#fffefae8" />
        <text y="36" textAnchor="middle" fill="#567162" fontSize="8">{pretty(object.id).length > 18 ? `${pretty(object.id).slice(0, 16)}…` : pretty(object.id)}</text>
      </g>;
    })}
    {result && <g transform={`translate(${gripper.x} ${gripper.y})`} opacity=".85"><circle r="32" fill="none" stroke="#93caaa" strokeWidth="2" /><path d="M-27 -10 V10 M27 -10 V10 M-33 0 H-20 M20 0 H33" fill="none" stroke="#a3d6b6" strokeWidth="5" strokeLinecap="round" /></g>}
    {!result && <text x="420" y="385" fill="#789274" textAnchor="middle" fontSize="15">Load a plan to see the scene from above.</text>}
    <text x="67" y="48" fill="#799178" fontSize="10" letterSpacing="1.5">TOP VIEW · TRAY MEMBERSHIP FROM SCENE GRAPH</text>
  </svg>;
}

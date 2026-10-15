"use client";
import { useEffect, useState } from "react";
import { Frame, elbow, motion, robotMotion3D, isSurface } from "@/lib/simulation";
import { Result, SceneObject, pretty } from "@/lib/types";

function color(object: SceneObject) {
  const value = `${object.colour || ""} ${object.type}`.toLowerCase();
  if (value.includes("red")) return "#ed7465";
  if (value.includes("yellow")) return "#edc759";
  if (value.includes("green") || value.includes("mug")) return "#53a998";
  if (/orange|mandarin/.test(value)) return "#f3a053";
  if (value.includes("blue")) return "#779ce2";
  if (value.includes("hand")) return "#dcac90";
  return "#aebdc4";
}

function ObjectShape({ object, x, y, active }: { object: SceneObject; x: number; y: number; active: boolean }) {
  const fill = color(object), label = object.type;
  return <g transform={`translate(${x} ${y})`}>
    <title>{pretty(object.id)}</title>
    {active && <ellipse cy="-10" rx="34" ry="30" fill="none" stroke="#158678" strokeWidth="1.5" strokeDasharray="4 4" />}
    {/gear/.test(label) ? <g transform="translate(0 -15)">
      {Array.from({ length: 10 }, (_, i) => <rect key={i} x="-5" y="-25" width="10" height="15" rx="2" fill={fill} transform={`rotate(${i * 36})`} />)}
      <circle r="20" fill={fill} /><circle r="7" fill="#f7f4ec" /><circle r="12" fill="none" stroke="#ffffff60" strokeWidth="2" />
    </g> : /mug|cup/.test(label) ? <g><path d="M13 -27 C37 -31 37 -2 13 -6" stroke={fill} strokeWidth="7" fill="none" /><rect x="-19" y="-36" width="36" height="36" rx="9" fill={fill} /><ellipse cx="-1" cy="-34" rx="17" ry="5" fill="#285c55" /><path d="M-12 -26 V-9" stroke="#ffffff50" strokeWidth="4" strokeLinecap="round" /></g>
    : /plate|saucer/.test(label) ? <g><ellipse cy="-5" rx="30" ry="8" fill="#dce4df" stroke="#99b5ac" strokeWidth="2" /><ellipse cy="-7" rx="19" ry="4" fill="#f5f6ef" /></g>
    : /banana/.test(label) ? <path d="M-23 -26 Q-8 1 25 -18 Q10 13 -14 -4 Z" fill="#e8c758" stroke="#c2a242" strokeWidth="2" />
    : /mandarin|fruit|orange/.test(label) ? <g><circle cy="-17" r="18" fill={fill} /><path d="M0 -34 Q8 -47 15 -40" stroke="#4b9e73" strokeWidth="4" fill="none" /></g>
    : /hand/.test(label) ? <g><rect x="-19" y="-30" width="38" height="30" rx="13" fill={fill} /><path d="M-14 -17 V-45 M-4 -17 V-49 M6 -17 V-46 M16 -15 V-36" stroke={fill} strokeWidth="8" strokeLinecap="round" /></g>
    : <g><rect x="-19" y="-36" width="38" height="36" rx="7" fill={fill} /><path d="M-12 -28 H8" stroke="#ffffff65" strokeWidth="4" strokeLinecap="round" /><path d="M13 -27 V-8" stroke="#00000012" strokeWidth="3" /></g>}
  </g>;
}

export default function RobotSimulator({ result, frame, previous, progress, animated, activeObject }: {
  result: Result | null; frame: Frame; previous: Frame; progress: number; animated: boolean; activeObject?: string;
}) {
  const [reducedMotion, setReducedMotion] = useState(false);
  useEffect(() => {
    const query = window.matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => setReducedMotion(query.matches);
    update(); query.addEventListener("change", update);
    return () => query.removeEventListener("change", update);
  }, []);
  const actionProgress = animated && !reducedMotion ? progress : 1;
  const moving = animated && !reducedMotion && actionProgress < 1;
  const pose = motion(previous, frame, moving ? actionProgress : 1);
  const { target, held } = pose;
  const joint = elbow(target);
  const yaw = robotMotion3D(previous, frame, actionProgress).yaw * 180 / Math.PI;
  const nodes = result?.objects.filter(o => !isSurface(o)) || [];
  const renderNodes = [...nodes].sort((a, b) => (frame.positions[b.id]?.y || 0) - (frame.positions[a.id]?.y || 0));
  return <svg viewBox="0 0 840 440" className="robot-svg" role="img" aria-label="Illustrative 2D robot arm and tabletop objects">
    <defs>
      <pattern id="dots" width="24" height="24" patternUnits="userSpaceOnUse"><circle cx="2" cy="2" r="1" fill="#c8d5c9" opacity=".55" /></pattern>
      <linearGradient id="arm" x1="0" y1="0" x2="1" y2="1"><stop stopColor="#a8d8b9" /><stop offset="1" stopColor="#5da68d" /></linearGradient>
      <linearGradient id="table" x1="0" y1="0" x2="0" y2="1"><stop stopColor="#e7ddc8" /><stop offset="1" stopColor="#cfc3a9" /></linearGradient>
      <filter id="shadow" x="-30%" y="-30%" width="160%" height="180%"><feDropShadow dx="0" dy="6" stdDeviation="5" floodColor="#2b5644" floodOpacity=".1" /></filter>
    </defs>
    <rect width="840" height="440" fill="url(#dots)" />
    <circle cx="640" cy="142" r="129" fill="#cbe5cd" opacity=".28" />
    <circle cx="640" cy="142" r="97" fill="none" stroke="#a8ccb4" strokeDasharray="5 8" opacity=".35" />
    <path d="M68 361 H780" stroke="#c1cabc" strokeWidth="1" />
    <g filter="url(#shadow)">
      <rect x="70" y="350" width="478" height="17" rx="6" fill="url(#table)" />
      <rect x="90" y="367" width="11" height="32" rx="3" fill="#bdbaa9" /><rect x="516" y="367" width="11" height="32" rx="3" fill="#bdbaa9" />
      <path d="M635 321 V350 H778 V321" fill="#dbe7dc" stroke="#9eb9a5" strokeWidth="8" strokeLinejoin="round" />
      <path d="M649 341 H764" stroke="#bfd2c2" strokeWidth="3" />
    </g>
    <text x="82" y="414" fill="#7c8c7d" fontSize="11" letterSpacing="2">WORKSPACE / GREY TRAY</text>
    <text x="655" y="384" fill="#7c8c7d" fontSize="11" letterSpacing="2">BLACK TRAY</text>
    {result?.answer.target_pixel && <g opacity=".45"><path d="M48 110 H68 M58 100 V120" stroke="#739989" strokeWidth="1.5" /><text x="78" y="114" fill="#638376" fontSize="11">TARGET [{result.answer.target_pixel.join(", ")}]</text></g>}
    {renderNodes.map(object => {
      let position = frame.positions[object.id];
      if (!position) return null;
      if (held === object.id) position = { x: target.x, y: target.y + 34 };
      else if (moving && previous.positions[object.id] && !(pose.released && pose.id === object.id)) position = previous.positions[object.id];
      return <ObjectShape key={object.id} object={object} x={position.x} y={position.y} active={object.id === activeObject} />;
    })}
    <g filter="url(#shadow)">
      <ellipse cx="585" cy="337" rx="38" ry="12" fill="#194e44" /><rect x="556" y="333" width="60" height="31" rx="7" fill="#266b5c" /><rect x="566" y="355" width="40" height="9" rx="3" fill="#194e44" />
      <path d={`M585 336 L${joint.x} ${joint.y} L${target.x} ${target.y}`} stroke="#367963" strokeWidth="28" strokeLinecap="round" strokeLinejoin="round" fill="none" />
      <path d={`M585 336 L${joint.x} ${joint.y} L${target.x} ${target.y}`} stroke="url(#arm)" strokeWidth="21" strokeLinecap="round" strokeLinejoin="round" fill="none" />
      {[{ x: 585, y: 336 }, joint, target].map((point, i) => <g key={i}><circle cx={point.x} cy={point.y} r={i === 2 ? 12 : 18} fill="#eaf2e2" stroke="#367963" strokeWidth="4" /><circle cx={point.x} cy={point.y} r={i === 2 ? 4 : 6} fill="#72a18a" /></g>)}
      <path d={`M${target.x} ${target.y + 10} V${target.y + 18} M${target.x - (held ? 17 : 23)} ${target.y + 18} H${target.x + (held ? 17 : 23)} M${target.x - (held ? 17 : 23)} ${target.y + 18} V${target.y + 34} M${target.x + (held ? 17 : 23)} ${target.y + 18} V${target.y + 34}`} fill="none" stroke="#326959" strokeWidth="5" strokeLinecap="round" />
    </g>
    <g transform="translate(585 345) scale(1 .38)" aria-label="Base joint rotation indicator">
      <circle r="48" fill="none" stroke="#367963" strokeWidth="2" opacity=".25" />
      <g transform={`rotate(${yaw})`}>
        <path d="M-34 -34 A48 48 0 1 1 -48 0" fill="none" stroke="#158678" strokeWidth="6" strokeLinecap="round" />
        <path d="M-59 -10 L-48 5 L-37 -10" fill="none" stroke="#158678" strokeWidth="6" strokeLinecap="round" strokeLinejoin="round" />
      </g>
    </g>
    <text x="560" y="395" fill="#597565" fontSize="10">BASE ROTATION {yaw.toFixed(0)}°</text>
    {!result && <g><text x="100" y="215" fill="#637f6e" fontSize="18">A little instruction.</text><text x="100" y="244" fill="#285d4d" fontSize="26" fontWeight="600">A little robot magic.</text></g>}
    <g transform="translate(710 45)"><rect width="86" height="25" rx="12" fill="#ffffff90" /><circle cx="14" cy="12" r="3" fill="#609680" /><text x="25" y="16" fill="#597565" fontSize="10" letterSpacing="1">2D VIEW</text></g>
  </svg>;
}

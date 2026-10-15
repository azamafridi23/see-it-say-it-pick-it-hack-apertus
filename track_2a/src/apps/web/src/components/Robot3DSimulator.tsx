"use client";

import { useEffect, useState } from "react";
import { Frame, isSurface, robotMotion3D, robotWorld } from "@/lib/simulation";
import { Result, pretty } from "@/lib/types";

type Vec = { x: number; y: number; z: number };

// Orthographic camera projects world geometry; dragging orbits the camera.
export default function Robot3DSimulator({ result, frame, previous, progress, animated }: {
  result: Result | null; frame: Frame; previous: Frame; progress: number; animated: boolean;
}) {
  const [angle, setAngle] = useState(-.45);
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    const query = matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => setReduced(query.matches);
    update(); query.addEventListener("change", update);
    return () => query.removeEventListener("change", update);
  }, []);
  const t = animated && !reduced ? progress : 1;
  const pose = robotMotion3D(previous, frame, t);
  const base: Vec = { x: 0, y: 0, z: 0 };
  const target = pose.target;
  // The pedestal yaws toward the target; both links bend in its vertical plane.
  const yaw = pose.yaw;
  const radius = Math.hypot(target.x, target.y);
  const length = 224;
  const distance = Math.min(length * 2 - 1, Math.hypot(radius, target.z));
  const bend = Math.atan2(target.z, radius) + Math.acos(distance / (2 * length));
  const joint: Vec = { x: Math.cos(yaw) * length * Math.cos(bend), y: Math.sin(yaw) * length * Math.cos(bend), z: length * Math.sin(bend) };
  function project(v: Vec) {
    const x = v.x * Math.cos(angle) - v.y * Math.sin(angle);
    const depth = v.x * Math.sin(angle) + v.y * Math.cos(angle);
    return { x: 500 + x, y: 335 + depth * .48 - v.z * .8 };
  }
  const points = (vertices: Vec[]) => vertices.map(v => { const p = project(v); return `${p.x},${p.y}`; }).join(" ");
  function box(id: string, center: Vec, width: number, depth: number, height: number, fill: string, rotation = 0) {
    const v = (x: number, y: number, z: number) => ({
      x: center.x + x * width / 2 * Math.cos(rotation) - y * depth / 2 * Math.sin(rotation),
      y: center.y + x * width / 2 * Math.sin(rotation) + y * depth / 2 * Math.cos(rotation), z: center.z + z,
    });
    return <g key={id}><polygon points={points([v(-1,-1,0),v(1,-1,0),v(1,-1,height),v(-1,-1,height)])} fill={fill} stroke="#315d5260" /><polygon points={points([v(1,-1,0),v(1,1,0),v(1,1,height),v(1,-1,height)])} fill={fill} stroke="#315d5260" style={{ filter: "brightness(.8)" }} /><polygon points={points([v(-1,-1,height),v(1,-1,height),v(1,1,height),v(-1,1,height)])} fill={fill} stroke="#315d5260" style={{ filter: "brightness(1.15)" }} /><title>{pretty(id)}</title></g>;
  }
  const b = project(base), j = project(joint), g = project(target);
  const objects = result?.objects.filter(o => !isSurface(o)) || [];
  return <div>
    <svg viewBox="0 0 840 440" className="robot-svg" role="img" aria-label="Animated 3D robot with rotating base and bending elbow. Drag to orbit camera." style={{ touchAction: "none", cursor: "grab" }} onPointerDown={e => e.currentTarget.setPointerCapture(e.pointerId)} onPointerMove={e => { if (e.currentTarget.hasPointerCapture(e.pointerId)) setAngle(a => a + e.movementX * .008); }} onPointerUp={e => e.currentTarget.releasePointerCapture(e.pointerId)}>
      <rect width="840" height="440" fill="#e8eee1" />
      {box("grey_tray", { x: -225, y: 0, z: -12 }, 380, 140, 12, "#bbc7c3")}
      {box("black_tray", { x: 98, y: 0, z: -12 }, 114, 140, 12, "#344b45")}
      {objects.map(object => {
        let p = t < 1 && !(pose.released && pose.id === object.id) ? previous.positions[object.id] : frame.positions[object.id];
        if (!p) return null;
        const v = pose.held === object.id ? { ...target, z: target.z - 34 * .8 } : robotWorld(p);
        const label = `${object.colour || ""} ${object.type}`;
        const fill = /red/.test(label) ? "#ed7465" : /yellow|banana/.test(label) ? "#edc759" : /orange|mandarin/.test(label) ? "#f3a053" : /mug|green/.test(label) ? "#53a998" : "#aebdc4";
        return box(object.id, v, 28, 28, /plate|gear/.test(label) ? 8 : 28, fill);
      })}
      {box("robot pedestal", base, 48, 48, 18, "#266b5c")}
      <ellipse cx={b.x} cy={b.y - 18} rx="27" ry="11" fill="#92c6aa" stroke="#367963" strokeWidth="4" />
      {box("rotating shoulder", { x: 0, y: 0, z: 18 }, 42, 22, 18, "#a8d8b9", yaw)}
      <line x1={b.x} y1={b.y - 18} x2={project({ x: Math.cos(yaw) * 40, y: Math.sin(yaw) * 40, z: 18 }).x} y2={project({ x: Math.cos(yaw) * 40, y: Math.sin(yaw) * 40, z: 18 }).y} stroke="#eaf2e2" strokeWidth="4" />
      <path d={`M${b.x} ${b.y-18} L${j.x} ${j.y} L${g.x} ${g.y}`} fill="none" stroke="#367963" strokeWidth="24" strokeLinecap="round" strokeLinejoin="round" />
      <path d={`M${b.x} ${b.y-18} L${j.x} ${j.y} L${g.x} ${g.y}`} fill="none" stroke="#9bcfb0" strokeWidth="15" strokeLinecap="round" strokeLinejoin="round" />
      {[j,g].map((p,i) => <circle key={i} cx={p.x} cy={p.y} r={i ? 10 : 15} fill="#eaf2e2" stroke="#367963" strokeWidth="4" />)}
      <path d={`M${g.x-18} ${g.y+10} V${g.y+25} M${g.x+18} ${g.y+10} V${g.y+25}`} stroke="#326959" strokeWidth="5" fill="none" />
      <text x="35" y="35" fill="#597565" fontSize="12">3D ROBOT · Drag to orbit</text>
      <text x="35" y="415" fill="#597565" fontSize="11">Base yaw {(yaw * 180 / Math.PI).toFixed(0)}° · Elbow bends to pick and place</text>
    </svg>
    <div style={{ display: "flex", gap: 8, justifyContent: "center" }}><button className="icon-button" aria-label="Rotate 3D camera left" onClick={() => setAngle(a => a - .2)}>↶</button><button className="icon-button" aria-label="Reset 3D camera" onClick={() => setAngle(-.45)}>Reset view</button><button className="icon-button" aria-label="Rotate 3D camera right" onClick={() => setAngle(a => a + .2)}>↷</button></div>
  </div>;
}

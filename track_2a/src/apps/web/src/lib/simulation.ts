import type { Result, SceneObject } from "./types";

export type Point = { x: number; y: number };
export type Frame = { positions: Record<string, Point>; held: string | null; gripper: Point; error?: string };
export const HOME: Point = { x: 365, y: 195 };
export function isSurface(object: SceneObject) { return /tray|mat/.test(object.id); }

export const TOP_TRAYS = {
  grey_tray: { x: 65, y: 75, width: 335, height: 245 },
  black_tray: { x: 440, y: 75, width: 335, height: 245 },
};
export const TOP_HOME: Point = { x: 420, y: 45 };

export function initialTopFrame(result: Result): Frame {
  const positions: Record<string, Point> = {};
  const [width, height] = result.image_size;
  let unknown = 0;
  for (const object of result.objects.filter(o => !isSurface(o))) {
    const tray = TOP_TRAYS[object.in as keyof typeof TOP_TRAYS];
    if (tray) {
      const u = object.pixel ? Math.max(0, Math.min(1, object.pixel[0] / width)) : .5;
      const v = object.pixel ? Math.max(0, Math.min(1, object.pixel[1] / height)) : .5;
      positions[object.id] = { x: tray.x + 35 + u * (tray.width - 70), y: tray.y + 40 + v * (tray.height - 80) };
    } else {
      positions[object.id] = { x: 95 + (unknown % 10) * 66, y: 375 + Math.floor(unknown++ / 10) * 45 };
    }
  }
  for (const [id, tray] of Object.entries(TOP_TRAYS)) positions[id] = { x: tray.x + tray.width / 2, y: tray.y + tray.height / 2 };
  // Known stacks share an XY position when viewed from above.
  const resolved = new Set<string>(), resolving = new Set<string>();
  function stack(object: SceneObject) {
    if (resolved.has(object.id) || resolving.has(object.id)) return;
    resolving.add(object.id);
    const support = result.objects.find(o => o.id === object.on?.[0] && positions[o.id]);
    if (support && !isSurface(support)) { stack(support); positions[object.id] = { ...positions[support.id] }; }
    resolving.delete(object.id); resolved.add(object.id);
  }
  result.objects.filter(o => !isSurface(o)).forEach(stack);
  return { positions, held: null, gripper: TOP_HOME };
}

export function initialFrame(result: Result): Frame {
  const positions: Record<string, Point> = {};
  const objects = result.objects.filter(o => !isSurface(o));
  const [width] = result.image_size;
  // Side view is a schematic projection, with a stable minimum spacing for readability.
  const sorted = [...objects].sort((a, b) => (a.pixel?.[0] ?? width) - (b.pixel?.[0] ?? width));
  sorted.forEach((object, index) => {
    const x = object.pixel ? 100 + (object.pixel[0] / width) * 420 : 120 + index * 62;
    positions[object.id] = { x: Math.max(85, Math.min(525, x)), y: 350 };
  });
  for (let i = 1; i < sorted.length; i++) {
    const prev = positions[sorted[i - 1].id], point = positions[sorted[i].id];
    if (point.x - prev.x < 47) point.x = prev.x + 47;
  }
  const lastX = sorted.length ? positions[sorted[sorted.length - 1].id].x : 0;
  if (lastX > 525) {
    const firstX = positions[sorted[0].id].x;
    sorted.forEach(object => { positions[object.id].x = 85 + (positions[object.id].x - firstX) / Math.max(1, lastX - firstX) * 440; });
  }
  // Preserve known support relations in live scene graphs.
  sorted.filter(object => object.in === "black_tray").forEach(object => {
    positions[object.id] = { x: 650 + (object.pixel ? Math.max(0, Math.min(1, object.pixel[0] / width)) : .5) * 110, y: 350 };
  });
  const resolving = new Set<string>();
  const resolved = new Set<string>();
  function stack(object: SceneObject) {
    if (resolved.has(object.id) || resolving.has(object.id)) return;
    resolving.add(object.id);
    const support = objects.find(o => o.id === object.on?.[0]);
    if (support) { stack(support); const base = positions[support.id]; positions[object.id] = { x: base.x, y: base.y - 34 }; }
    resolving.delete(object.id); resolved.add(object.id);
  }
  objects.forEach(stack);
  positions.grey_tray = { x: 205, y: 350 };
  positions.black_tray = { x: 705, y: 350 };
  return { positions, held: null, gripper: HOME };
}

export function frames(result: Result, view: "side" | "top" = "side"): Frame[] {
  const timeline = [view === "top" ? initialTopFrame(result) : initialFrame(result)];
  let placements = 0;
  for (const action of result.answer.actions) {
    const previous = timeline[timeline.length - 1];
    const next: Frame = { ...previous, positions: { ...previous.positions } };
    const oid = action.object;
    if (action.action === "pick") {
      if (!oid || !next.positions[oid]) next.error = `No position available for ${oid || "this object"}.`;
      else if (next.held) next.error = "The gripper is already holding an object.";
      else {
        next.held = oid;
        next.gripper = view === "top" ? { ...next.positions[oid] } : { x: next.positions[oid].x, y: 235 };
        next.positions[oid] = { x: next.gripper.x, y: next.gripper.y + (view === "top" ? 0 : 34) };
      }
    } else if (action.action === "place") {
      const destination = oid ? next.positions[oid] : undefined;
      if (!next.held || !destination) next.error = "This placement needs a held object and a known destination.";
      else {
        const surface = /tray|mat/.test(oid!);
        const offset = surface ? ((placements++ % 3) - 1) * 45 : 0;
        let position = { x: destination.x + offset, y: surface ? 350 : destination.y - 34 };
        if (view === "top") {
          position = { ...destination };
          const tray = TOP_TRAYS[oid as keyof typeof TOP_TRAYS];
          if (tray) {
            const slots = Array.from({ length: 12 }, (_, i) => ({ x: tray.x + 55 + (i % 4) * 75, y: tray.y + 60 + Math.floor(i / 4) * 65 }));
            position = slots.find(slot => Object.entries(next.positions).every(([id, point]) => /tray|mat/.test(id) || id === next.held || Math.hypot(slot.x - point.x, slot.y - point.y) > 42)) || slots[(placements - 1) % slots.length];
          }
        }
        next.positions[next.held] = position;
        next.held = null;
        next.gripper = view === "top" ? { ...position } : { x: position.x, y: 235 };
      }
    }
    timeline.push(next);
    if (next.error || action.action === "refuse" || action.action === "ask_user") break;
  }
  return timeline;
}

export function interpolate(a: Point, b: Point, progress: number): Point {
  return { x: a.x + (b.x - a.x) * progress, y: a.y + (b.y - a.y) * progress };
}

// Choose the upper elbow branch on both sides of the rotating pedestal.
export function elbow(target: Point): Point {
  const base = { x: 585, y: 336 }, length = 280;
  const dx = target.x - base.x, dy = target.y - base.y;
  const distance = Math.min(length * 2 - 1, Math.max(1, Math.hypot(dx, dy)));
  const angle = Math.atan2(dy, dx) + (dx < 0 ? 1 : -1) * Math.acos(distance / (2 * length));
  return { x: base.x + length * Math.cos(angle), y: base.y + length * Math.sin(angle) };
}

export function motion(previous: Frame, frame: Frame, progress: number) {
  const t = Math.max(0, Math.min(1, progress));
  const picking = !previous.held && !!frame.held;
  const placing = !!previous.held && !frame.held;
  const id = picking ? frame.held : placing ? previous.held : null;
  let target = interpolate(previous.gripper, frame.gripper, t);
  let held = previous.held;
  if (id && (picking || placing)) {
    const point = picking ? previous.positions[id] : frame.positions[id];
    const contact = { x: point.x, y: point.y - 34 };
    const above = { x: point.x, y: Math.min(220, contact.y - 70) };
    target = t < .35 ? interpolate(previous.gripper, above, t / .35)
      : t < .65 ? interpolate(above, contact, (t - .35) / .3)
      : interpolate(contact, frame.gripper, (t - .65) / .35);
    if (t >= .65) held = frame.held;
  }
  if (t === 1) held = frame.held;
  return { target, held, picking, placing, id, released: placing && t >= .65 };
}

export type WorldPoint = { x: number; y: number; z: number };
export function robotWorld(point: Point): WorldPoint {
  return { x: (point.x - 585) * .8, y: 35, z: (350 - point.y) * .8 };
}

// Interpolate around the vertical axis, rather than crossing through the base.
// Angles can span the full circle; take the shortest turn across the 0/360 seam.
export function rotateTravel(from: WorldPoint, to: WorldPoint, progress: number): WorldPoint {
  const t = Math.max(0, Math.min(1, progress));
  const start = Math.atan2(from.y, from.x), end = Math.atan2(to.y, to.x);
  const turn = Math.atan2(Math.sin(end - start), Math.cos(end - start));
  const yaw = start + turn * t;
  const radius = Math.hypot(from.x, from.y) + (Math.hypot(to.x, to.y) - Math.hypot(from.x, from.y)) * t;
  return { x: Math.cos(yaw) * radius, y: Math.sin(yaw) * radius, z: from.z + (to.z - from.z) * t };
}

export function robotMotion3D(previous: Frame, frame: Frame, progress: number) {
  const t = Math.max(0, Math.min(1, progress));
  const pose = motion(previous, frame, t);
  let target = robotWorld(pose.target);
  if (pose.id && t < .35) {
    const point = pose.picking ? previous.positions[pose.id] : frame.positions[pose.id];
    const above = { x: point.x, y: Math.min(220, point.y - 104) };
    const travel = t / .35;
    const eased = travel * travel * (3 - 2 * travel);
    target = rotateTravel(robotWorld(previous.gripper), robotWorld(above), eased);
  }
  return { ...pose, target, yaw: Math.atan2(target.y, target.x) };
}

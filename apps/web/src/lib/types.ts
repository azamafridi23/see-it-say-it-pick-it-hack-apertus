export type Pipeline = "path_a_graph" | "path_a" | "path_b";
export type Mode = "replay" | "live";
export type Action = { action: "pick" | "place" | "ask_user" | "refuse"; object?: string; pixel?: [number, number]; question?: string; reason?: string };
export type SceneObject = { id: string; type: string; pixel?: [number, number] | null; colour?: string; on?: string[]; in?: string | null; pickable?: boolean };
export type Instruction = { id: string; text: string; replays: Pipeline[] };
export type Scene = { id: string; image_size: [number, number]; instructions: Instruction[]; instruction_file: string | null };
export type Result = {
  scene_id: string; pipeline: Pipeline; mode: Mode; instruction: string;
  answer: { instruction_id: string; target_pixel: [number, number] | null; actions: Action[] };
  objects: SceneObject[]; image_size: [number, number]; geometry_source: "saved_pixels" | "pipeline_graph";
  elapsed_seconds: number | null;
};
export const PIPELINES: { id: Pipeline; label: string; detail: string }[] = [
  { id: "path_a_graph", label: "Path A · Multi-agent", detail: "SAM 3 + Apertus agents" },
  { id: "path_a", label: "Path A · Single planner", detail: "SAM 3 + Apertus planner" },
  { id: "path_b", label: "Path B · Apertus vision", detail: "Apertus + OpenCV" },
];
export const pretty = (id: string) => id.replace(/_/g, " ");

import { spawn } from "node:child_process";
import { existsSync } from "node:fs";

// The venv is in the project root (track_2a/), one folder above src/.
const python = process.env.SIMULATOR_PYTHON
  || [".venv/bin/python", "../.venv/bin/python"].find(path => existsSync(path))
  || "python3";
const processGroups = process.platform !== "win32";
const children = [
  spawn(python, ["-m", "uvicorn", "apps.api.main:app", "--host", "127.0.0.1", "--port", "8000"], { stdio: "inherit", detached: processGroups }),
  spawn("npm", ["run", "dev:web"], { stdio: "inherit", detached: processGroups }),
];
let stopping = false;
function stop(code = 0) {
  if (stopping) return;
  stopping = true;
  children.forEach(child => {
    try {
      if (processGroups && child.pid) process.kill(-child.pid, "SIGTERM");
      else child.kill("SIGTERM");
    } catch (error) { if (error.code !== "ESRCH") console.error(error.message); }
  });
  process.exitCode = code;
}
for (const child of children) {
  child.on("error", error => { console.error(error.message); stop(1); });
  child.on("exit", code => { if (!stopping) stop(code || 0); });
}
process.on("SIGINT", () => stop());
process.on("SIGTERM", () => stop());

export async function api<T>(url: string, options?: RequestInit): Promise<T> {
  let response: Response;
  try { response = await fetch(`/api${url}`, options); }
  catch { throw new Error("Cannot reach the local backend. Start both services with npm run dev."); }
  if (!response.ok) {
    let detail = "The backend could not complete this request. Check that the Python service is running.";
    try { const error = await response.json(); if (typeof error.detail === "string") detail = error.detail; } catch { /* Non-JSON proxy errors */ }
    throw new Error(detail);
  }
  return response.json();
}

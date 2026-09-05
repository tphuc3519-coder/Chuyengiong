import { NextResponse } from "next/server";

import { modalApiUrlOrNull } from "@/lib/server";

/**
 * Runtime configuration for the browser: where the Modal API lives, and what
 * that deployment can do.
 *
 * The upload cannot go through this app — a 3 minute mp3 is 4–7 MB and the
 * serverless request body limit is 4.5 MB (plan §6) — so the client has to know
 * the Modal URL. Serving it here rather than baking it into the bundle keeps it
 * an environment variable, which is what the plan asks for.
 *
 * `beatGenerator` is probed here, server side, rather than by the browser. The
 * browser asking meant a cross-origin `GET /health`, and that request has two
 * ways to fail — CORS and the network — that both look exactly like a
 * deployment without the generator. One silent false is how the feature came to
 * be "deployed and invisible": the flag was on, the API said so, and the page
 * had no way to tell a refused request from an honest no. From here it is a
 * same-process fetch with no preflight and no browser cache in the way.
 */
export const dynamic = "force-dynamic";

// Long enough for a cold Modal container to answer, short enough that the form
// is not held hostage by it: the page cannot upload before this reply lands.
const HEALTH_TIMEOUT_MS = 4000;

async function beatGenerator(apiBase: string): Promise<boolean> {
  try {
    const response = await fetch(`${apiBase}/health`, {
      cache: "no-store",
      signal: AbortSignal.timeout(HEALTH_TIMEOUT_MS),
    });
    if (!response.ok) return false;
    const body = await response.json();
    return Boolean(body?.beat_generator);
  } catch {
    // Not knowing means not offering. The upload source still works, and the
    // note under it says why the choice is missing.
    return false;
  }
}

export async function GET() {
  const apiBase = modalApiUrlOrNull();
  if (!apiBase) {
    return NextResponse.json({ error: "MODAL_API_URL is not configured" }, { status: 503 });
  }
  return NextResponse.json(
    { apiBase, beatGenerator: await beatGenerator(apiBase) },
    { headers: { "cache-control": "no-store" } },
  );
}

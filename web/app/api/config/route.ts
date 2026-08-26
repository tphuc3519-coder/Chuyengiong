import { NextResponse } from "next/server";

import { modalApiUrlOrNull } from "@/lib/server";

/**
 * Runtime configuration for the browser: where the Modal API lives.
 *
 * The upload cannot go through this app — a 3 minute mp3 is 4–7 MB and the
 * serverless request body limit is 4.5 MB (plan §6) — so the client has to know
 * the Modal URL. Serving it here rather than baking it into the bundle keeps it
 * an environment variable, which is what the plan asks for.
 */
export const dynamic = "force-dynamic";

export async function GET() {
  const apiBase = modalApiUrlOrNull();
  if (!apiBase) {
    return NextResponse.json({ error: "MODAL_API_URL is not configured" }, { status: 503 });
  }
  return NextResponse.json({ apiBase }, { headers: { "cache-control": "no-store" } });
}

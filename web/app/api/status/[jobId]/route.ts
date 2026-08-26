import { NextResponse } from "next/server";

import { modalApiUrlOrNull } from "@/lib/server";

/**
 * Status polling, proxied.
 *
 * Uploads and downloads go straight to Modal because of their size; a status
 * record is a few hundred bytes, so it comes through here — which keeps the
 * poll loop on the same origin as the page and gives one place to add caching
 * or a circuit breaker later.
 */
export const dynamic = "force-dynamic";

const JOB_ID = /^[0-9a-f]{32}$/;

export async function GET(_request: Request, context: { params: Promise<{ jobId: string }> }) {
  const { jobId } = await context.params;
  if (!JOB_ID.test(jobId)) {
    return NextResponse.json({ error: "no such job" }, { status: 404 });
  }

  const apiBase = modalApiUrlOrNull();
  if (!apiBase) {
    return NextResponse.json({ error: "MODAL_API_URL is not configured" }, { status: 503 });
  }

  try {
    const upstream = await fetch(`${apiBase}/status/${jobId}`, { cache: "no-store" });
    const body = await upstream.text();
    return new NextResponse(body, {
      status: upstream.status,
      headers: {
        "content-type": upstream.headers.get("content-type") ?? "application/json",
        "cache-control": "no-store",
      },
    });
  } catch {
    // A failed poll is not a failed job: the client keeps polling and this
    // shows up as a transient error rather than ending the run.
    return NextResponse.json({ error: "could not reach the converter" }, { status: 502 });
  }
}

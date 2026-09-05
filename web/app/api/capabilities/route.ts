import { NextResponse } from "next/server";

import { modalApiUrlOrNull } from "@/lib/server";

/**
 * What the Modal deployment behind this app can actually do.
 *
 * Asked here, server side, rather than by the browser. The browser asking meant
 * a cross-origin `GET ${apiBase}/health`, and that request has two ways to fail
 * — CORS and the network — that both land in the same `catch` as an honest "no
 * generator here". One silent `false` is how the feature came to be deployed
 * and invisible: the flag was on, the API said `beat_generator: true`, and the
 * page had no way to tell a refused request from a real no. From this side it
 * is a same-origin request for the browser and a plain server-to-server fetch
 * for us, with no preflight and no browser cache in the way.
 *
 * Separate from `/api/config` on purpose. Config is on the upload's critical
 * path; this is not — nothing waits on it except the beat-source control, which
 * simply appears when the answer arrives. That is what buys the generous
 * deadline below.
 */
export const dynamic = "force-dynamic";

// A Modal container that has scaled to zero answers this in tens of seconds,
// not milliseconds, and `api()` sets no `min_containers`. The old browser-side
// probe had no deadline at all and was right not to: giving up early on a cold
// start would hide the generator on exactly the first page load of the day.
const HEALTH_TIMEOUT_MS = 12_000;
const HEALTH_ATTEMPTS = 2;

async function probe(apiBase: string): Promise<boolean> {
  for (let attempt = 1; attempt <= HEALTH_ATTEMPTS; attempt++) {
    try {
      const response = await fetch(`${apiBase}/health`, {
        cache: "no-store",
        signal: AbortSignal.timeout(HEALTH_TIMEOUT_MS),
      });
      // A reply is an answer. Only a request that never arrived is worth
      // repeating — and the second attempt is the one a cold start needs,
      // since the first is what woke the container up.
      if (!response.ok) return false;
      const body = await response.json();
      return Boolean(body?.beat_generator);
    } catch {
      if (attempt === HEALTH_ATTEMPTS) break;
    }
  }
  return false;
}

export async function GET() {
  const apiBase = modalApiUrlOrNull();
  // Not knowing means not offering. This route has no failure mode the browser
  // could act on, so it has no error status either — the upload source works
  // without a generator, and the note under it says why the choice is missing.
  const beatGenerator = apiBase ? await probe(apiBase) : false;
  return NextResponse.json({ beatGenerator }, { headers: { "cache-control": "no-store" } });
}

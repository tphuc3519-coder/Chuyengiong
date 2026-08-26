import "server-only";

/**
 * The Modal base URL, read from the environment at request time.
 *
 * Not `NEXT_PUBLIC_`: that would inline the value into the client bundle at
 * build time, which the plan rules out (§6) and which would tie one build to
 * one deployment. The browser gets the URL from `/api/config` instead, so the
 * same build runs against preview and production.
 */
export function modalApiUrl(): string {
  const url = process.env.MODAL_API_URL?.trim();
  if (!url) {
    throw new Error("MODAL_API_URL is not set — see web/.env.example");
  }
  return url.replace(/\/+$/, "");
}

/** `modalApiUrl` without the throw, for routes that would rather 503. */
export function modalApiUrlOrNull(): string | null {
  try {
    return modalApiUrl();
  } catch {
    return null;
  }
}

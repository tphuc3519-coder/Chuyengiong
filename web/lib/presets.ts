/**
 * Preset reference voices.
 *
 * Plan §6 wants 4–6 ready-made voices so a first-time visitor can hear a result
 * without recording or uploading anything. What it does **not** allow (§8 item
 * 4) is a celebrity voice or anything user-contributed, so the audio has to be
 * self-recorded or come from a dataset with an explicit licence — see
 * `public/presets/README.md`.
 *
 * The manifest ships empty on purpose: no audio, no preset row in the UI. That
 * is better than shipping a voice whose provenance nobody can point to.
 */

export type Preset = {
  id: string;
  name: string;
  /** Path under `public/presets/`, e.g. `alto-vi.mp3`. */
  file: string;
  note?: string;
  /** Where the recording came from. Required — see §8 item 4. */
  license: string;
};

const MANIFEST = "/presets/index.json";

export async function loadPresets(signal?: AbortSignal): Promise<Preset[]> {
  try {
    const response = await fetch(MANIFEST, { signal });
    if (!response.ok) return [];
    const parsed: unknown = await response.json();
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (item): item is Preset =>
        typeof item === "object" &&
        item !== null &&
        typeof (item as Preset).id === "string" &&
        typeof (item as Preset).name === "string" &&
        typeof (item as Preset).file === "string" &&
        typeof (item as Preset).license === "string",
    );
  } catch {
    return [];
  }
}

export async function fetchPreset(preset: Preset, signal?: AbortSignal): Promise<File> {
  const response = await fetch(`/presets/${preset.file}`, { signal });
  if (!response.ok) throw new Error(`could not load the preset voice "${preset.name}"`);
  const blob = await response.blob();
  return new File([blob], preset.file, { type: blob.type || "audio/mpeg" });
}

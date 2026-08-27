/**
 * The knobs `/submit` accepts, mirrored from the backend.
 *
 * These bounds exist twice on purpose: the backend clamps because a client
 * cannot be trusted, this copy exists so a slider cannot offer a value that
 * will be silently changed. `modal_app/audio_utils.py` is the source of truth —
 * when a number moves there, move it here too.
 */

export type Mode = "song" | "speech";

export const MODES: { id: Mode; label: string; hint: string }[] = [
  { id: "song", label: "Bài hát", hint: "Tách nhạc nền, đổi giọng, ghép lại" },
  { id: "speech", label: "Giọng nói", hint: "Đổi giọng trực tiếp, nhanh hơn" },
];

/** Plan §3: ±8 for speech, because a big shift on a tonal language distorts it. */
export const MAX_SEMITONE_SHIFT: Record<Mode, number> = { song: 12, speech: 8 };

export const DIFFUSION_STEPS_MIN = 10;
export const DIFFUSION_STEPS_MAX = 100;
export const DEFAULT_DIFFUSION_STEPS: Record<Mode, number> = { song: 50, speech: 25 };

export const MAX_VOCAL_GAIN_DB = 12;

/** Plan §9, enforced in `modal_app/ratelimit.py`. Shown so the wall is visible. */
export const JOBS_PER_HOUR = 5;

/** Reference voice window, enforced again in `audio_utils.prepare_reference`. */
export const REFERENCE_MIN_SEC = 5;
export const REFERENCE_MAX_SEC = 20;
/** What the recorder counts down to. Past this it is trimmed anyway. */
export const RECORD_TARGET_SEC = 15;

export const MAX_INPUT_BYTES = 60 * 1024 * 1024;
export const MAX_REFERENCE_BYTES = 20 * 1024 * 1024;
export const SOURCE_MAX_SEC = 15 * 60;

/**
 * What the file pickers offer, mirrored from `ALLOWED_EXTS` in
 * `modal_app/separation.py`.
 *
 * The extensions are spelled out rather than left to `audio/*` because iOS
 * Safari resolves the wildcard against its own file types and greys out
 * everything it fails to map: an mp3 sitting in Files cannot be picked while an
 * mp4 goes through, which reads as "the app refuses music". Offering the
 * extension as well gets those files back, and costs nothing elsewhere — a
 * picker matches the union of what `accept` lists. `video/mp4` is deliberate:
 * the backend takes an mp4 and lets ffmpeg pull the audio out of it.
 */
export const AUDIO_ACCEPT = [
  "audio/*",
  "video/mp4",
  ".mp3",
  ".wav",
  ".m4a",
  ".mp4",
  ".flac",
  ".ogg",
  ".opus",
  ".aac",
  ".wma",
].join(",");

export type Params = {
  /**
   * null = auto-detect (plan §7). Not the same as 0, which is a deliberate
   * "leave the pitch where it is" — the backend distinguishes the two and
   * measures the vocal stem only for null.
   */
  semitoneShift: number | null;
  diffusionSteps: number;
  vocalGainDb: number;
};

export function defaultParams(mode: Mode): Params {
  return { semitoneShift: null, diffusionSteps: DEFAULT_DIFFUSION_STEPS[mode], vocalGainDb: 0 };
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

/**
 * Carry the tuning across a mode switch.
 *
 * Pitch is clamped, because `speech` allows a narrower range. Quality is
 * different: `song` defaults to 50 steps and `speech` to 25, so a value the
 * user never touched has to follow the mode — otherwise picking "Giọng nói"
 * silently costs twice the GPU time it should. A value they did move is theirs
 * and stays.
 */
export function forMode(params: Params, next: Mode, previous: Mode): Params {
  const limit = MAX_SEMITONE_SHIFT[next];
  const untouched = params.diffusionSteps === DEFAULT_DIFFUSION_STEPS[previous];
  return {
    ...params,
    semitoneShift:
      params.semitoneShift === null ? null : clamp(params.semitoneShift, -limit, limit),
    diffusionSteps: untouched ? DEFAULT_DIFFUSION_STEPS[next] : params.diffusionSteps,
  };
}

/** `+3` / `−2` / `0`, with a real minus sign. */
export function formatSemitones(shift: number): string {
  if (shift === 0) return "0";
  return shift > 0 ? `+${shift}` : `−${Math.abs(shift)}`;
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function formatSeconds(seconds: number): string {
  const whole = Math.max(0, Math.round(seconds));
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

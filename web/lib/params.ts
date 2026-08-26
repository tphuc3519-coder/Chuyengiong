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

/** Reference voice window, enforced again in `audio_utils.prepare_reference`. */
export const REFERENCE_MIN_SEC = 5;
export const REFERENCE_MAX_SEC = 20;
/** What the recorder counts down to. Past this it is trimmed anyway. */
export const RECORD_TARGET_SEC = 15;

export const MAX_INPUT_BYTES = 60 * 1024 * 1024;
export const MAX_REFERENCE_BYTES = 20 * 1024 * 1024;
export const SOURCE_MAX_SEC = 15 * 60;

export type Params = {
  semitoneShift: number;
  diffusionSteps: number;
  vocalGainDb: number;
};

export function defaultParams(mode: Mode): Params {
  return { semitoneShift: 0, diffusionSteps: DEFAULT_DIFFUSION_STEPS[mode], vocalGainDb: 0 };
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

/** Re-clamp after a mode switch: `speech` allows a narrower pitch range. */
export function forMode(params: Params, mode: Mode): Params {
  const limit = MAX_SEMITONE_SHIFT[mode];
  return { ...params, semitoneShift: clamp(params.semitoneShift, -limit, limit) };
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

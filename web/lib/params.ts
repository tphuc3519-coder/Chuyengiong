/**
 * The knobs `/submit` accepts, mirrored from the backend.
 *
 * These bounds exist twice on purpose: the backend clamps because a client
 * cannot be trusted, this copy exists so a slider cannot offer a value that
 * will be silently changed. `modal_app/audio_utils.py` is the source of truth —
 * when a number moves there, move it here too.
 */

export type Mode = "song" | "beat" | "rebeat" | "vocal" | "speech" | "tts";

/**
 * The modes that convert a voice, and therefore need a reference.
 *
 * `rebeat` is the only one that does not — it keeps the singer it was given —
 * which is why it has no voice sample step, no pitch slider and no voice
 * profile. The backend refuses a reference sent to it rather than ignoring one.
 */
export function convertsVoice(mode: Mode): boolean {
  return mode !== "rebeat";
}

/**
 * `beat` is `song` with the backing track replaced rather than kept: same
 * separation, same conversion, and then the original instrumental is measured
 * for tempo and key and thrown away. What goes back under the voice is a beat
 * the user uploaded, one generated from a description, or the song's own
 * chords rebuilt from scratch.
 *
 * `rebeat` is that without the conversion — the singer is left exactly as they
 * were. It exists as a mode of its own because bundling it into `beat` made
 * changing a backing track cost a voice sample, a consent question about
 * somebody's voice and a second GPU pass, none of which somebody who only
 * wants a different beat should be asked for.
 *
 * `vocal` is `song` without the separator, and that is the whole difference.
 *
 * Separation is not free in either direction: it costs a GPU pass, and what it
 * hands the converter is a stem carrying the artefacts every source separator
 * leaves behind — smeared transients, a faint ghost of the backing track —
 * which then get converted along with the voice. For a file that is already
 * just a voice, both costs are paid for nothing, so this branch skips straight
 * to the conversion and returns it with no mix.
 */
export const MODES: { id: Mode; label: string; hint: string }[] = [
  { id: "song", label: "Bài hát", hint: "Tách nhạc nền, đổi giọng, ghép lại" },
  { id: "rebeat", label: "Đổi beat", hint: "Giữ nguyên giọng gốc, chỉ thay nhạc nền" },
  { id: "beat", label: "Đổi beat + giọng", hint: "Vừa thay nhạc nền vừa đổi sang giọng mẫu" },
  { id: "vocal", label: "Giọng hát", hint: "File đã tách sẵn — đổi giọng, giữ nguyên" },
  { id: "speech", label: "Giọng nói", hint: "Đổi giọng trực tiếp, nhanh hơn" },
  { id: "tts", label: "Văn bản", hint: "Gõ chữ, đọc lên bằng giọng mẫu" },
];

/**
 * Plan §3: ±8 for speech, because a big shift on a tonal language distorts it.
 * `tts` is speech by the time the pitch is applied — the text has already been
 * read out loud — so it lives under the same limit.
 */
export const MAX_SEMITONE_SHIFT: Record<Mode, number> = {
  song: 12,
  beat: 12,
  // Never read — `rebeat` converts nothing — but the record has to be total.
  rebeat: 12,
  vocal: 12,
  speech: 8,
  tts: 8,
};

export const DIFFUSION_STEPS_MIN = 10;
export const DIFFUSION_STEPS_MAX = 100;
export const DEFAULT_DIFFUSION_STEPS: Record<Mode, number> = {
  song: 50,
  // Same checkpoint as a song: it is a song, with a different bed under it.
  beat: 50,
  rebeat: 50,
  // A vocal take converts with the singing checkpoint, so it wants the same
  // number of steps a song does.
  vocal: 50,
  speech: 25,
  tts: 25,
};

/**
 * How much of the sample voice to take, mirrored from `CFG_RATE_*` in
 * `modal_app/audio_utils.py`.
 *
 * Classifier-free guidance: the balance between what the model predicts having
 * seen the reference and what it would have predicted without it. Up, and the
 * result is more clearly the target and more obviously processed — a diffusion
 * model's artefacts are conditioning artefacts, so pushing the conditioning
 * harder pushes them too. Down, and more of whoever is on the source recording
 * survives.
 */
export const CFG_RATE_MIN = 0;
export const CFG_RATE_MAX = 1;
export const DEFAULT_CFG_RATE = 0.7;

/**
 * How much of the clarity chain runs on the result, from `modal_app/enhance.py`.
 *
 * 0 is not "less" — it is no filter at all, which is the output this app
 * produced before the chain existed. That is deliberate: every complaint about
 * post-processing should have a one-slider answer.
 */
export const CLARITY_MIN = 0;
export const CLARITY_MAX = 1;
export const DEFAULT_CLARITY = 0.5;

export const MAX_VOCAL_GAIN_DB = 12;

/**
 * The `beat` branch, mirrored from `modal_app/beatgen.py` and
 * `modal_app/pipeline.py`.
 *
 * Three genuinely different things, named rather than inferred from which
 * field happened to be filled in — `remake` sends neither a file nor a
 * description, so there would be nothing to infer it from.
 *
 * `upload` — a file the user already has the right to use. No GPU, no model,
 * and the licence of what comes out is the licence they came with.
 *
 * `generate` — music invented from a description. **Off on most deployments**
 * — see `BEAT_GENERATOR_ENABLED`. Nothing to do with the
 * original song, which is the point and also the limit: it cannot know the
 * song's chord progression. The model will not land on the BPM it is asked
 * for either, and does not need to — the result is measured and fitted
 * afterwards (`modal_app/beats.py`), so the prompt only has to get the
 * *character* right.
 *
 * `remake` — the song's own harmony, at its own tempo, played on instruments
 * synthesised from scratch. The only one that is still the same song, and the
 * only one whose copyright story has to be said out loud: it removes the sound
 * recording and not the composition, so what comes out is a cover. Covers are
 * licensable, cheaply and often compulsorily; masters usually are not. That is
 * the real gain and it is narrower than "tránh bản quyền" suggests.
 */
export type BeatSource = "upload" | "generate" | "remake";

/**
 * How a remade backing track is played, mirrored from `STYLES` in
 * `modal_app/arrange.py`.
 *
 * `auto` picks from the measured tempo, which is a better guess than any fixed
 * default: 72 BPM wants a ballad and 150 does not.
 */
export const ARRANGE_STYLES: { id: string; label: string; hint: string }[] = [
  { id: "auto", label: "Tự chọn", hint: "Theo tốc độ đo được của bài" },
  { id: "ballad", label: "Ballad", hint: "Chậm, hợp âm ngân dài" },
  { id: "lofi", label: "Lo-fi", hint: "Trống lệch nhịp, pad ấm" },
  { id: "boombap", label: "Boom bap", hint: "Trống mộc, hợp âm nảy" },
  { id: "pop", label: "Pop", hint: "Đều nhịp, đầy đặn" },
  { id: "trap", label: "Trap", hint: "808, hi-hat dày" },
];
export const DEFAULT_ARRANGE_STYLE = "auto";

/**
 * Whether this deployment ships the beat generator, mirrored from
 * `BEAT_GENERATOR` in `modal_app/beatgen.py`.
 *
 * Off, because its container image does not build yet — and `modal deploy`
 * builds every registered image in one pass, so a broken one fails the whole
 * deploy rather than its own function. Until somebody has watched that image
 * build, the backend refuses the source with a 400 and this hides the option
 * rather than offering a button that cannot work.
 *
 * Flip both when it is fixed: this, and `BEAT_GENERATOR=1` on the deployment.
 */
export const BEAT_GENERATOR_ENABLED = false;

export const BEAT_PROMPT_CHARS = 300;
export const BEAT_PROMPT_EXAMPLES = [
  "boom bap, 90 BPM, piano buồn, trống mộc",
  "lo-fi hip hop, 85 BPM, guitar sạch, tiếng vinyl",
  "trap, 140 BPM, 808 nặng, hi-hat nhanh",
  "house, 124 BPM, bass tròn, synth ấm",
];

/** -1 là mỗi lần một beat khác; số cố định lấy lại đúng beat cũ. */
export const BEAT_RANDOM_SEED = -1;

/**
 * The ceiling this labels the counter with — plan §9's number.
 *
 * Only ever rendered when the server reports one, which it does not unless
 * `JOBS_PER_HOUR` is set on the deployment: `ratelimit.remaining` returns null
 * with no cap configured, and null hides the line rather than quoting a wall
 * that is not there. Set the deployment to something other than 5 and this has
 * to move with it.
 */
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

/**
 * The `tts` branch, mirrored from `modal_app/tts.py`.
 *
 * Most of these are MMS-TTS checkpoints that read their language as it is
 * written. Everything outside a Latin script would need the text romanised
 * first, and a checkpoint handed unromanised text returns silence rather than
 * an error — so the backend refuses what is not on this list, and this offers
 * nothing it would refuse.
 *
 * Japanese is read by a different engine for exactly that reason: the
 * romaniser MMS was trained with turns kanji into Mandarin. See the module
 * docstring in `modal_app/tts.py` — it carries the measurements.
 *
 * `maxChars` differs per language because a character is not a unit of speech.
 * 2000 characters of Vietnamese and 700 of Japanese are the same two or three
 * minutes of audio, and the cap is there to bound the recording.
 */
export const LANGUAGES: { id: string; label: string; maxChars: number }[] = [
  { id: "vie", label: "Tiếng Việt", maxChars: 2000 },
  { id: "eng", label: "English", maxChars: 2000 },
  { id: "jpn", label: "日本語", maxChars: 700 },
  { id: "ind", label: "Bahasa Indonesia", maxChars: 2000 },
  { id: "fra", label: "Français", maxChars: 2000 },
  { id: "spa", label: "Español", maxChars: 2000 },
  { id: "deu", label: "Deutsch", maxChars: 2000 },
  { id: "por", label: "Português", maxChars: 2000 },
  { id: "ita", label: "Italiano", maxChars: 2000 },
];
export const DEFAULT_LANGUAGE = "vie";

/** The ceiling, for a language that somehow is not on the list. */
export const MAX_TEXT_CHARS = 2000;

/** ~2 to 3 minutes of speech, in whatever the language spends per character. */
export function maxCharsFor(language: string): number {
  return LANGUAGES.find((item) => item.id === language)?.maxChars ?? MAX_TEXT_CHARS;
}

export const SPEAKING_RATE_MIN = 0.5;
export const SPEAKING_RATE_MAX = 2;
export const DEFAULT_SPEAKING_RATE = 1;

/**
 * How the text is read, mirrored from `modal_app/prosody.py`.
 *
 * Neither synthesiser has an emotion input — both speak in one fixed voice —
 * so a style here is not a label handed to a model. It is the five things the
 * acoustic literature says actually differ between deliveries (pace, height,
 * range, level, and how long the silences are) applied to the reading from the
 * outside, per sentence. The backend falls back to the natural one for a name
 * it does not know, so this list can never break a job; it just has to stay
 * honest about what is on offer.
 */
export const EMOTIONS: { id: string; label: string; hint: string }[] = [
  { id: "natural", label: "Tự nhiên", hint: "Đọc bình thường, đúng nhịp câu chữ" },
  { id: "warm", label: "Ấm áp", hint: "Chậm và mềm hơn, hợp kể chuyện" },
  { id: "cheerful", label: "Vui vẻ", hint: "Nhanh hơn, cao hơn, lên xuống rõ hơn" },
  { id: "sad", label: "Trầm buồn", hint: "Chậm, thấp, ngắt nghỉ dài hơn" },
  { id: "serious", label: "Nghiêm túc", hint: "Đều và chắc, kiểu đọc bản tin" },
];
export const DEFAULT_EMOTION = "natural";

/**
 * How far the style is taken, against a completely flat reading.
 *
 * 0 is not "no style" — it is the reading this app produced before there was
 * one: every sentence the same pace, height and loudness. The pauses that come
 * from punctuation survive it, because a comma is a pause whatever the mood is.
 */
export const EXPRESSIVENESS_MIN = 0;
export const EXPRESSIVENESS_MAX = 1.5;
export const DEFAULT_EXPRESSIVENESS = 1;

export type Params = {
  /** `beat` only: where the replacement backing track comes from. */
  beatSource: BeatSource;
  /** `beat` only, and only when `beatSource` is `generate`. */
  beatPrompt: string;
  beatSeed: number;
  /** `beat` only, and only when `beatSource` is `remake`. */
  arrangeStyle: string;
  /** Classifier-free guidance, `CFG_RATE_MIN`…`CFG_RATE_MAX`. */
  cfgRate: number;
  /** How much of the output clarity chain to run, `CLARITY_MIN`…`CLARITY_MAX`. */
  clarity: number;
  /**
   * null = auto-detect (plan §7). Not the same as 0, which is a deliberate
   * "leave the pitch where it is" — the backend distinguishes the two and
   * measures the vocal stem only for null.
   */
  semitoneShift: number | null;
  diffusionSteps: number;
  vocalGainDb: number;
  /** `tts` only, ignored by the other two branches. */
  language: string;
  speakingRate: number;
  /** How the text is read, and how far that is taken. `tts` only as well. */
  emotion: string;
  expressiveness: number;
};

export function defaultParams(mode: Mode): Params {
  return {
    beatSource: "upload",
    beatPrompt: "",
    beatSeed: BEAT_RANDOM_SEED,
    arrangeStyle: DEFAULT_ARRANGE_STYLE,
    cfgRate: DEFAULT_CFG_RATE,
    clarity: DEFAULT_CLARITY,
    semitoneShift: null,
    diffusionSteps: DEFAULT_DIFFUSION_STEPS[mode],
    vocalGainDb: 0,
    language: DEFAULT_LANGUAGE,
    speakingRate: DEFAULT_SPEAKING_RATE,
    emotion: DEFAULT_EMOTION,
    expressiveness: DEFAULT_EXPRESSIVENESS,
  };
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

/** `70%` — the two 0–1 sliders as their labels read them. */
export function formatPercent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

/** `1,0×` — the speaking rate as the slider labels it. */
export function formatRate(rate: number): string {
  return `${rate.toFixed(2).replace(/0$/, "").replace(".", ",")}×`;
}

/**
 * The expressiveness slider's own label.
 *
 * A percentage rather than the raw multiplier: 1 means "as much as this style
 * normally has", which reads as 100% and not as a quantity of anything.
 */
export function formatExpressiveness(depth: number): string {
  return `${Math.round(depth * 100)}%`;
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

"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { REFERENCE_MAX_SEC, REFERENCE_MIN_SEC, RECORD_TARGET_SEC } from "@/lib/params";

/**
 * Record the reference voice in the browser.
 *
 * Plan §6 puts this above every other frontend feature, and it is also the one
 * with the most device-specific behaviour. What iOS Safari forces:
 *
 * * **Container.** Safari records `audio/mp4`; everything else records
 *   `audio/webm`. Both are fine downstream — the backend decodes through
 *   ffmpeg on a temp file precisely so an iPhone recording works — but the
 *   filename has to match the container or the extension lies about the bytes.
 * * **No timeslice.** `ondataavailable` is only reliable at `stop()` on iOS, so
 *   the chunks are collected and joined at the end rather than streamed.
 * * **Length comes from a timer.** A `<audio>` element reports `Infinity` for
 *   the duration of a MediaRecorder blob on Safari, so elapsed time is measured
 *   here instead of read back off the recording.
 * * **AudioContext starts suspended** until a gesture; `resume()` after
 *   `getUserMedia` is what makes the level meter move.
 */

const MIME_CANDIDATES = [
  { mime: "audio/mp4", ext: "m4a" }, // Safari, iOS included
  { mime: "audio/webm;codecs=opus", ext: "webm" },
  { mime: "audio/webm", ext: "webm" },
  { mime: "audio/ogg;codecs=opus", ext: "ogg" },
];

const METER_BARS = 32;

export function recorderSupported(): boolean {
  return (
    typeof window !== "undefined" &&
    typeof window.MediaRecorder !== "undefined" &&
    !!navigator.mediaDevices?.getUserMedia
  );
}

function pickFormat(): { mime: string; ext: string } {
  for (const candidate of MIME_CANDIDATES) {
    if (MediaRecorder.isTypeSupported?.(candidate.mime)) return candidate;
  }
  // No `isTypeSupported` at all: let the browser choose and guess the wrapper
  // from the blob's own type when the recording stops.
  return { mime: "", ext: "webm" };
}

export function Recorder({
  onRecorded,
  disabled,
}: {
  onRecorded: (file: File) => void;
  disabled?: boolean;
}) {
  const [recording, setRecording] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [levels, setLevels] = useState<number[]>(() => new Array(METER_BARS).fill(0));
  const [error, setError] = useState<string | null>(null);

  const recorder = useRef<MediaRecorder | null>(null);
  const stream = useRef<MediaStream | null>(null);
  const audioContext = useRef<AudioContext | null>(null);
  const frame = useRef<number | null>(null);
  const startedAt = useRef(0);
  const ticker = useRef<ReturnType<typeof setInterval> | null>(null);

  const teardown = useCallback(() => {
    if (frame.current !== null) cancelAnimationFrame(frame.current);
    frame.current = null;
    if (ticker.current) clearInterval(ticker.current);
    ticker.current = null;
    stream.current?.getTracks().forEach((track) => track.stop());
    stream.current = null;
    void audioContext.current?.close().catch(() => {});
    audioContext.current = null;
    recorder.current = null;
  }, []);

  // A live microphone that outlives the page is the worst bug this component
  // could ship, so the unmount path releases the tracks unconditionally.
  useEffect(() => teardown, [teardown]);

  function meter(source: MediaStream) {
    const Context = window.AudioContext ?? window.webkitAudioContext;
    if (!Context) return;
    const context = new Context();
    audioContext.current = context;
    void context.resume().catch(() => {});

    const analyser = context.createAnalyser();
    analyser.fftSize = 1024;
    context.createMediaStreamSource(source).connect(analyser);
    const samples = new Uint8Array(analyser.fftSize);

    const draw = () => {
      analyser.getByteTimeDomainData(samples);
      let peak = 0;
      for (const sample of samples) peak = Math.max(peak, Math.abs(sample - 128) / 128);
      setLevels((previous) => [...previous.slice(1), Math.min(1, peak * 1.6)]);
      frame.current = requestAnimationFrame(draw);
    };
    frame.current = requestAnimationFrame(draw);
  }

  async function start() {
    setError(null);
    try {
      const source = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true },
      });
      stream.current = source;

      const format = pickFormat();
      const instance = new MediaRecorder(source, format.mime ? { mimeType: format.mime } : {});
      const chunks: BlobPart[] = [];
      instance.ondataavailable = (event) => {
        if (event.data.size) chunks.push(event.data);
      };
      instance.onstop = () => {
        const seconds = (Date.now() - startedAt.current) / 1000;
        const type = instance.mimeType || format.mime || "audio/webm";
        const ext = type.includes("mp4") ? "m4a" : type.includes("ogg") ? "ogg" : format.ext;
        teardown();
        setRecording(false);
        setLevels(new Array(METER_BARS).fill(0));
        if (seconds < REFERENCE_MIN_SEC) {
          setError(`Mới ${seconds.toFixed(1)}s — cần ít nhất ${REFERENCE_MIN_SEC}s`);
          return;
        }
        onRecorded(new File([new Blob(chunks, { type })], `reference.${ext}`, { type }));
      };

      recorder.current = instance;
      startedAt.current = Date.now();
      // No timeslice argument: iOS only delivers the data on stop.
      instance.start();
      setRecording(true);
      setElapsed(0);
      meter(source);

      ticker.current = setInterval(() => {
        const seconds = (Date.now() - startedAt.current) / 1000;
        setElapsed(seconds);
        // The backend trims anything past this anyway, so stop rather than
        // upload seconds that get thrown away.
        if (seconds >= REFERENCE_MAX_SEC) instance.stop();
      }, 100);
    } catch (caught) {
      teardown();
      setRecording(false);
      setError(
        caught instanceof DOMException && caught.name === "NotAllowedError"
          ? "Chưa được cấp quyền micro. Bật trong cài đặt trình duyệt rồi thử lại."
          : "Không mở được micro trên thiết bị này.",
      );
    }
  }

  function stop() {
    recorder.current?.stop();
  }

  const remaining = Math.max(0, RECORD_TARGET_SEC - elapsed);
  const longEnough = elapsed >= REFERENCE_MIN_SEC;

  return (
    <div className="recorder">
      <div className="meter" aria-hidden="true">
        {levels.map((level, index) => (
          <span
            key={index}
            className="meter-bar"
            style={{ transform: `scaleY(${0.06 + level})` }}
          />
        ))}
      </div>

      <div className="recorder-controls">
        <button
          type="button"
          className={recording ? "button danger" : "button"}
          disabled={disabled}
          onClick={recording ? stop : start}
        >
          {recording ? "Dừng" : "Bắt đầu ghi"}
        </button>
        <p className="recorder-status" aria-live="polite">
          {recording
            ? remaining > 0
              ? `Còn ${remaining.toFixed(0)}s · đọc một câu bất kỳ`
              : longEnough
                ? "Đủ rồi — bấm Dừng bất cứ lúc nào"
                : "Nói thêm một chút nữa"
            : `Ghi khoảng ${RECORD_TARGET_SEC}s, nói tự nhiên, tránh chỗ ồn`}
        </p>
      </div>

      {error && (
        <p className="field-error" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}

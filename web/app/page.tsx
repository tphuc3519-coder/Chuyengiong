"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { Advanced } from "./components/Advanced";
import { ConsentGate } from "./components/ConsentGate";
import { FileDrop } from "./components/FileDrop";
import { ModeSelect } from "./components/ModeSelect";
import { Progress } from "./components/Progress";
import { ReferencePicker } from "./components/ReferencePicker";
import { Result } from "./components/Result";
import {
  AUDIO_ACCEPT,
  JOBS_PER_HOUR,
  MAX_INPUT_BYTES,
  SOURCE_MAX_SEC,
  defaultParams,
  forMode,
  type Mode,
  type Params,
} from "@/lib/params";
import { useConversion } from "@/lib/useConversion";

/**
 * The whole flow, in the order plan §6 lays it out:
 *
 *   mode → nguồn → giọng mẫu → tinh chỉnh → đồng thuận → convert → nghe → tải
 *
 * One page, no routing: every step is visible at once so nothing is hidden
 * behind a "next" button on a phone screen, and the run replaces the form in
 * place rather than navigating away from it.
 */
export default function Page() {
  const [mode, setMode] = useState<Mode>("song");
  const [params, setParams] = useState<Params>(() => defaultParams("song"));
  const [source, setSource] = useState<File | null>(null);
  const [reference, setReference] = useState<File | null>(null);
  const [consent, setConsent] = useState(false);

  const [remaining, setRemaining] = useState<number | null>(null);
  const [detected, setDetected] = useState<number | null>(null);

  const { state, start, reset } = useConversion();
  const busy = state.phase === "uploading" || state.phase === "running";
  const ready = Boolean(source && reference && consent) && !busy;

  // `/submit` reports what is left of the hourly allowance, and `reset` wipes
  // the run state — so keep it here, where it survives into the next attempt.
  useEffect(() => {
    if (state.jobsRemaining !== null) setRemaining(state.jobsRemaining);
  }, [state.jobsRemaining]);

  // Likewise for the measured pitch shift: it is what the "Tự động" control
  // shows as the last applied value, and what the slider opens on if the user
  // decides to override it (plan §7).
  useEffect(() => {
    if (state.semitoneShift !== null) setDetected(state.semitoneShift);
  }, [state.semitoneShift]);

  function switchMode(next: Mode) {
    setMode(next);
    setParams((current) => forMode(current, next, mode));
  }

  function convert() {
    if (!source || !reference) return;
    void start({
      mode,
      params,
      source,
      reference,
      referenceName: reference.name || "reference.wav",
      consent,
    });
  }

  return (
    <main className="page">
      <header className="masthead">
        <h1>Chuyển giọng</h1>
        <p>
          Đưa vào một bài hát hoặc đoạn thoại cùng một giọng mẫu — nhận về bản đã đổi giọng, nhạc
          nền giữ nguyên.
        </p>
      </header>

      {state.phase === "done" && state.blob && state.jobId ? (
        <section className="card">
          <h2>Xong</h2>
          <Result
            blob={state.blob}
            jobId={state.jobId}
            semitoneShift={state.semitoneShift}
            onReset={reset}
          />
        </section>
      ) : busy ? (
        <section className="card">
          <h2>Đang xử lý</h2>
          <Progress
            status={state.status}
            progress={state.progress}
            uploaded={state.uploaded}
            elapsedMs={state.elapsedMs}
            semitoneShift={state.semitoneShift}
          />
          <button type="button" className="button ghost" onClick={reset}>
            Huỷ
          </button>
        </section>
      ) : (
        <form
          className="card"
          onSubmit={(event) => {
            event.preventDefault();
            convert();
          }}
        >
          <fieldset className="step">
            <legend>1 · Kiểu nội dung</legend>
            <ModeSelect value={mode} onChange={switchMode} />
          </fieldset>

          <fieldset className="step">
            <legend>2 · File nguồn</legend>
            <FileDrop
              file={source}
              onFile={setSource}
              accept={AUDIO_ACCEPT}
              maxBytes={MAX_INPUT_BYTES}
              label={mode === "song" ? "Bài hát" : "Đoạn thoại"}
              hint={`Kéo thả hoặc bấm để chọn · tối đa ${SOURCE_MAX_SEC / 60} phút`}
            />
          </fieldset>

          <fieldset className="step">
            <legend>3 · Giọng mẫu</legend>
            <ReferencePicker file={reference} onFile={setReference} />
          </fieldset>

          <fieldset className="step">
            <legend>4 · Tuỳ chọn</legend>
            <Advanced mode={mode} params={params} detected={detected} onChange={setParams} />
          </fieldset>

          <fieldset className="step">
            <legend>5 · Đồng thuận</legend>
            <ConsentGate checked={consent} onChange={setConsent} />
          </fieldset>

          {state.phase === "error" && state.error && (
            <p className="banner error" role="alert">
              {state.error}
            </p>
          )}

          <button type="submit" className="button primary" disabled={!ready}>
            Chuyển giọng
          </button>
          {!ready && (
            <p className="field-note">
              {!source
                ? "Chọn file nguồn để tiếp tục."
                : !reference
                  ? "Thêm giọng mẫu — tải lên, ghi âm, hoặc chọn giọng có sẵn."
                  : "Tick ô đồng thuận để tiếp tục."}
            </p>
          )}
        </form>
      )}

      <footer className="footnote">
        {remaining !== null && (
          <p>
            Còn {remaining}/{JOBS_PER_HOUR} lượt chuyển trong giờ này.
          </p>
        )}
        <p>
          File tải lên và kết quả bị xoá khỏi máy chủ sau 6 giờ. Đừng dùng giọng của người khác khi
          chưa được họ cho phép — xem <Link href="/terms">điều khoản sử dụng</Link>.
        </p>
      </footer>
    </main>
  );
}

"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { Advanced } from "./components/Advanced";
import { BeatSource } from "./components/BeatSource";
import { ConsentGate } from "./components/ConsentGate";
import { FileDrop } from "./components/FileDrop";
import { ModeSelect } from "./components/ModeSelect";
import { Progress } from "./components/Progress";
import { ReferencePicker } from "./components/ReferencePicker";
import { Result } from "./components/Result";
import { TextInput } from "./components/TextInput";
import {
  AUDIO_ACCEPT,
  MAX_INPUT_BYTES,
  SOURCE_MAX_SEC,
  defaultParams,
  forMode,
  maxCharsFor,
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
 *
 * Step 2 is the one place the modes differ: everything except `tts` takes a
 * file, `tts` takes text. Everything after it — the voice sample, the tuning,
 * the gate, the run — is the same for all of them, which is the point of
 * putting text in as another *source* rather than as a second app.
 */

/**
 * What step 2 calls the file it is asking for. `vocal` is the mode that says
 * "this is already only a voice, do not go looking for a backing track in it".
 */
const SOURCE_LABEL: Record<Exclude<Mode, "tts">, string> = {
  song: "Bài hát",
  beat: "Bài hát",
  vocal: "Giọng hát đã tách",
  speech: "Đoạn thoại",
};
export default function Page() {
  const [mode, setMode] = useState<Mode>("song");
  const [params, setParams] = useState<Params>(() => defaultParams("song"));
  const [source, setSource] = useState<File | null>(null);
  const [text, setText] = useState("");
  const [reference, setReference] = useState<File | null>(null);
  // `beat` mode only, and only when the user brought their own rather than
  // describing one. Kept beside `source` rather than inside `params` because
  // it is a file, and `params` is the thing that survives a mode switch.
  const [beat, setBeat] = useState<File | null>(null);
  const [consent, setConsent] = useState(false);

  const [remaining, setRemaining] = useState<number | null>(null);
  const [detected, setDetected] = useState<number | null>(null);

  const { state, start, resume, reset } = useConversion();
  const busy = state.phase === "uploading" || state.phase === "running";
  // Over the limit is its own state, not "no source": the words are there and
  // the fix is to cut some, which is a different sentence from "write
  // something". Switching to a language with a shorter limit is how a finished
  // paragraph lands here.
  const tooLong = mode === "tts" && text.length > maxCharsFor(params.language);
  const hasSource = mode === "tts" ? text.trim().length > 0 : source !== null;
  // The `beat` branch needs exactly one of the two, and the backend refuses a
  // job carrying neither — so the button waits for it rather than letting the
  // upload find out.
  const hasBeat =
    mode !== "beat" ||
    (params.beatSource === "generate" ? params.beatPrompt.trim().length > 0 : beat !== null);
  const ready = hasSource && hasBeat && !tooLong && reference !== null && consent && !busy;

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
    if (!ready || !reference) return;
    void start({
      mode,
      params,
      // `tts` sends the text and no file; the other two send the file and the
      // backend never reads `text`.
      source: mode === "tts" ? null : source,
      text,
      beat: mode === "beat" && params.beatSource === "upload" ? beat : null,
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
          Đưa vào một bài hát, một đoạn thoại, hoặc chỉ một đoạn văn bản — cùng với giọng mẫu. Nhận
          về bản hát hoặc đọc bằng đúng giọng đó, nhạc nền giữ nguyên.
        </p>
      </header>

      {/* No `state.blob` in the condition: the run is done when the server says
          so, and the bytes are fetched afterwards for the player alone. */}
      {state.phase === "done" && state.jobId && state.resultUrl ? (
        <section className="card">
          <h2>Xong</h2>
          <Result
            blob={state.blob}
            resultUrl={state.resultUrl}
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
            <legend>2 · {mode === "tts" ? "Văn bản" : "File nguồn"}</legend>
            {mode === "tts" ? (
              <TextInput
                text={text}
                onText={setText}
                language={params.language}
                onLanguage={(language) => setParams((current) => ({ ...current, language }))}
              />
            ) : (
              <FileDrop
                file={source}
                onFile={setSource}
                accept={AUDIO_ACCEPT}
                maxBytes={MAX_INPUT_BYTES}
                label={SOURCE_LABEL[mode as Exclude<Mode, "tts">]}
                hint={`Kéo thả hoặc bấm để chọn · tối đa ${SOURCE_MAX_SEC / 60} phút`}
              />
            )}
          </fieldset>

          {mode === "beat" && (
            <fieldset className="step">
              <legend>2b · Beat mới</legend>
              <BeatSource
                params={params}
                beat={beat}
                onBeat={setBeat}
                onChange={setParams}
                disabled={busy}
              />
            </fieldset>
          )}

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
            <div className="banner error" role="alert">
              <p>{state.error}</p>
              {/*
                Only when there is something to go back for. A job the server
                reported as `failed` is finished being wrong: polling it again
                takes two seconds to reproduce the same message, which reads as
                a button that does nothing.
              */}
              {state.jobId && state.status !== "failed" && (
                <button
                  type="button"
                  className="link-button"
                  onClick={() => resume(state.jobId as string)}
                >
                  Lấy lại kết quả của lượt vừa rồi
                </button>
              )}
              {/*
                And a way out when even that fails: the in-page download reads
                megabytes into a blob, and that is the step that breaks on a
                phone. Strictly once the run reached `done`, because that is
                when a file exists — `/download` answers anything earlier with
                409 JSON, and following the link would navigate the tab off the
                app and take the job id with it.
              */}
              {state.resultUrl && state.status === "done" && (
                <a className="link-button" href={state.resultUrl}>
                  Hoặc tải thẳng file về máy
                </a>
              )}
            </div>
          )}

          <button type="submit" className="button primary" disabled={!ready}>
            Chuyển giọng
          </button>
          {!ready && (
            <p className="field-note">
              {!hasSource
                ? mode === "tts"
                  ? "Gõ văn bản cần đọc để tiếp tục."
                  : "Chọn file nguồn để tiếp tục."
                : !hasBeat
                  ? params.beatSource === "generate"
                    ? "Mô tả beat muốn sinh để tiếp tục."
                    : "Chọn file beat thay thế để tiếp tục."
                  : tooLong
                    ? `Văn bản dài quá giới hạn của ngôn ngữ đang chọn (${maxCharsFor(params.language)} ký tự) — cắt bớt để tiếp tục.`
                    : !reference
                      ? "Thêm giọng mẫu — tải lên, ghi âm, hoặc chọn giọng có sẵn."
                      : "Tick ô đồng thuận để tiếp tục."}
            </p>
          )}
        </form>
      )}

      <footer className="footnote">
        {/* No denominator: the cap is whatever `JOBS_PER_HOUR` says on the
            deployment, and printing 5 there gave "Còn 9/5 lượt". */}
        {remaining !== null && <p>Còn {remaining} lượt chuyển trong giờ này.</p>}
        <p>
          File tải lên và kết quả bị xoá khỏi máy chủ sau 6 giờ. Đừng dùng giọng của người khác khi
          chưa được họ cho phép — xem <Link href="/terms">điều khoản sử dụng</Link>.
        </p>
      </footer>
    </main>
  );
}

"use client";

import Link from "next/link";
import { useCallback, useEffect, useId, useState } from "react";

import { FileDrop } from "../components/FileDrop";
import { AUDIO_ACCEPT, MAX_INPUT_BYTES } from "@/lib/params";

/**
 * Quản lý giọng RVC: dán link, đặt tên, bấm nút.
 *
 * Đây là thứ thay cho việc mở console gõ `fetch("/api/rvc/add-model", …)`.
 * Cùng một endpoint, chỉ khác là không phải là lập trình viên mới dùng được.
 *
 * Trang đứng riêng khỏi `/` vì nó không nằm trong luồng chuyển giọng — nạp
 * giọng là việc làm một lần rồi thôi, còn trang chính là việc làm mỗi ngày.
 */

type Model = { name: string; has_index: boolean; size_mb: number };

/**
 * Phải khớp từng bước với `_safe_name()` trong modal_rvc.py: trang này hứa
 * trước tên sẽ lưu, nên lệch nhau là hứa một đằng lưu một nẻo — hoặc tệ hơn,
 * server từ chối một cái tên mà trang vừa bảo là hợp lệ.
 *
 * Bỏ dấu chứ không từ chối: "Sơn Tùng MTP" → "son-tung-mtp".
 */
function normalise(name: string): string {
  return name
    .trim()
    .toLowerCase()
    .replace(/đ/g, "d")
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9._-]+/g, "-")
    .replace(/-{2,}/g, "-")
    .replace(/^[-._]+|[-._]+$/g, "");
}

export default function RvcModelsPage() {
  const urlId = useId();
  const nameId = useId();
  const modelId = useId();
  const pitchId = useId();
  const indexId = useId();
  const protectId = useId();
  const methodId = useId();

  const [url, setUrl] = useState("");
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState<string | null>(null);

  const [models, setModels] = useState<Model[] | null>(null);
  const [listError, setListError] = useState<string | null>(null);

  const [vocal, setVocal] = useState<File | null>(null);
  const [chosen, setChosen] = useState("");
  const [pitch, setPitch] = useState(0);
  const [indexRate, setIndexRate] = useState(0.6);
  const [protect, setProtect] = useState(0.33);
  const [f0Method, setF0Method] = useState("rmvpe");
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);
  const [result, setResult] = useState<string | null>(null);

  const load = useCallback(async () => {
    setListError(null);
    try {
      const res = await fetch("/api/rvc/models", { cache: "no-store" });
      const body = await res.json();
      if (!res.ok) {
        setListError(body?.error ?? "Không đọc được danh sách giọng");
        return;
      }
      setModels(body.models ?? []);
    } catch {
      setListError("Không gọi được tới server");
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  // Chọn sẵn giọng đầu tiên: gần như lúc nào cũng chỉ có một, bắt chọn tay là
  // thừa một bước.
  useEffect(() => {
    if (!chosen && models && models.length > 0) setChosen(models[0].name);
  }, [models, chosen]);

  // Object URL sống tới khi bị revoke; không dọn thì mỗi lần chạy lại giữ thêm
  // vài chục MB trong tab.
  useEffect(() => {
    return () => {
      if (result) URL.revokeObjectURL(result);
    };
  }, [result]);

  const slug = normalise(name);
  const ready = url.trim().length > 0 && slug.length > 0 && !busy;

  async function add() {
    setBusy(true);
    setError(null);
    setDone(null);
    try {
      const res = await fetch("/api/rvc/add-model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: url.trim(), name: slug }),
      });
      const body = await res.json();

      // Hai tầng lỗi: route trả {error}, còn Modal trả {ok:false, error}.
      // Người dùng không cần biết tầng nào hỏng, chỉ cần đọc được câu tiếng Việt.
      if (!res.ok || body?.ok === false) {
        setError(body?.error ?? `Lỗi ${res.status}`);
        return;
      }

      setDone(
        body.has_index
          ? `Đã thêm “${body.name}” (${body.size_mb} MB, có file .index)`
          : `Đã thêm “${body.name}” (${body.size_mb} MB, không có .index — giọng sẽ kém giống hơn)`,
      );
      setUrl("");
      setName("");
      await load();
    } catch {
      setError("Không gọi được tới server. Nếu đây là lần đầu, thử lại lần nữa.");
    } finally {
      setBusy(false);
    }
  }

  async function run() {
    if (!vocal || !chosen) return;
    setRunning(true);
    setRunError(null);
    if (result) URL.revokeObjectURL(result);
    setResult(null);

    try {
      // Địa chỉ để POST lấy từ server, không nhúng vào bundle — cùng lý do
      // /api/config làm thế cho pipeline chính: một bản build chạy được với
      // mọi deployment.
      const cfgRes = await fetch("/api/rvc/config", { cache: "no-store" });
      const cfg = await cfgRes.json();
      if (!cfgRes.ok || !cfg?.convertUrl) {
        setRunError(cfg?.error ?? "Chưa cấu hình địa chỉ đổi giọng");
        return;
      }

      const form = new FormData();
      form.append("audio", vocal);
      form.append("model", chosen);
      form.append("pitch", String(pitch));
      form.append("index_rate", String(indexRate));
      form.append("protect", String(protect));
      form.append("f0_method", f0Method);

      // Thẳng lên Modal, không qua route: xem chú thích đầu file route.ts.
      const res = await fetch(cfg.convertUrl, { method: "POST", body: form });
      if (!res.ok) {
        const body = await res.json().catch(() => null);
        setRunError(body?.error ?? `Modal trả lỗi ${res.status}`);
        return;
      }

      setResult(URL.createObjectURL(await res.blob()));
    } catch {
      setRunError(
        "Không gọi được tới máy chủ xử lý. Lần đầu sau một lúc không dùng thì container phải khởi động lại — thử lại lần nữa.",
      );
    } finally {
      setRunning(false);
    }
  }

  return (
    <main className="page">
      <header className="masthead">
        <h1>Giọng RVC</h1>
        <p>
          Nạp giọng đã train sẵn từ voice-models.com. Dán link tải, đặt tên, bấm thêm — làm một lần,
          giọng nằm lại trên server.
        </p>
      </header>

      <section className="card">
        <h2>Thêm giọng</h2>

        <div className="step">
          <label htmlFor={urlId}>Link tải model</label>
          <input
            id={urlId}
            className="rvc-input"
            type="url"
            inputMode="url"
            value={url}
            disabled={busy}
            placeholder="https://…/model.zip"
            onChange={(event) => setUrl(event.target.value)}
          />
          {/*
            Chỗ hỏng thường gặp nhất, và nó hỏng một cách khó đoán: link trả về
            trang HTML thì server chỉ báo "không tìm thấy .pth", nghe như model
            lỗi chứ không như link sai.
          */}
          <p className="field-note">
            Phải là link tới đúng file (.zip hoặc .pth), không phải link trang tải. Nếu voice-models
            đưa bạn sang Mega hay Pixeldrain: bấm tải cho nó chạy, rồi vào{" "}
            <code>chrome://downloads</code>, chuột phải mục vừa tải → Copy link address.
          </p>
        </div>

        <div className="step">
          <label htmlFor={nameId}>Đặt tên</label>
          <input
            id={nameId}
            className="rvc-input"
            type="text"
            value={name}
            disabled={busy}
            placeholder="son-tung"
            onChange={(event) => setName(event.target.value)}
          />
          <p className="field-note">
            {slug && slug !== name.trim()
              ? `Sẽ lưu thành “${slug}”.`
              : "Chữ thường, không dấu, không khoảng trắng."}
          </p>
        </div>

        {error ? <p className="field-error">{error}</p> : null}
        {done ? <p className="field-note">{done}</p> : null}

        <button
          className="button primary"
          type="button"
          disabled={!ready}
          onClick={() => void add()}
        >
          {busy ? "Đang tải về server…" : "Thêm giọng"}
        </button>

        <p className="field-note">
          Lần đầu có thể mất 30-60 giây: server phải khởi động rồi mới tải. Bấm một lần rồi chờ.
        </p>
      </section>

      <section className="card">
        <h2>Đổi giọng</h2>

        {models !== null && models.length === 0 ? (
          <p className="field-note">Thêm một giọng ở trên trước đã.</p>
        ) : (
          <>
            <FileDrop
              file={vocal}
              onFile={setVocal}
              accept={AUDIO_ACCEPT}
              maxBytes={MAX_INPUT_BYTES}
              label="Giọng hát đã tách"
              hint="Chỉ đưa vocal vào, đừng đưa cả bài — RVC không tự tách nhạc nền."
            />

            <div className="step">
              <label htmlFor={modelId}>Đổi sang giọng</label>
              <select
                id={modelId}
                className="rvc-input"
                value={chosen}
                disabled={running}
                onChange={(event) => setChosen(event.target.value)}
              >
                {(models ?? []).map((model) => (
                  <option key={model.name} value={model.name}>
                    {model.name}
                  </option>
                ))}
              </select>
            </div>

            <div className="step">
              <label htmlFor={pitchId}>
                Dịch cao độ: {pitch > 0 ? `+${pitch}` : pitch} nửa cung
              </label>
              <input
                id={pitchId}
                className="slider"
                type="range"
                min={-12}
                max={12}
                step={1}
                value={pitch}
                disabled={running}
                onChange={(event) => setPitch(Number(event.target.value))}
              />
              {/*
                Tham số ảnh hưởng nhiều nhất, và là thứ duy nhất người dùng gần
                như chắc chắn phải chỉnh — nên nó nằm ngoài, không nằm trong
                phần thu gọn.
              */}
              <p className="field-note">
                Nam sang nữ thường là +12, nữ sang nam −12, cùng giới để 0. Sai quãng thì giọng ra
                nghe như robot.
              </p>
            </div>

            <details className="disclosure">
              <summary>Tinh chỉnh thêm</summary>

              <div className="step">
                <label htmlFor={indexId}>Bám giọng gốc của model: {indexRate.toFixed(2)}</label>
                <input
                  id={indexId}
                  className="slider"
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={indexRate}
                  disabled={running}
                  onChange={(event) => setIndexRate(Number(event.target.value))}
                />
                <p className="field-note">0,5–0,7 là vùng dùng được. Cao quá sẽ có tiếng rè.</p>
              </div>

              <div className="step">
                <label htmlFor={protectId}>Giữ phụ âm: {protect.toFixed(2)}</label>
                <input
                  id={protectId}
                  className="slider"
                  type="range"
                  min={0}
                  max={0.5}
                  step={0.01}
                  value={protect}
                  disabled={running}
                  onChange={(event) => setProtect(Number(event.target.value))}
                />
                <p className="field-note">Lời bị nhoè thì tăng lên 0,5.</p>
              </div>

              <div className="step">
                <label htmlFor={methodId}>Cách dò cao độ</label>
                <select
                  id={methodId}
                  className="rvc-input"
                  value={f0Method}
                  disabled={running}
                  onChange={(event) => setF0Method(event.target.value)}
                >
                  <option value="rmvpe">rmvpe — mặc định, hợp với hát</option>
                  <option value="harvest">harvest — chậm hơn, đôi khi mượt hơn ở giọng trầm</option>
                  <option value="crepe">crepe</option>
                  <option value="pm">pm — nhanh nhất, kém nhất</option>
                </select>
              </div>
            </details>

            {runError ? <p className="field-error">{runError}</p> : null}

            <button
              className="button primary"
              type="button"
              disabled={!vocal || !chosen || running}
              onClick={() => void run()}
            >
              {running ? "Đang đổi giọng…" : "Đổi giọng"}
            </button>

            <p className="field-note">
              Một bài 4 phút mất chừng 40–60 giây. Lần chạy đầu lâu hơn vì phải nạp model.
            </p>

            {result ? (
              <div className="result">
                <audio src={result} controls preload="metadata" />
                <div className="result-actions">
                  <a className="button" href={result} download={`${chosen}.wav`}>
                    Tải về
                  </a>
                </div>
              </div>
            ) : null}
          </>
        )}
      </section>

      <section className="card">
        <h2>Đã có trên server</h2>

        {listError ? <p className="field-error">{listError}</p> : null}

        {models === null && !listError ? <p className="field-note">Đang đọc…</p> : null}

        {models !== null && models.length === 0 ? (
          <p className="field-note">Chưa có giọng nào. Thêm cái đầu tiên ở trên.</p>
        ) : null}

        {models !== null && models.length > 0 ? (
          <ul className="rvc-list">
            {models.map((model) => (
              <li key={model.name}>
                <span>{model.name}</span>
                <span className="field-note">
                  {model.size_mb} MB{model.has_index ? " · có .index" : " · thiếu .index"}
                </span>
              </li>
            ))}
          </ul>
        ) : null}

        <button className="button ghost" type="button" onClick={() => void load()}>
          Đọc lại
        </button>
      </section>

      <p className="footnote">
        <Link href="/">← Về trang chuyển giọng</Link>
      </p>
    </main>
  );
}

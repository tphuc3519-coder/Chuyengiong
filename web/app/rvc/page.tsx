"use client";

import Link from "next/link";
import { useCallback, useEffect, useId, useState } from "react";

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

  const [url, setUrl] = useState("");
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState<string | null>(null);

  const [models, setModels] = useState<Model[] | null>(null);
  const [listError, setListError] = useState<string | null>(null);

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

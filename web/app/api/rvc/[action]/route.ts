// web/app/api/rvc/[action]/route.ts
//
// Proxy same-origin sang Modal cho những lệnh gọi NHỎ: liệt kê giọng và thêm
// giọng từ URL. Cả hai chỉ trao đổi vài trăm byte JSON.
//
// `convert` KHÔNG đi qua đây. Body của serverless function trên Vercel bị chặn
// ở 4.5MB cả hai chiều, mà một file vocal 4 phút đã hơn thế và wav trả về còn
// to hơn nữa — proxy không tải nổi audio theo chiều nào. Trình duyệt POST
// thẳng lên Modal, đúng cách `lib/api.ts` đã làm với pipeline chính. Địa chỉ
// để POST lấy ở `GET /api/rvc/config`.
//
// Cần 3 biến môi trường trên Vercel:
//   MODAL_RVC_CONVERT_URL
//   MODAL_RVC_LIST_URL
//   MODAL_RVC_ADD_URL
// Lấy từ output của `modal deploy modal_rvc.py`.

import { NextRequest, NextResponse } from "next/server";

export const maxDuration = 60;
export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const ENDPOINTS: Record<string, { url?: string; method: "GET" | "POST" }> = {
  models: { url: process.env.MODAL_RVC_LIST_URL, method: "GET" },
  "add-model": { url: process.env.MODAL_RVC_ADD_URL, method: "POST" },
};

export async function GET(_req: NextRequest, { params }: { params: Promise<{ action: string }> }) {
  const { action } = await params;

  // Không phải proxy: chỉ đưa địa chỉ để trình duyệt tự gọi, giống /api/config
  // đưa apiBase cho pipeline chính.
  if (action === "config") {
    const convertUrl = process.env.MODAL_RVC_CONVERT_URL;
    if (!convertUrl) {
      return NextResponse.json({ error: "Thiếu MODAL_RVC_CONVERT_URL" }, { status: 503 });
    }
    return NextResponse.json({ convertUrl }, { headers: { "cache-control": "no-store" } });
  }

  return forward(action, null);
}

export async function POST(req: NextRequest, { params }: { params: Promise<{ action: string }> }) {
  const { action } = await params;
  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Body không phải JSON" }, { status: 400 });
  }
  return forward(action, body);
}

async function forward(action: string, body: unknown) {
  const target = ENDPOINTS[action];

  if (!target) {
    return NextResponse.json({ error: `Action không hợp lệ: ${action}` }, { status: 404 });
  }
  if (!target.url) {
    return NextResponse.json({ error: `Thiếu biến môi trường cho '${action}'` }, { status: 500 });
  }

  try {
    const res = await fetch(target.url, {
      method: target.method,
      headers: { "Content-Type": "application/json" },
      body: target.method === "POST" ? JSON.stringify(body) : undefined,
      cache: "no-store",
      // add-model phải tải nguyên file model về Volume, và cold start của Modal
      // cộng thêm vào đó — 50s là chật, nhưng maxDuration của route là trần cứng.
      signal: AbortSignal.timeout(55_000),
    });

    const text = await res.text();
    try {
      return NextResponse.json(JSON.parse(text), { status: res.status });
    } catch {
      return NextResponse.json(
        { error: "Modal trả về dữ liệu không đọc được", raw: text.slice(0, 500) },
        { status: 502 },
      );
    }
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    const timedOut = msg.includes("timeout") || msg.includes("aborted");
    return NextResponse.json(
      { error: timedOut ? "Modal quá thời gian chờ — thử lại lần nữa" : msg },
      { status: timedOut ? 504 : 502 },
    );
  }
}

// web/app/api/rvc/[action]/route.ts
//
// Proxy same-origin sang Modal. Giữ URL Modal ở phía server để
// trình duyệt không gọi thẳng (tránh CORS và lộ endpoint).
//
// Cần 3 biến môi trường trên Vercel:
//   MODAL_RVC_CONVERT_URL
//   MODAL_RVC_LIST_URL
//   MODAL_RVC_ADD_URL
// Lấy từ output của `modal deploy modal_rvc.py`.
//
// Ba URL riêng, không phải một base như MODAL_API_URL: app RVC publish ba
// fastapi_endpoint độc lập, mỗi cái một domain — khác với modal_app, nơi
// `api` là một ASGI app duy nhất.

import { NextRequest, NextResponse } from "next/server";

export const maxDuration = 300;
export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const ENDPOINTS: Record<string, { url?: string; method: "GET" | "POST" }> = {
  convert: { url: process.env.MODAL_RVC_CONVERT_URL, method: "POST" },
  models: { url: process.env.MODAL_RVC_LIST_URL, method: "GET" },
  "add-model": { url: process.env.MODAL_RVC_ADD_URL, method: "POST" },
};

export async function GET(_req: NextRequest, { params }: { params: Promise<{ action: string }> }) {
  const { action } = await params;
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
      // Modal cold start có thể mất 30-60s khi tải model lần đầu
      signal: AbortSignal.timeout(280_000),
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
      { error: timedOut ? "Modal quá thời gian chờ — thử file ngắn hơn" : msg },
      { status: timedOut ? 504 : 502 },
    );
  }
}

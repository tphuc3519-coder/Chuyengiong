import type { Metadata, Viewport } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "Chuyển giọng",
  description: "Đổi giọng hát hoặc giọng nói sang một giọng mẫu, giữ nguyên nhạc nền.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  // iOS Safari zooms the page when a control smaller than 16px takes focus;
  // the stylesheet keeps inputs at 16px so this does not have to lock zoom.
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f7f7f8" },
    { media: "(prefers-color-scheme: dark)", color: "#111113" },
  ],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="vi">
      <body>{children}</body>
    </html>
  );
}

import type { Metadata } from "next";
import Link from "next/link";

import { contactEmail } from "@/lib/server";

/**
 * Plan §8 item 3: a short terms page, and the one place the rules the app
 * enforces elsewhere are written down in words a person reads.
 *
 * Short on purpose. Everything here is either something the code actually does
 * (the consent gate, the six hour TTL, the AI-generated tag, the rate limit) or
 * something it refuses to do (celebrity presets, a shared voice library). A
 * clause that describes neither would be decoration.
 *
 * Rendered per request rather than at build time so the contact address can be
 * an environment variable — a takedown route that only changes with a rebuild
 * is not much of a route.
 */
export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: "Điều khoản sử dụng — Chuyển giọng",
  description: "Quy tắc sử dụng, cách xử lý file, và giới hạn của dịch vụ chuyển giọng.",
};

const UPDATED = "26/08/2026";

export default function TermsPage() {
  const contact = contactEmail();

  return (
    <main className="page">
      <header className="masthead">
        <h1>Điều khoản sử dụng</h1>
        <p>Cập nhật {UPDATED}. Đọc mất khoảng hai phút.</p>
      </header>

      <article className="card legal">
        <h2>1. Dịch vụ này làm gì</h2>
        <p>
          Bạn đưa vào một file âm thanh và một giọng mẫu; hệ thống trả về bản đã đổi sang giọng mẫu
          đó. Với bài hát, nhạc nền được tách ra rồi ghép lại nguyên vẹn. Không có mô hình nào được
          huấn luyện trên file của bạn.
        </p>

        <h2>2. Bạn cam kết điều gì khi bấm chuyển giọng</h2>
        <p>
          Ô đồng thuận trước nút chuyển giọng là một cam kết: giọng trong file tham chiếu là giọng
          của chính bạn, hoặc bạn có quyền sử dụng nó. Máy chủ từ chối mọi yêu cầu không kèm cam kết
          này — đây không phải một ô tick cho có.
        </p>
        <p>Không được dùng dịch vụ để:</p>
        <ul>
          <li>mạo danh người khác, kể cả để đùa;</li>
          <li>
            tạo nội dung khiến người nghe tin rằng một người có thật đã nói hoặc hát điều họ chưa
            từng;
          </li>
          <li>lừa đảo, quấy rối, bôi nhọ, hoặc tạo nội dung tình dục về người có thật;</li>
          <li>
            tạo giọng trẻ em cho bất kỳ mục đích nào ngoài giọng con bạn với sự có mặt của bạn.
          </li>
        </ul>

        <h2>3. File kết quả được đánh dấu</h2>
        <p>
          Mọi file tải về đều mang metadata ghi rõ đây là nội dung tạo bởi AI. Đừng gỡ dấu này khi
          chia sẻ lại, và nên nói rõ với người nghe rằng giọng là do máy tạo.
        </p>
        <p>
          Ngoài metadata, file kết quả còn mang một watermark âm thanh không nghe thấy được. Nó
          không chứa thông tin gì về bạn — chỉ đủ để chúng tôi xác nhận một file có phải do dịch vụ
          này tạo ra hay không, khi có khiếu nại. Watermark sống sót qua việc nén lại, còn metadata
          thì không.
        </p>

        <h2>4. File của bạn ở lại bao lâu</h2>
        <p>
          File nguồn, giọng mẫu và kết quả bị xoá khỏi máy chủ sau 6 giờ, bằng một tác vụ chạy tự
          động chứ không phải khi có người nhớ ra. Không có tài khoản, không có lịch sử, không có
          cách nào lấy lại một job đã hết hạn — tải về trước khi đóng tab.
        </p>
        <p>
          Nhật ký hệ thống chỉ lưu mã job, thời điểm, trạng thái và kích thước file, để trả lời được
          khi có khiếu nại.{" "}
          <strong>Không lưu nội dung âm thanh, không lưu tên file, không lưu địa chỉ IP</strong>{" "}
          (địa chỉ chỉ được băm để đếm giới hạn lượt).
        </p>

        <h2>5. Giọng có sẵn</h2>
        <p>
          Dịch vụ không cung cấp giọng người nổi tiếng và không có thư viện giọng do người dùng chia
          sẻ. Mọi giọng mẫu có sẵn (nếu có) đều là bản thu riêng hoặc có giấy phép ghi rõ nguồn.
        </p>

        <h2>6. Giới hạn</h2>
        <ul>
          <li>5 lượt chuyển mỗi giờ cho mỗi người dùng;</li>
          <li>file nguồn tối đa 15 phút;</li>
          <li>giọng mẫu 5–30 giây;</li>
          <li>
            dịch vụ ở giai đoạn thử nghiệm, có thể dừng hoặc lỗi bất cứ lúc nào, không bảo hành.
          </li>
        </ul>

        <h2>7. Vi phạm và khiếu nại</h2>
        <p>
          Nếu bạn cho rằng giọng của mình bị dùng mà không được phép, hãy gửi mã job (hiện trên màn
          hình kết quả, cũng nằm trong tên file tải về) — đó là thứ tra được trong nhật ký. Tài
          khoản hoặc địa chỉ vi phạm sẽ bị chặn.
        </p>
        {contact ? (
          <p>
            Liên hệ: <a href={`mailto:${contact}`}>{contact}</a>
          </p>
        ) : (
          <p>Kênh liên hệ sẽ được công bố trước khi dịch vụ mở rộng ra ngoài nhóm thử nghiệm.</p>
        )}
      </article>

      <footer className="footnote">
        <p>
          <Link href="/">← Về trang chuyển giọng</Link>
        </p>
      </footer>
    </main>
  );
}

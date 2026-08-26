"use client";

/**
 * Plan §8 item 1: not pre-ticked, and submit stays disabled until it is ticked.
 *
 * The backend refuses a submit without `consent=true` regardless of what this
 * checkbox does, which is the part that actually matters — this is the place
 * the user reads the sentence, not the place the rule is enforced.
 */
export function ConsentGate({
  checked,
  onChange,
  disabled,
}: {
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <label className="consent">
      <input
        type="checkbox"
        checked={checked}
        disabled={disabled}
        onChange={(event) => onChange(event.target.checked)}
      />
      <span>
        Tôi xác nhận có quyền sử dụng giọng nói trong file tham chiếu, hoặc đó là giọng của chính
        tôi.
      </span>
    </label>
  );
}

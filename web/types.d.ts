/** Safari still exposes the prefixed constructor; the meter falls back to it. */
declare global {
  interface Window {
    webkitAudioContext?: typeof AudioContext;
  }
}

export {};

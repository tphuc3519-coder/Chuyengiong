# Preset reference voices

Drop 4–6 clips here (5–20 seconds each, one speaker, no music, no background
noise) and list them in `index.json`:

```json
[
  {
    "id": "alto-vi",
    "name": "Nữ trung — tiếng Việt",
    "file": "alto-vi.mp3",
    "license": "Tự thu, 2026"
  }
]
```

The UI hides the preset row entirely while `index.json` is empty, so shipping
without audio is a supported state — not a broken one.

Two rules, from plan §8 item 4:

- **No celebrity or public-figure voices.** Not as a preset, not as a demo.
- **Every entry needs a `license` field** naming where the recording came from:
  a voice you recorded yourself, or a dataset with an explicit licence. If you
  cannot fill that field in one line, the clip does not belong here.

`*.mp3` in this directory is exempt from the repo's audio `.gitignore` rule, so
committed presets are tracked.

#!/usr/bin/env python3
"""
Fix mojibake in PROJECT_XRAY.md.

Strategy: build an explicit character-level replacement map for every
non-ASCII Unicode codepoint that appears in the file as mojibake.
This is safer than a streaming decoder because it handles partial sequences.
"""

def build_mojibake_map() -> dict:
    """
    For each Unicode codepoint of interest, compute the mojibake string
    that results from: UTF-8-encode → decode-as-cp1252 → re-encode-as-UTF-8 → decode-as-UTF-8.
    Returns {mojibake_string: correct_unicode_char}.
    """
    # Characters that appear in typical Markdown/docs files: box-drawing,
    # arrows, em-dash, en-dash, bullets, checkmarks, emojis, math symbols.
    candidates = list(range(0x2500, 0x2600))   # box drawing (inc. ┌┐└┘│─├┤┬┴┼╔╗╚╝═║...)
    candidates += [
        0x2192, 0x2190, 0x2191, 0x2193,   # arrows →←↑↓
        0x21d2, 0x21d0,                    # ⇒ ⇐
        0x2014, 0x2013, 0x2012,            # em-dash, en-dash, figure-dash
        0x00d7, 0x00f7,                    # × ÷
        0x2022, 0x2023,                    # bullets •‣
        0x2026,                            # ellipsis …
        0x00b1, 0x2248, 0x2260,            # ± ≈ ≠
        0x2705, 0x231b, 0x2728,            # ✅ ⌛ ✨
        0x2713, 0x2714, 0x2717, 0x2718,    # checkmarks / x-marks
        0x2764, 0x2665,                    # ❤ ♥
        0x1f4cb, 0x1f527,                  # 📋 🔧  (4-byte UTF-8)
        0x1f4a5, 0x1f6e0,                  # 💥 🛠
        0x1f534, 0x1f7e1, 0x1f7e2,        # 🔴 🟡 🟢
        0x26a0, 0x2139,                    # ⚠ ℹ
        0x2714, 0x2716,                    # ✔ ✖
        0x1f4dd, 0x1f4c4,                  # 📝 📄
        0x1f9ea,                           # 🧪
        0x1f50c,                           # 🔌
    ]
    # Also include Latin extended and other common non-ASCII
    candidates += list(range(0x00c0, 0x0180))   # Latin extended A

    result = {}
    for cp in candidates:
        try:
            char = chr(cp)
            utf8_bytes = char.encode("utf-8")
            # Decode as cp1252 (what went wrong)
            try:
                mojibake_str = utf8_bytes.decode("cp1252")
            except UnicodeDecodeError:
                # Some UTF-8 byte sequences are invalid in cp1252; skip
                continue
            # Only record if the mojibake is non-trivial (multi-char or has non-ASCII)
            if mojibake_str != char:
                result[mojibake_str] = char
        except (ValueError, UnicodeEncodeError):
            continue
    return result


def fix_mojibake(txt: str, mmap: dict) -> str:
    """
    Replace all mojibake sequences in txt using mmap.
    Longest-match first (sort by key length descending).
    """
    keys_sorted = sorted(mmap.keys(), key=len, reverse=True)
    for moji, correct in zip(keys_sorted, [mmap[k] for k in keys_sorted]):
        if moji in txt:
            txt = txt.replace(moji, correct)
    return txt


if __name__ == "__main__":
    import re

    path = "ref/PROJECT_XRAY.md"
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()

    before_count = len(re.findall(r"[^\x00-\x7F]", txt))
    print(f"Non-ASCII chars before: {before_count}")

    mmap = build_mojibake_map()
    print(f"Replacement map entries: {len(mmap)}")

    fixed = fix_mojibake(txt, mmap)

    after_count = len(re.findall(r"[^\x00-\x7F]", fixed))
    print(f"Non-ASCII chars after:  {after_count}")
    print(f"Chars fixed: {before_count - after_count} → {after_count} remain")

    # Spot-check glyphs
    glyphs = [
        ("\u2502", "│  pipe"),
        ("\u251c", "├  tee"),
        ("\u2500", "─  dash"),
        ("\u2514", "└  elbow"),
        ("\u2192", "→  right arrow"),
        ("\u2014", "—  em-dash"),
        ("\u00d7", "×  times"),
        ("\u2705", "\u2705  checkmark"),
        ("\u231b", "\u231b  hourglass"),
        ("\u2713", "\u2713  tick"),
    ]
    print("\nGlyph checks after fix:")
    for char, name in glyphs:
        print(f"  {name}: {'present' if char in fixed else 'MISSING'}")

    # Verify no raw mojibake box remains (quick check)
    moji_box_raw = "\u00e2\u201d\u201a"   # mojibake │
    moji_tee_raw = "\u00e2\u201d\u0153"   # mojibake ├
    print(f"\n  Mojibake │ still present: {moji_box_raw in fixed}")
    print(f"  Mojibake ├ still present: {moji_tee_raw in fixed}")

    # Key content preserved?
    preserved = [
        "MultiHeadCNN + Cascade", "never_miss_mode", "0.464", "224 tiles",
        "8,316", "cv_folds_k5_seed42.json", "46 tests", "Run A",
        "2026-02-25", "cv5_fold0", "reports/", "Never-Miss",
        "bootstrap_tile_ci", "summarize_cascade_cv",
    ]
    print("\nContent preservation:")
    all_ok = True
    for needle in preserved:
        found = needle in fixed
        print(f"  {'OK  ' if found else 'FAIL'}: {needle!r}")
        if not found:
            all_ok = False

    print(f"\nLines: {fixed.count(chr(10))}")

    if all_ok:
        with open(path, "w", encoding="utf-8") as f:
            f.write(fixed)
        print("\nFile written successfully.")
    else:
        print("\nABORTED: content loss detected, file NOT written.")

#!/usr/bin/env python3
"""Fix remaining mojibake fragments (mainly leftover hourglass emoji sequences)."""
import re

with open('ref/PROJECT_XRAY.md', 'r', encoding='utf-8') as f:
    txt = f.read()

# Find remaining U+00E2 instances and their context
positions = [m.start() for m in re.finditer(chr(0xe2), txt)]
print(f"U+00E2 instances: {len(positions)}")
for pos in positions[:20]:
    ctx = txt[pos:pos+10]
    cps = [hex(ord(c)) for c in ctx]
    print(f"  pos {pos}: codepoints={cps}")

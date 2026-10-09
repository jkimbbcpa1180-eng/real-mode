# SPDX-License-Identifier: CC0-1.0
"""Split the v3 modules into paste parts parts/v3_partN.md (40-60 lines each, one global
numbering "Part n/N", each part from one file) and verify every file rebuilds byte-exact.
License: CC0 1.0 Universal (public domain)."""
import glob
import os
import re

LO, HI, TARGET = 40, 60, 50
INF = float("inf")
HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ("common_v3.py", "safe_return_v3.py", "mission_v3.py", "repass_v3.py", "swarm_v3.py",
         "change_v3.py", "sensors_v3.py", "thermal_v3.py")


def cuts_for(lines):
    n = len(lines)

    def cost(end):
        if end == n:
            return 0.0
        prev, nxt = lines[end - 1].strip(), lines[end].strip()
        if prev == "" and not nxt.startswith((")", "]", "}")):
            return 0.0
        if nxt.startswith(("def ", "class ", "@", "# ---", "    def ")):
            return 1.0
        return 10.0
    best, back = [INF] * (n + 1), [0] * (n + 1)
    best[0] = 0.0
    for end in range(1, n + 1):
        for size in range(LO, HI + 1):
            st = end - size
            if st < 0 or best[st] == INF:
                continue
            c = best[st] + cost(end) + 0.01 * abs(size - TARGET)
            if c < best[end]:
                best[end], back[end] = c, st
    assert best[n] < INF, "no valid split"
    out, e = [], n
    while e > 0:
        out.append((back[e], e)); e = back[e]
    return out[::-1]


def main():
    pdir = os.path.join(HERE, "parts")
    os.makedirs(pdir, exist_ok=True)
    for f in glob.glob(os.path.join(pdir, "v3_part*.md")):
        os.remove(f)
    plan = []
    for src in FILES:
        lines = open(os.path.join(HERE, src)).read().splitlines(keepends=True)
        plan += [(src, a, b, lines) for a, b in cuts_for(lines)]
    N = len(plan)
    for i, (src, a, b, lines) in enumerate(plan, 1):
        with open(os.path.join(pdir, f"v3_part{i}.md"), "w") as f:
            f.write(f"**Part {i}/{N}** ({src}, lines {a + 1}-{b})\n\n```python\n{''.join(lines[a:b])}```\n")
    rebuilt = {}
    for i in range(1, N + 1):
        txt = open(os.path.join(pdir, f"v3_part{i}.md")).read()
        src = re.match(r"\*\*Part \d+/\d+\*\* \((\S+),", txt).group(1)
        rebuilt[src] = rebuilt.get(src, "") + re.search(r"```python\n(.*)```\n$", txt, re.S).group(1)
    for src in FILES:
        assert rebuilt[src] == open(os.path.join(HERE, src)).read(), src
        sizes = [b - a for s, a, b, _ in plan if s == src]
        print(f"{src}: {sum(sizes)} lines, {len(sizes)} parts, sizes {min(sizes)}-{max(sizes)}, rebuild OK")
    print(f"total {N} parts -> parts/v3_part1..{N}.md")


if __name__ == "__main__":
    main()

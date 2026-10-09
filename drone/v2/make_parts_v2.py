"""Split the v2 modules into paste parts (40-60 lines each) and verify each rebuild. CC0.
drone_core_v2.py -> v2_part*.md, drone_uplink_v2.py -> uplink_v2_part*.md,
drone_recon_v2.py -> recon_v2_part*.md, drone_slam_v2.py -> slam_v2_part*.md"""
import glob
import os
import re

LO, HI, TARGET = 40, 60, 50
INF = float("inf")
FILES = (("drone_core_v2.py", "v2_part"), ("drone_uplink_v2.py", "uplink_v2_part"),
         ("drone_recon_v2.py", "recon_v2_part"), ("drone_slam_v2.py", "slam_v2_part"))


def split(src: str, prefix: str) -> str:
    lines = open(src).read().splitlines(keepends=True)
    n = len(lines)

    def cut_cost(end):          # cost of ending a part after line index end-1
        if end == n:
            return 0.0
        prev, nxt = lines[end - 1].strip(), lines[end].strip()
        if prev == "" and not nxt.startswith((")", "]", "}")):
            return 0.0
        if nxt.startswith(("def ", "class ", "@", "# ===", "    def ")):
            return 1.0
        return 10.0

    best = [INF] * (n + 1)
    back = [0] * (n + 1)
    best[0] = 0.0
    for end in range(1, n + 1):
        for size in range(LO, HI + 1):
            st = end - size
            if st < 0 or best[st] == INF:
                continue
            c = best[st] + cut_cost(end) + 0.01 * abs(size - TARGET)
            if c < best[end]:
                best[end], back[end] = c, st
    assert best[n] < INF, "no valid split"
    cuts, e = [], n
    while e > 0:
        cuts.append((back[e], e))
        e = back[e]
    cuts.reverse()
    for f in glob.glob(f"{prefix}[0-9]*.md"):
        os.remove(f)
    N = len(cuts)
    for i, (a, b) in enumerate(cuts, 1):
        with open(f"{prefix}{i}.md", "w") as f:
            f.write(f"**Part {i}/{N}** ({src}, lines {a + 1}-{b})\n\n```python\n{''.join(lines[a:b])}```\n")
    rebuilt = ""
    for i in range(1, N + 1):
        with open(f"{prefix}{i}.md") as f:
            txt = f.read()
        rebuilt += re.search(r"```python\n(.*)```\n$", txt, re.S).group(1)
    with open(src) as f:
        assert rebuilt == f.read(), f"rebuild mismatch for {src}"
    sizes = [b - a for a, b in cuts]
    return f"{src}: {n} lines -> {N} parts ({prefix}1..{N}.md), sizes {min(sizes)}-{max(sizes)} lines, rebuild OK"


if __name__ == "__main__":
    for src, prefix in FILES:
        print(split(src, prefix))

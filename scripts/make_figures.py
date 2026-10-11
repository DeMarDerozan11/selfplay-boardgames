"""Render the README figures as dependency-free SVG from the logged results.

Reads runs/<name>/{log.jsonl,match_results.jsonl} when present (and copies them to docs/results/,
which is committed because it is small), otherwise reads docs/results/.
"""
import json, pathlib, shutil

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES, IMG = ROOT / "docs" / "results", ROOT / "docs" / "img"
RES.mkdir(parents=True, exist_ok=True); IMG.mkdir(parents=True, exist_ok=True)
for run in ("main", "stable", "dobutsu_main"):
    for f in ("log.jsonl", "match_results.jsonl"):
        src = ROOT / "runs" / run / f
        if src.exists():
            shutil.copy(src, RES / f"{run}_{f}")


def load(run, f):
    return [json.loads(l) for l in (RES / f"{run}_{f}").read_text().splitlines()]


BLUE, ORANGE, GREY = "#1f5fbf", "#d4600a", "#6b7280"


class Chart:
    def __init__(self, title, xlabel, ylabel, xmax, ymin=0.0, ymax=1.0, w=520, h=300):
        self.t, self.xl, self.yl, self.xmax, self.ymin, self.ymax, self.w, self.h = title, xlabel, ylabel, xmax, ymin, ymax, w, h
        self.L, self.R, self.T, self.B = 52, 14, 34, 62
        self.body, self.legend = [], []

    def X(self, x): return self.L + (self.w - self.L - self.R) * x / self.xmax
    def Y(self, y): return self.h - self.B - (self.h - self.B - self.T) * (y - self.ymin) / (self.ymax - self.ymin)

    def line(self, pts, color, label, dash=False, errs=None):
        d = " ".join(f"{'M' if i == 0 else 'L'}{self.X(x):.1f},{self.Y(y):.1f}" for i, (x, y) in enumerate(pts))
        self.body.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="2"' + (' stroke-dasharray="5 4"' if dash else "") + "/>")
        for i, (x, y) in enumerate(pts):
            if errs:
                lo, hi = errs[i]
                self.body.append(f'<line x1="{self.X(x):.1f}" x2="{self.X(x):.1f}" y1="{self.Y(lo):.1f}" y2="{self.Y(hi):.1f}" stroke="{color}" stroke-width="1.2" opacity=".55"/>')
            self.body.append(f'<circle cx="{self.X(x):.1f}" cy="{self.Y(y):.1f}" r="3.2" fill="{color}"/>')
        self.legend.append((color, label, dash))

    def svg(self):
        o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" font-family="system-ui,sans-serif" font-size="11">',
             f'<rect width="{self.w}" height="{self.h}" fill="#ffffff"/>',
             f'<text x="{self.L}" y="20" font-size="13" font-weight="600" fill="#111">{self.t}</text>']
        for i in range(6):
            v = self.ymin + (self.ymax - self.ymin) * i / 5
            o.append(f'<line x1="{self.L}" x2="{self.w - self.R}" y1="{self.Y(v):.1f}" y2="{self.Y(v):.1f}" stroke="#e5e7eb"/>')
            o.append(f'<text x="{self.L - 6}" y="{self.Y(v) + 4:.1f}" text-anchor="end" fill="#444">{v:.1f}</text>')
        step = 10 if self.xmax >= 20 else 1
        for x in range(0, int(self.xmax) + 1, step):
            o.append(f'<text x="{self.X(x):.1f}" y="{self.h - self.B + 16}" text-anchor="middle" fill="#444">{x}</text>')
        o.append(f'<text x="{(self.L + self.w - self.R) / 2}" y="{self.h - self.B + 32}" text-anchor="middle" fill="#444">{self.xl}</text>')
        o.append(f'<text transform="translate(13,{(self.T + self.h - self.B) / 2}) rotate(-90)" text-anchor="middle" fill="#444">{self.yl}</text>')
        o += self.body
        for i, (c, lab, dash) in enumerate(self.legend):
            x = self.L + i * 190
            o.append(f'<line x1="{x}" x2="{x + 22}" y1="{self.h - 12}" y2="{self.h - 12}" stroke="{c}" stroke-width="2"' + (' stroke-dasharray="5 4"' if dash else "") + "/>")
            o.append(f'<text x="{x + 27}" y="{self.h - 8}" fill="#222">{lab}</text>')
        return "\n".join(o + ["</svg>"])


def save(chart, name):
    (IMG / name).write_text(chart.svg())


def match_series(rows, opp):
    best = {}  # keep the largest sample (most games) for each iteration
    for r in rows:
        if r["opponent"] == opp and (r["iter"] not in best or r["games"] >= best[r["iter"]]["games"]):
            best[r["iter"]] = r
    rs = [best[i] for i in sorted(best)]
    return [(r["iter"], r["score"]) for r in rs], [tuple(r["ci95"]) for r in rs]


# 1-2. Connect 4: strength in matches, with and without stabilisation
c4 = {"unstabilised": load("main", "match_results.jsonl"), "stabilised": load("stable", "match_results.jsonl")}
for opp, fname, title in (("minimax-d4", "c4_vs_minimax.svg", "Connect 4: score vs minimax (depth 4)"),
                          ("mcts-500", "c4_vs_mcts.svg", "Connect 4: score vs MCTS (500 simulations)")):
    ch = Chart(title, "training iteration", "score (win=1, draw=0.5)", 40)
    for (name, rows), col in zip(c4.items(), (ORANGE, BLUE)):
        pts, errs = match_series(rows, opp)
        ch.line(pts, col, name, errs=errs)
    save(ch, fname)

# 3. Connect 4: exact-solver metric during training
ch = Chart("Connect 4: moves that keep the game result (exact solver)", "training iteration", "fraction of positions", 40, 0.5, 1.0)
for run, name, col in (("main", "unstabilised", ORANGE), ("stable", "stabilised", BLUE)):
    rs = load(run, "log.jsonl")
    ch.line([(r["iter"], r["solver_preserving_search"]) for r in rs], col, name)
save(ch, "c4_solver_curve.svg")

# 4. Dobutsu Shogi: score vs MCTS during training
d = load("dobutsu_main", "match_results.jsonl")
ch = Chart("Dobutsu Shogi: score vs MCTS (500 simulations), 40 games", "training iteration", "score (win=1, draw=0.5)", 40)
rows = [r for r in d if r["opponent"] == "mcts-500" and r["games"] == 40]
pts, errs = match_series(rows, "mcts-500")
ch.line(pts, BLUE, "AlphaZero-lite", errs=errs)
save(ch, "dobutsu_vs_mcts.svg")
print("figures written to", IMG)

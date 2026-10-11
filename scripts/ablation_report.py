"""Aggregate the ablation grid: per configuration, mean and spread over seeds, plus pooled counts.

Writes docs/results/ablation.json, docs/results/ablation.md and docs/img/ablation.svg.
"""
import json, math, pathlib, statistics, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.engine.match import wilson

ROOT = pathlib.Path(__file__).resolve().parents[1]
ORDER = ["sims25", "base_sims100_net64x3", "sims300", "net16x1", "net32x2", "opponent_data"]
LABEL = {"sims25": "25 simulations", "base_sims100_net64x3": "100 simulations, net 64x3 (reference)",
         "sims300": "300 simulations", "net16x1": "net 16x1", "net32x2": "net 32x2",
         "opponent_data": "trained on minimax-2 games"}
OPPS = ["minimax-d4", "mcts-500"]

rows = {}  # (config, opp) -> list of (seed, wins, draws, losses)
for d in sorted((ROOT / "runs").glob("abl_*")):
    cfg, seed = d.name[4:].rsplit("_s", 1)
    f = d / "match_results.jsonl"
    if not f.exists():
        continue
    for line in f.read_text().splitlines():
        r = json.loads(line)
        if r["opponent"] in OPPS:
            rows.setdefault((cfg, r["opponent"]), []).append((int(seed), r["wins"], r["draws"], r["losses"]))

out, md = {}, ["| Configuration | Seeds | vs minimax-4: mean score (per-seed range) | vs MCTS-500: mean score (per-seed range) |", "|---|---|---|---|"]
for cfg in ORDER:
    cell, nseeds = [], 0
    for opp in OPPS:
        rs = rows.get((cfg, opp), [])
        nseeds = max(nseeds, len(rs))
        if not rs:
            cell.append("-"); continue
        scores = [(w + d / 2) / (w + d + l) for _, w, d, l in rs]
        W, D, L = (sum(r[i] for r in rs) for i in (1, 2, 3))
        lo, hi = wilson(W + D / 2, W + D + L)
        out.setdefault(cfg, {})[opp] = dict(seeds=len(rs), mean=statistics.mean(scores), sd=statistics.pstdev(scores) if len(rs) > 1 else None,
                                            per_seed=scores, pooled=dict(W=W, D=D, L=L), pooled_ci95=[lo, hi])
        cell.append(f"{statistics.mean(scores):.2f} ({min(scores):.2f}-{max(scores):.2f})")
    md.append(f"| {LABEL[cfg]} | {nseeds} | {cell[0]} | {cell[1]} |")
(ROOT / "docs" / "results" / "ablation.json").write_text(json.dumps(out, indent=1))
(ROOT / "docs" / "results" / "ablation.md").write_text("\n".join(md) + "\n")
print("\n".join(md))

# bar chart: mean score per configuration, one dot per seed
W_, H_, L_, B_ = 760, 340, 56, 110
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W_} {H_}" width="{W_}" font-family="system-ui,sans-serif" font-size="11">',
       f'<rect width="{W_}" height="{H_}" fill="#fff"/>',
       '<text x="56" y="20" font-size="13" font-weight="600" fill="#111">Connect 4 ablations: score after 20 iterations (dots = seeds)</text>']
y = lambda v: H_ - B_ - (H_ - B_ - 34) * v
for i in range(6):
    v = i / 5
    svg.append(f'<line x1="{L_}" x2="{W_-10}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="#e5e7eb"/><text x="{L_-6}" y="{y(v)+4:.1f}" text-anchor="end" fill="#444">{v:.1f}</text>')
slot = (W_ - L_ - 10) / len(ORDER)
for i, cfg in enumerate(ORDER):
    cx = L_ + slot * (i + 0.5)
    for j, (opp, col) in enumerate(zip(OPPS, ("#1f5fbf", "#d4600a"))):
        c = out.get(cfg, {}).get(opp)
        if not c:
            continue
        x0 = cx - 34 + j * 36
        svg.append(f'<rect x="{x0:.1f}" y="{y(c["mean"]):.1f}" width="30" height="{y(0)-y(c["mean"]):.1f}" fill="{col}" opacity=".35"/>')
        for k, s in enumerate(c["per_seed"]):
            svg.append(f'<circle cx="{x0 + 8 + k * 7:.1f}" cy="{y(s):.1f}" r="2.8" fill="{col}"/>')
    words = LABEL[cfg].split(" (")[0]
    svg.append(f'<text x="{cx:.1f}" y="{H_-B_+16}" text-anchor="middle" fill="#222" transform="rotate(25 {cx:.1f} {H_-B_+16})">{words}</text>')
svg.append(f'<rect x="{L_}" y="{H_-22}" width="12" height="12" fill="#1f5fbf" opacity=".5"/><text x="{L_+17}" y="{H_-12}" fill="#222">vs minimax depth 4</text>')
svg.append(f'<rect x="{L_+150}" y="{H_-22}" width="12" height="12" fill="#d4600a" opacity=".5"/><text x="{L_+167}" y="{H_-12}" fill="#222">vs MCTS 500 simulations</text></svg>')
(ROOT / "docs" / "img" / "ablation.svg").write_text("\n".join(svg))

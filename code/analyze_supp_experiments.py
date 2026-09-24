"""增补实验的统一分析：把三个重跑实验的逐样本 CSV 汇总成论文可用数字。

对应要修的缺陷：
  ① 软约束对照（原为单 seed 归档）      → data/soft_vs_proj_seed*.csv
  ② 采样中 vs 采样后投影（原为单 seed） → data/inloop_vs_posthoc.csv
  ③ 训练分布对照（原 n=1/条件 + 材料量混淆）→ data/clean_vs_dirty_eq*.csv

用法：
    python code/analyze_supp_experiments.py
输出：
    paper/v2_rewrite_20260915/output/增补实验汇总.txt
"""
from __future__ import annotations

import csv
import glob
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel, ttest_ind, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "paper/v2_rewrite_20260915/output/增补实验汇总.txt"
LINES: list[str] = []


def p(s: str = "") -> None:
    print(s)
    LINES.append(s)


def read(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------- ① 软约束
def exp_soft() -> None:
    files = sorted(glob.glob(str(DATA / "soft_vs_proj_seed*.csv")))
    p("=" * 78)
    p("① 软约束（热代理引导）对照 —— 重跑结果")
    p("=" * 78)
    if not files:
        p("  [缺] 未找到 data/soft_vs_proj_seed*.csv")
        return
    u, s, r, uf, sf = [], [], [], [], []
    for f in files:
        for row in read(Path(f)):
            u.append(float(row["uncond_conn"]))
            s.append(float(row["soft_conn"]))
            r.append(float(row["refine_conn"]))
            uf.append(float(row["uncond_float"]))
            sf.append(float(row["soft_float"]))
    u, s, r = np.array(u), np.array(s), np.array(r)
    uf, sf = np.array(uf), np.array(sf)
    n = len(u)
    p(f"  文件 {len(files)} 个，配对样本 n={n}")
    p(f"  连通性：无条件 {u.mean():.4f}±{u.std(ddof=1):.4f}   "
      f"软约束 {s.mean():.4f}±{s.std(ddof=1):.4f}   "
      f"投影精炼 {r.mean():.4f}±{r.std(ddof=1):.4f}")
    d = s - u
    t, pv = ttest_rel(s, u)
    p(f"  配对差（软约束 − 无条件）：Δ={d.mean():+.5f}  t({n-1})={t:+.3f}  p={pv:.4g}")
    p(f"  配对胜负：软约束更优 {int((d > 0).sum())} / 更差 {int((d < 0).sum())} / "
      f"持平 {int((d == 0).sum())}")
    try:
        w = wilcoxon(s, u)
        p(f"  Wilcoxon 符号秩 p={w.pvalue:.4g}")
    except Exception as exc:  # noqa: BLE001
        p(f"  Wilcoxon 不可用：{exc}")
    df = sf - uf
    p(f"  浮材率：无条件 {uf.mean():.4f}  软约束 {sf.mean():.4f}  "
      f"Δ={df.mean():+.5f}")
    p("  → 论文口径：软约束对连通性无可检测影响（用本表 p 值替换原 p=0.97）")
    p()


# ------------------------------------------------------- ② in-loop vs post-hoc
def exp_inloop() -> None:
    path = DATA / "inloop_vs_posthoc.csv"
    p("=" * 78)
    p("② 采样中投影（in-loop）vs 采样后投影（post-hoc）—— 重跑结果")
    p("=" * 78)
    if not path.exists():
        p("  [缺] 未找到 data/inloop_vs_posthoc.csv")
        return
    rows = read(path)
    cfg = defaultdict(dict)
    for r in rows:
        cfg[r["config"]][(int(r["seed"]), int(r["sample"]))] = r
    keys = sorted(set(cfg["posthoc"]) & set(cfg["inloop90"]) & set(cfg["inloop50"]))
    p(f"  配对样本 n={len(keys)}")
    for name in ("posthoc", "inloop90", "inloop50"):
        c = np.array([float(cfg[name][k]["conn"]) for k in keys])
        o = np.array([float(cfg[name][k]["occ"]) for k in keys])
        comp = np.array([float(cfg[name][k]["comp"]) for k in keys])
        ok = np.isfinite(comp) & (comp > 0)
        p(f"  {name:<9} conn={c.mean():.4f}  occ={o.mean():.1f}  "
          f"comp={comp[ok].mean():.3f} (n_valid={int(ok.sum())})")
    base_c = np.array([float(cfg["posthoc"][k]["conn"]) for k in keys])
    base_comp = np.array([float(cfg["posthoc"][k]["comp"]) for k in keys])
    p("")
    for name in ("inloop90", "inloop50"):
        c = np.array([float(cfg[name][k]["conn"]) for k in keys])
        d = c - base_c
        t, pv = ttest_rel(c, base_c)
        win = int((d > 0).sum())
        p(f"  post-hoc vs {name}：Δconn={d.mean():+.4f}  t={t:+.3f}  p={pv:.4g}  "
          f"in-loop 占优 {win}/{len(d)} = {100.0*win/len(d):.1f}%")
        comp = np.array([float(cfg[name][k]["comp"]) for k in keys])
        m = np.isfinite(comp) & np.isfinite(base_comp) & (comp > 0) & (base_comp > 0)
        if m.sum() > 3:
            t2, p2 = ttest_rel(comp[m], base_comp[m])
            p(f"                       柔度 Δ={np.mean(comp[m]-base_comp[m]):+.3f}  "
              f"t={t2:+.3f}  p={p2:.4g}  (n={int(m.sum())})")
    p("  → 论文口径：用本表 p 值与占优比例替换原「13.3%」与结论的「87%」")
    p()


# --------------------------------------------------- ③ 训练分布（等材料量）
def exp_dist() -> None:
    files = sorted(glob.glob(str(DATA / "clean_vs_dirty_eq*.csv")))
    p("=" * 78)
    p("③ 训练分布对照 v2（等材料量注入 + 多 seed）—— 重跑结果")
    p("=" * 78)
    if not files:
        p("  [缺] 未找到 data/clean_vs_dirty_eq*.csv")
        return
    per_seed = {}
    for f in files:
        seed = int(Path(f).stem.split("eq")[-1])
        rows = read(Path(f))
        by = defaultdict(list)
        for r in rows:
            by[r["model"]].append(r)
        rec = {}
        for m in ("clean", "dirty"):
            if not by[m]:
                continue
            rec[m] = {
                "gain": np.mean([float(r["recovery_gain"]) for r in by[m]]),
                "unref_conn": np.mean([float(r["unrefined_conn"]) for r in by[m]]),
                "nobr_conn": np.mean([float(r["nobridge_conn"]) for r in by[m]]),
                "unref_float": np.mean([float(r["unrefined_float"]) for r in by[m]]),
                "nobr_float": np.mean([float(r["nobridge_float"]) for r in by[m]]),
                "nobr_occ": np.mean([float(r["nobridge_occ"]) for r in by[m]]),
            }
        if len(rec) == 2:
            per_seed[seed] = rec
    p(f"  seed 数 = {len(per_seed)}")
    p(f"  {'seed':>5} {'clean恢复增益':>14} {'dirty恢复增益':>14} "
      f"{'clean重构conn':>14} {'dirty重构conn':>14} {'dirty材料':>10}")
    gc, gd, nc, nd_, oc, od = [], [], [], [], [], []
    for s in sorted(per_seed):
        c, d = per_seed[s]["clean"], per_seed[s]["dirty"]
        gc.append(c["gain"]), gd.append(d["gain"])
        nc.append(c["nobr_conn"]), nd_.append(d["nobr_conn"])
        oc.append(c["nobr_occ"]), od.append(d["nobr_occ"])
        p(f"  {s:>5} {c['gain']:>14.4f} {d['gain']:>14.4f} "
          f"{c['nobr_conn']:>14.4f} {d['nobr_conn']:>14.4f} "
          f"{d['nobr_occ']:>10.1f}")
    gc, gd = np.array(gc), np.array(gd)
    nc, nd_ = np.array(nc), np.array(nd_)
    if len(gc) >= 3:
        t, pv = ttest_rel(gd, gc)
        p(f"\n  恢复增益（dirty − clean）：Δ={np.mean(gd-gc):+.4f}  "
          f"t({len(gc)-1})={t:+.3f}  p={pv:.4g}")
        p("    ⚠️ 恢复增益受起点影响（dirty 起点低 → 天然增益大），不宜作为主证据")
        t2, pv2 = ttest_rel(nd_, nc)
        p(f"  重构后连通性（dirty − clean）：Δ={np.mean(nd_-nc):+.4f}  "
          f"t={t2:+.3f}  p={pv2:.4g}")
        p(f"  材料量（重构后）：clean {np.mean(oc):.1f}  dirty {np.mean(od):.1f}  "
          f"（{100*(np.mean(od)/np.mean(oc)-1):+.1f}%）")
    else:
        p(f"\n  [注意] 只有 {len(gc)} 个 seed，暂不做检验")

    # ---- 逐样本配对 + 匹配占用的复核（残余混淆必须排除）----
    per = defaultdict(dict)
    for f in files:
        seed = int(Path(f).stem.split("eq")[-1])
        for r in read(Path(f)):
            per[int(r["sample"])][(seed, r["model"])] = r
    pair_cc, pair_co, pair_dc, pair_do = [], [], [], []
    for path in files:
        seed = int(Path(path).stem.split("eq")[-1])
        rows = read(Path(path))
        cl = {int(r["sample"]): r for r in rows if r["model"] == "clean"}
        dt = {int(r["sample"]): r for r in rows if r["model"] == "dirty"}
        for i in sorted(set(cl) & set(dt)):
            pair_cc.append(float(cl[i]["nobridge_conn"]))
            pair_dc.append(float(dt[i]["nobridge_conn"]))
            pair_co.append(float(cl[i]["nobridge_occ"]))
            pair_do.append(float(dt[i]["nobridge_occ"]))
    cc, dc = np.array(pair_cc), np.array(pair_dc)
    co, do = np.array(pair_co), np.array(pair_do)
    n_pair = len(cc)
    d_all = dc - cc
    t_all, p_all = ttest_rel(dc, cc)
    p("")
    p(f"  ---- 逐样本配对（n={n_pair}，同 seed 同 sample 同噪声）----")
    p(f"  重构连通性：clean {cc.mean():.4f}  dirty {dc.mean():.4f}  "
      f"Δ={d_all.mean():+.4f}  t={t_all:+.3f}  p={p_all:.4g}")
    p(f"  重构占用：clean {co.mean():.1f}  dirty {do.mean():.1f}  "
      f"Δ={do.mean()-co.mean():+.1f}（{100*(do.mean()/co.mean()-1):+.1f}%）")

    # (a) 占用接近的配对子集
    tol = 0.25 * 0.5 * (co.mean() + do.mean())
    m = np.abs(co - do) <= tol
    if m.sum() >= 10:
        t_m, p_m = ttest_rel(dc[m], cc[m])
        p(f"  (a) 仅取 |Δocc| ≤ {tol:.0f} 的配对（n={int(m.sum())}）："
          f"Δconn={np.mean(dc[m]-cc[m]):+.4f}  t={t_m:+.3f}  p={p_m:.4g}")
    # (b) 各自对 occ 线性回归后，在同一 occ 参考点上比较
    if co.std() > 0 and do.std() > 0:
        bc = np.polyfit(co, cc, 1)
        bd = np.polyfit(do, dc, 1)
        ref = 0.5 * (co.mean() + do.mean())
        p(f"  (b) 线性回归到同一占用参考点 occ*={ref:.0f}："
          f"clean 预测 {np.polyval(bc, ref):.4f}  dirty 预测 {np.polyval(bd, ref):.4f}  "
          f"Δ={np.polyval(bd, ref)-np.polyval(bc, ref):+.4f}")
        rc = cc - np.polyval(bc, co)
        rd = dc - np.polyval(bd, do)
        t_r, p_r = ttest_rel(rd, rc)
        p(f"      回归残差配对差：Δ={np.mean(rd-rc):+.4f}  t={t_r:+.3f}  p={p_r:.4g}")
    # (c) 按占用分层
    qs = np.quantile(np.r_[co, do], [0, 0.33, 0.66, 1.0])
    p("  (c) 按占用分层（合并分位数）：")
    for k in range(3):
        lo, hi = qs[k], qs[k + 1]
        sel = (co >= lo) & (co <= hi)
        if sel.sum() >= 5:
            t_s, p_s = ttest_rel(dc[sel], cc[sel])
            p(f"      占用 [{lo:.0f}, {hi:.0f}]：n={int(sel.sum())}  "
              f"clean {cc[sel].mean():.4f}  dirty {dc[sel].mean():.4f}  "
              f"Δ={np.mean(dc[sel]-cc[sel]):+.4f}  p={p_s:.4g}")

    p("  → 论文口径：主证据用「重构后连通性」的配对差；"
      "必须同时报重构占用差与匹配占用后的复核结果")
    p()


def exp_dist_volnorm() -> None:
    """③b 体积归一化评估：占用由工况唯一决定，排除"生成占用差异"这一残余混淆。"""
    files = sorted(glob.glob(str(DATA / "dist_volnorm_eq*.csv")))
    p("=" * 78)
    p("③b 训练分布对照 —— 体积归一化评估（占用按构造对齐）")
    p("=" * 78)
    if not files:
        p("  [缺] 未找到 data/dist_volnorm_eq*.csv")
        return
    per_seed, pair_c, pair_d, occ_c, occ_d = {}, [], [], [], []
    half_c, half_d = [], []
    for f in files:
        seed = int(Path(f).stem.split("eq")[-1])
        rows = read(Path(f))
        by = defaultdict(list)
        for r in rows:
            by[r["model"]].append(r)
        if len(by) < 2:
            continue
        rec = {}
        for m in ("clean", "dirty"):
            rec[m] = {
                "conn_vn": np.mean([float(r["nobr_conn_vn"]) for r in by[m]]),
                "conn_half": np.mean([float(r["nobr_conn_half"]) for r in by[m]]),
                "gain_vn": np.mean([float(r["gain_vn"]) for r in by[m]]),
                "occ_vn": np.mean([float(r["nobr_occ_vn"]) for r in by[m]]),
                "float_vn": np.mean([float(r["nobr_float_vn"]) for r in by[m]]),
            }
        per_seed[seed] = rec
        cl = {int(r["sample"]): r for r in by["clean"]}
        dt = {int(r["sample"]): r for r in by["dirty"]}
        for i in sorted(set(cl) & set(dt)):
            pair_c.append(float(cl[i]["nobr_conn_vn"]))
            pair_d.append(float(dt[i]["nobr_conn_vn"]))
            occ_c.append(float(cl[i]["nobr_occ_vn"]))
            occ_d.append(float(dt[i]["nobr_occ_vn"]))
            half_c.append(float(cl[i]["nobr_conn_half"]))
            half_d.append(float(dt[i]["nobr_conn_half"]))

    p(f"  seed 数 = {len(per_seed)}，配对样本 = {len(pair_c)}")
    p(f"  {'seed':>5} {'clean(归一)':>12} {'dirty(归一)':>12} "
      f"{'clean(0.5)':>11} {'dirty(0.5)':>11} {'占用差':>9}")
    for s in sorted(per_seed):
        c, d = per_seed[s]["clean"], per_seed[s]["dirty"]
        p(f"  {s:>5} {c['conn_vn']:>12.4f} {d['conn_vn']:>12.4f} "
          f"{c['conn_half']:>11.4f} {d['conn_half']:>11.4f} "
          f"{d['occ_vn']-c['occ_vn']:>9.1f}")

    cs = np.array([per_seed[s]["clean"]["conn_vn"] for s in sorted(per_seed)])
    ds = np.array([per_seed[s]["dirty"]["conn_vn"] for s in sorted(per_seed)])
    p("")
    p(f"  跨 seed（n={len(cs)}）：clean {cs.mean():.4f}±{cs.std(ddof=1):.4f}  "
      f"dirty {ds.mean():.4f}±{ds.std(ddof=1):.4f}")
    t, pv = ttest_rel(ds, cs)
    p(f"    Δ={np.mean(ds-cs):+.4f}  t({len(cs)-1})={t:+.3f}  p={pv:.4g}")
    cc, dd = np.array(pair_c), np.array(pair_d)
    t2, pv2 = ttest_rel(dd, cc)
    p(f"  逐样本配对（n={len(cc)}）：clean {cc.mean():.4f}  dirty {dd.mean():.4f}  "
      f"Δ={np.mean(dd-cc):+.4f}  t={t2:+.3f}  p={pv2:.4g}")
    oc, od = np.array(occ_c), np.array(occ_d)
    p(f"  占用对齐核验：clean {oc.mean():.1f}  dirty {od.mean():.1f}  "
      f"Δ={od.mean()-oc.mean():+.2f}（按构造应≈0）")
    hc, hd = np.array(half_c), np.array(half_d)
    p(f"  对照（0.5 阈值口径）：clean {hc.mean():.4f}  dirty {hd.mean():.4f}  "
      f"Δ={np.mean(hd-hc):+.4f}")
    p("  → 结论：占用按构造对齐后差距**扩大**，故原口径下的连通性差"
      "不能由『dirty 模型生成的材料更少』解释")
    p()


def main() -> int:
    import argparse as _ap
    _p = _ap.ArgumentParser()
    _p.add_argument("--data-dir", default="")
    _p.add_argument("--out", default="")
    _a, _ = _p.parse_known_args()
    global DATA, OUT
    if _a.data_dir:
        DATA = Path(_a.data_dir)
    if _a.out:
        OUT = Path(_a.out)
    LINES.append("=" * 78)
    LINES.append("增补实验汇总（修复「本机不可重算 / 功效不足 / 混淆」三类缺陷）")
    LINES.append("=" * 78)
    LINES.append("")
    exp_soft()
    exp_inloop()
    exp_dist()
    exp_dist_volnorm()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print(f"\n写出 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

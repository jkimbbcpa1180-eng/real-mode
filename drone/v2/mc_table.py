"""Prints the MC comparison table from mc_results_v2.json. CC0."""
import json
r = json.load(open("mc_results_v2.json"))
print(f"seed {r['seed']}, {r['n_flights']} flights per variant, total runtime {r.get('total_runtime_s')} s\n")
hdr = "| variant | collisions | min clearance median / p5 / min (m) | pos err median / p95 (m) | NEES mean (ideal 3) | NEES<=7.81 (ideal 0.95) | runtime s |"
print(hdr); print("|" + "---|" * 7)
for k, v in r["runs"].items():
    a = v["all"]
    mc = a["min_clearance_m"]
    print(f"| {k} | {a['collision_rate']:.1%} | {mc['median']} / {mc['p5']} / {mc['min']} | "
          f"{a['pos_err_m']['median']} / {a['pos_err_m']['p95']} | {a['nees_mean_expect_3']} | "
          f"{a['frac_nees_le_7.81_expect_0.95']} | {v['runtime_s']} |")
print("\nBy GPS sigma (collisions; pos err median; NEES mean; NEES<=7.81):")
for k, v in r["runs"].items():
    row = []
    for g, a in v["by_gps_sigma"].items():
        row.append(f"{g} m: {a['collision_rate']:.0%}, {a['pos_err_m']['median']}, {a['nees_mean_expect_3']}, {a['frac_nees_le_7.81_expect_0.95']}")
    print(f"- {k}: " + " | ".join(row))
print("\nBy speed and obstacle (collision rate):")
for k, v in r["runs"].items():
    print(f"- {k}: v0<=9 {v['v0_le_9']['collision_rate']:.0%} (n={v['v0_le_9']['flights']}), "
          f"v0>9 {v['v0_gt_9']['collision_rate']:.0%} (n={v['v0_gt_9']['flights']}), "
          f"wall {v['by_kind']['wall']['collision_rate']:.0%}, pole {v['by_kind']['pole']['collision_rate']:.0%}")

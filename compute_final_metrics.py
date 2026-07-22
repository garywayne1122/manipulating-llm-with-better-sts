"""
Ye script results/eval.json se real Top-k, NDCG@5, MRR, aur paired t-test
nikalta hai — seedha Colab mein chalao, jahan results/eval.json ban chuka hai.
"""
import json
import math
import statistics
from scipy import stats as scipy_stats

with open("results/eval.json", "r") as f:
    data = json.load(f)

baseline_ranks = data["rank_list"]
sts_ranks = data["rank_list_opt"]

def compute_all_metrics(ranks):
    n = len(ranks)
    metrics = {}
    for k in (1, 3, 5):
        metrics[f"Top-{k} Frequency"] = round(sum(1 for r in ranks if r <= k) / n, 4)
    metrics["NDCG@5"] = round(sum((1 / math.log2(r + 1)) if r <= 5 else 0 for r in ranks) / n, 4)
    metrics["MRR"] = round(sum(1 / r for r in ranks) / n, 4)
    metrics["Mean Rank"] = round(statistics.mean(ranks), 2)
    metrics["Median Rank"] = round(statistics.median(ranks), 2)
    metrics["Std Dev"] = round(statistics.stdev(ranks), 2) if n > 1 else 0.0
    return metrics

baseline_metrics = compute_all_metrics(baseline_ranks)
sts_metrics = compute_all_metrics(sts_ranks)
t_stat, p_value = scipy_stats.ttest_rel(baseline_ranks, sts_ranks)

print("=" * 60)
print("FINAL SUMMARY — Table I ke liye")
print("=" * 60)
print(f"N trials: {len(baseline_ranks)}")
print(f"{'Metric':<20}{'Baseline':<15}{'STS-Optimized':<15}")
for key in baseline_metrics:
    print(f"{key:<20}{baseline_metrics[key]:<15}{sts_metrics[key]:<15}")
print(f"\nPaired t-test: t={t_stat:.4f}, p={p_value:.6f}")
print(f"\nRank Advantage (from evaluate.py): {data['advantage']}")

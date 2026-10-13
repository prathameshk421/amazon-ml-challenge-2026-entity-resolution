"""Bar chart of hard-negative AUC per signal. Writes reports/signal_auc.png."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("artifacts/metrics/eda_signals.json"))["signals"]
sigs = json.load(open("artifacts/metrics/eda_signals.json"))["ranked_by_hard_auc"]
vals = [d[s]["auc_hard"] for s in sigs]
plt.figure(figsize=(9, 4))
plt.barh(sigs, vals)
plt.xlabel("ROC-AUC (true match vs hard negative)")
plt.xlim(0.5, 1.0)
plt.tight_layout()
plt.savefig("reports/signal_auc.png", dpi=100)
print("WROTE reports/signal_auc.png")

from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(BASE / "data" / "customers.csv")
X = df[["AnnualIncome", "SpendingScore"]]
X_scaled = StandardScaler().fit_transform(X)

scores = {}
for k in range(2, 6):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    scores[k] = silhouette_score(X_scaled, labels)

best_k = max(scores, key=scores.get)
model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["Cluster"] = model.fit_predict(X_scaled)

df.to_csv(OUT / "segmented_customers.csv", index=False)
df.groupby("Cluster")[["Age","AnnualIncome","SpendingScore"]].mean().round(2).to_csv(
    OUT / "cluster_summary.csv"
)

plt.figure(figsize=(8,5))
for c in sorted(df["Cluster"].unique()):
    p = df[df["Cluster"] == c]
    plt.scatter(p["AnnualIncome"], p["SpendingScore"], label=f"Cluster {c}")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation using K-Means")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "customer_clusters.png", dpi=160)
plt.close()

plt.figure(figsize=(7,5))
plt.plot(list(scores), list(scores.values()), marker="o")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("K Selection using Silhouette Score")
plt.tight_layout()
plt.savefig(OUT / "silhouette_scores.png", dpi=160)
plt.close()

(OUT/"project_results.txt").write_text(
    f"Best K: {best_k}\nSilhouette Score: {scores[best_k]:.4f}\n",
    encoding="utf-8"
)
print("PROJECT COMPLETED SUCCESSFULLY!")
print(f"Best K: {best_k}")
print(f"Silhouette Score: {scores[best_k]:.4f}")

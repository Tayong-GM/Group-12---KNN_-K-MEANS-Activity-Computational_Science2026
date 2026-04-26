# =========================
# K-MEANS CLUSTERING (AUTOS DATASET)
# No sklearn, no pandas, no scipy — pure Python + matplotlib only
# =========================

import csv
import math
import random
import matplotlib.pyplot as plt

# -------------------------
# STEP 1: LOAD DATA
# -------------------------
FEATURES = [
    'wheel_base', 'length', 'width', 'height',
    'curb_weight', 'engine_size', 'horsepower',
    'city_mpg', 'highway_mpg', 'price'
]

def load_data(filepath):
    """Load CSV and return only rows with all selected features as floats."""
    data = []
    with open(filepath, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                point = [float(row[feat]) for feat in FEATURES]
                data.append(point)
            except (ValueError, KeyError):
                pass  # Skip rows with missing/non-numeric values
    return data

data = load_data('autos-k-means.csv')  # <-- update path if needed
print(f"Loaded {len(data)} rows after cleaning.\n")

# -------------------------
# STEP 2: STANDARD SCALING (Z-SCORE)
# -------------------------
def compute_stats(data):
    """Compute mean and std for each feature column."""
    n_features = len(data[0])
    n = len(data)
    means = [sum(row[j] for row in data) / n for j in range(n_features)]
    stds = [
        math.sqrt(sum((row[j] - means[j]) ** 2 for row in data) / n)
        for j in range(n_features)
    ]
    # Avoid division by zero
    stds = [s if s > 0 else 1.0 for s in stds]
    return means, stds

def scale(data, means, stds):
    """Apply z-score normalization."""
    return [
        [(row[j] - means[j]) / stds[j] for j in range(len(row))]
        for row in data
    ]

means, stds = compute_stats(data)
X_scaled = scale(data, means, stds)

# Print before/after summary for 'price' (last column, index 9) as a sanity check
prices_raw = [row[9] for row in data]
prices_scaled = [row[9] for row in X_scaled]
print("=== BEFORE SCALING (price) ===")
print(f"  Min={min(prices_raw):.2f}, Max={max(prices_raw):.2f}, "
      f"Mean={sum(prices_raw)/len(prices_raw):.2f}")
print("=== AFTER SCALING (price) ===")
print(f"  Min={min(prices_scaled):.4f}, Max={max(prices_scaled):.4f}, "
      f"Mean={sum(prices_scaled)/len(prices_scaled):.4f}\n")

# -------------------------
# STEP 3: K-MEANS FROM SCRATCH
# -------------------------
def euclidean_distance(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))

def assign_clusters(data, centroids):
    """Assign each point to the nearest centroid."""
    assignments = []
    for point in data:
        distances = [euclidean_distance(point, c) for c in centroids]
        assignments.append(distances.index(min(distances)))
    return assignments

def update_centroids(data, assignments, k):
    """Recompute centroids as the mean of assigned points."""
    n_features = len(data[0])
    centroids = []
    for cluster_id in range(k):
        cluster_points = [data[i] for i in range(len(data)) if assignments[i] == cluster_id]
        if cluster_points:
            centroid = [
                sum(p[j] for p in cluster_points) / len(cluster_points)
                for j in range(n_features)
            ]
        else:
            # If a cluster is empty, reinitialize randomly
            centroid = random.choice(data)
        centroids.append(centroid)
    return centroids

def compute_inertia(data, assignments, centroids):
    """Sum of squared distances of each point to its assigned centroid."""
    return sum(
        euclidean_distance(data[i], centroids[assignments[i]]) ** 2
        for i in range(len(data))
    )

def kmeans(data, k, max_iters=300, tol=1e-4, random_state=42):
    """Run K-Means and return (assignments, centroids, inertia)."""
    random.seed(random_state)
    # Initialize centroids by picking k random distinct points
    centroids = random.sample(data, k)

    for _ in range(max_iters):
        assignments = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, assignments, k)

        # Check convergence (max centroid shift)
        shift = max(
            euclidean_distance(centroids[i], new_centroids[i])
            for i in range(k)
        )
        centroids = new_centroids
        if shift < tol:
            break

    inertia = compute_inertia(data, assignments, centroids)
    return assignments, centroids, inertia

# -------------------------
# STEP 4: ELBOW METHOD
# -------------------------
inertias = []
k_range = range(1, 11)

for k in k_range:
    _, _, inertia = kmeans(X_scaled, k)
    inertias.append(inertia)
    print(f"k={k:2d}  Inertia={inertia:.4f}")

plt.figure()
plt.plot(list(k_range), inertias, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.tight_layout()
plt.show()

# -------------------------
# STEP 5: APPLY FINAL K-MEANS (k=3)
# -------------------------
K = 3
assignments, centroids, inertia = kmeans(X_scaled, K)

# -------------------------
# STEP 6: CLUSTER SUMMARY
# -------------------------
print(f"\n=== Cluster Summary (k={K}) ===")
print(f"{'Feature':<15}", end="")
for c in range(K):
    print(f"  Cluster {c}", end="")
print()

feat_idx = {feat: i for i, feat in enumerate(FEATURES)}

for feat in FEATURES:
    j = feat_idx[feat]
    print(f"{feat:<15}", end="")
    for c in range(K):
        cluster_vals = [data[i][j] for i in range(len(data)) if assignments[i] == c]
        cluster_mean = sum(cluster_vals) / len(cluster_vals) if cluster_vals else float('nan')
        print(f"  {cluster_mean:>9.2f}", end="")
    print()

# Print cluster sizes
print(f"\n{'Cluster Size':<15}", end="")
for c in range(K):
    size = sum(1 for a in assignments if a == c)
    print(f"  {size:>9}", end="")
print()

# -------------------------
# STEP 7: VISUALIZATION — Horsepower vs Price
# -------------------------
# Extract raw horsepower (index 6) and price (index 9)
hp_idx = FEATURES.index('horsepower')
price_idx = FEATURES.index('price')

colors = ['#e74c3c', '#3498db', '#2ecc71']  # one color per cluster
labels = [f'Cluster {c}' for c in range(K)]

plt.figure()
for c in range(K):
    x = [data[i][hp_idx] for i in range(len(data)) if assignments[i] == c]
    y = [data[i][price_idx] for i in range(len(data)) if assignments[i] == c]
    plt.scatter(x, y, c=colors[c], label=labels[c], alpha=0.7, edgecolors='k', linewidths=0.4)

plt.xlabel('Horsepower')
plt.ylabel('Price')
plt.title(f'K-Means Clustering (k={K}) — Horsepower vs Price')
plt.legend()
plt.tight_layout()
plt.show()
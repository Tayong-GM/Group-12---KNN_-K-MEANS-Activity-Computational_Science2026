import csv
import math
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import Counter

# ── 1. LOAD CSV ───────────────────────────────────────────────────────────────
def load_csv(filepath):
    with open(filepath, newline='') as f:
        reader = csv.DictReader(f)
        return [row for row in reader]

# ── 2. STANDARDIZE (z-score, returns scaled list + stats) ────────────────────
def standardize_col(values):
    mean = sum(values) / len(values)
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
    if std == 0:
        std = 1
    return [(v - mean) / std for v in values], mean, std

def scale_value(v, mean, std):
    return (v - mean) / std

# ── 3. KNN ────────────────────────────────────────────────────────────────────
def euclidean_2d(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def knn_predict_2d(train_x, train_y, train_labels, qx, qy, k):
    distances = [
        (euclidean_2d(qx, qy, train_x[i], train_y[i]), train_labels[i])
        for i in range(len(train_x))
    ]
    distances.sort(key=lambda d: d[0])
    k_labels = [label for _, label in distances[:k]]
    return Counter(k_labels).most_common(1)[0][0]

# ── 4. LOAD & PREP DATA ───────────────────────────────────────────────────────
data = load_csv('diabetes-k-nn.csv')

glucose_raw = [float(row['Glucose']) for row in data]
bmi_raw     = [float(row['BMI'])     for row in data]
labels      = [int(row['Outcome'])   for row in data]

glucose_scaled, g_mean, g_std = standardize_col(glucose_raw)
bmi_scaled,     b_mean, b_std = standardize_col(bmi_raw)

# ── 5. BUILD MESH GRID ────────────────────────────────────────────────────────
k = 5
h = 0.15  # step size (smaller = smoother but slower)

x_min = min(glucose_scaled) - 1
x_max = max(glucose_scaled) + 1
y_min = min(bmi_scaled)     - 1
y_max = max(bmi_scaled)     + 1

# Generate grid points
x_steps = []
val = x_min
while val <= x_max:
    x_steps.append(val)
    val += h

y_steps = []
val = y_min
while val <= y_max:
    y_steps.append(val)
    val += h

print(f"Grid size: {len(x_steps)} x {len(y_steps)} = {len(x_steps)*len(y_steps)} points")
print("Predicting across mesh grid... (takes a bit)")

# Predict each grid point
grid_z = []
total = len(y_steps) * len(x_steps)
done  = 0
for gy in y_steps:
    row_preds = []
    for gx in x_steps:
        pred = knn_predict_2d(glucose_scaled, bmi_scaled, labels, gx, gy, k)
        row_preds.append(pred)
        done += 1
        if done % 5000 == 0:
            print(f"  {done}/{total} ({100*done//total}%)")
    grid_z.append(row_preds)

# ── 6. PLOT ───────────────────────────────────────────────────────────────────
cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
cmap_bold  = ListedColormap(['#FF0000', '#0000FF'])

plt.figure(figsize=(8, 6))

# Decision regions via pcolormesh
plt.pcolormesh(x_steps, y_steps, grid_z, cmap=cmap_light)

# Actual data points
colors = [('#FF0000' if l == 1 else '#0000FF') for l in labels]
plt.scatter(glucose_scaled, bmi_scaled, c=colors, edgecolor='k', s=20)

plt.title(f"KNN Classification (K={k}) using Glucose and BMI")
plt.xlabel("Glucose (Standardized)")
plt.ylabel("BMI (Standardized)")
plt.tight_layout()
plt.savefig('decision_boundary.png', dpi=150)
plt.show()

print("Done! Saved as decision_boundary.png")

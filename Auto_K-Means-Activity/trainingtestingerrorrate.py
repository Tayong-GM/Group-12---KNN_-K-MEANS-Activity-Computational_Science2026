import csv
import math
import random
import matplotlib.pyplot as plt
from collections import Counter

# ── 0. LOAD CSV ──────────────────────────────────────────────────────────────
def load_csv(filepath):
    with open(filepath, newline='') as f:
        reader = csv.DictReader(f)
        data = [row for row in reader]
    return data

# ── 1. CLEAN DATA (replace 0s with median for specific cols) ─────────────────
def median(values):
    s = sorted(v for v in values if v is not None)
    n = len(s)
    return (s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2)

def clean_data(data):
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    # Convert to float first
    for row in data:
        for key in row:
            row[key] = float(row[key])

    for col in zero_cols:
        vals = [row[col] for row in data if row[col] != 0]
        med = median(vals)
        for row in data:
            if row[col] == 0:
                row[col] = med
    return data

# ── 2. STANDARDIZE (z-score) ─────────────────────────────────────────────────
def standardize(data, feature_cols):
    stats = {}
    for col in feature_cols:
        vals = [row[col] for row in data]
        mean = sum(vals) / len(vals)
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
        stats[col] = (mean, std if std != 0 else 1)

    for row in data:
        for col in feature_cols:
            mean, std = stats[col]
            row[col] = (row[col] - mean) / std
    return data, stats

# ── 3. TRAIN/TEST SPLIT ───────────────────────────────────────────────────────
def train_test_split(data, test_size=0.2, seed=42):
    random.seed(seed)
    shuffled = data[:]
    random.shuffle(shuffled)
    split = int(len(shuffled) * (1 - test_size))
    return shuffled[:split], shuffled[split:]

# ── 4. KNN ────────────────────────────────────────────────────────────────────
def euclidean(a, b, cols):
    return math.sqrt(sum((a[c] - b[c]) ** 2 for c in cols))

def knn_predict(train, test_point, k, feature_cols):
    distances = [(euclidean(test_point, row, feature_cols), row['Outcome']) for row in train]
    distances.sort(key=lambda x: x[0])
    k_nearest = [label for _, label in distances[:k]]
    return Counter(k_nearest).most_common(1)[0][0]

def accuracy(train, data, k, feature_cols):
    correct = sum(
        1 for row in data
        if knn_predict(train, row, k, feature_cols) == row['Outcome']
    )
    return correct / len(data)

# ── 5. MAIN ───────────────────────────────────────────────────────────────────
feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

raw = load_csv('diabetes-k-nn.csv')
cleaned = clean_data(raw)
standardized, _ = standardize(cleaned, feature_cols)

train_data, test_data = train_test_split(standardized, test_size=0.2, seed=42)

k_range = range(1, 26)
train_acc = []
test_acc  = []

print("Computing KNN for K=1 to 25... (this takes a minute)")
for k in k_range:
    ta = accuracy(train_data, train_data, k, feature_cols)
    te = accuracy(train_data, test_data,  k, feature_cols)
    train_acc.append(ta)
    test_acc.append(te)
    print(f"K={k:2d} | Train: {ta:.4f} | Test: {te:.4f}")

# ── 6. PLOT: Training vs Testing Accuracy ─────────────────────────────────────
plt.figure(figsize=(10, 6))
plt.plot(list(k_range), train_acc, label='Training Accuracy', marker='o')
plt.plot(list(k_range), test_acc,  label='Testing Accuracy',  marker='s')
plt.title('Training vs Testing Accuracy')
plt.xlabel('K Value')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('training_vs_testing.png', dpi=150)
plt.show()

# ── 7. PLOT: Error Rate vs K ──────────────────────────────────────────────────
error_rate = [1 - x for x in test_acc]
plt.figure(figsize=(10, 6))
plt.plot(list(k_range), error_rate, color='red', linestyle='--', marker='x')
plt.title('Error Rate vs K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.grid(True)
plt.tight_layout()
plt.savefig('error_rate.png', dpi=150)
plt.show()

print("\nDone! Graphs saved as training_vs_testing.png and error_rate.png")

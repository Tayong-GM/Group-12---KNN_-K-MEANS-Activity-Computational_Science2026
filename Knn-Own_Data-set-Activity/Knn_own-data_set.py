import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# DATA
# -----------------------------
df = pd.DataFrame({
    "BP": [130,110,165,118,135,155,112,168,129,108,175,132,111,160,136,115,170,128,105,138,162],
    "HR": [82,65,98,72,85,95,68,102,84,66,110,82,69,100,79,70,105,78,60,88,99],
    "Risk": ["Moderate","Low","High","Low","Moderate","High","Low","High","Moderate","Low",
             "High","Moderate","Low","High","Moderate","Low","High","Moderate","Low","Moderate","High"]
})

# Encode labels
label_map = {"Low":0, "Moderate":1, "High":2}
reverse = {v:k for k,v in label_map.items()}
df["Risk"] = df["Risk"].map(label_map)

# -----------------------------
# NORMALIZE
# -----------------------------
X_raw = df[["BP", "HR"]].values
y = df["Risk"].values

X_min, X_max = X_raw.min(axis=0), X_raw.max(axis=0)
X = (X_raw - X_min) / (X_max - X_min)

# -----------------------------
# SPLIT
# -----------------------------
np.random.seed(42)
idx = np.random.permutation(len(X))
train, test = idx[:16], idx[16:]

X_train, X_test = X[train], X[test]
y_train, y_test = y[train], y[test]

# -----------------------------
# KNN
# -----------------------------
def knn(X_train, y_train, x, k):
    dists = np.linalg.norm(X_train - x, axis=1)
    k_idx = np.argsort(dists)[:k]
    labels = y_train[k_idx]
    return np.bincount(labels).argmax(), dists

# -----------------------------
# TEST EXAMPLE
# -----------------------------
test_point = np.array([132, 82])
test_scaled = (test_point - X_min) / (X_max - X_min)

print("Test (scaled):", test_scaled)

for i in range(10):
    d = np.linalg.norm(X_train[i] - test_scaled)
    print(f"{i+1}. {d:.4f} -> {reverse[y_train[i]]}")

# -----------------------------
# EVALUATION
# -----------------------------
for k in [3,5,7]:
    preds = [knn(X_train, y_train, x, k)[0] for x in X_test]
    acc = np.mean(preds == y_test)

    print(f"\nK={k} Accuracy: {acc:.2f}")
    print("Confusion Matrix:")
    cm = np.zeros((3,3), int)
    for a, p in zip(y_test, preds):
        cm[a][p] += 1
    print(cm)

# -----------------------------
# VISUALIZATION
# -----------------------------
k = 3
new = np.array([134, 83])
new_scaled = (new - X_min) / (X_max - X_min)

pred, dists = knn(X, y, new_scaled, k)

plt.figure()

for c in np.unique(y):
    pts = X[y == c]
    plt.scatter(pts[:,0], pts[:,1], label=reverse[c])

plt.scatter(new_scaled[0], new_scaled[1], marker="*", s=200)

radius = np.sort(dists)[k-1]
plt.gca().add_patch(plt.Circle(new_scaled, radius, fill=False))

plt.title(f"KNN (k={k}) Prediction: {reverse[pred]}")
plt.legend()
plt.show()


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# # -----------------------------
# # DATASET
# # -----------------------------
# data = {
#     "BP": [130,110,165,118,135,155,112,168,129,108,175,132,111,160,136,115,170,128,105,138,162],
#     "HR": [82,65,98,72,85,95,68,102,84,66,110,82,69,100,79,70,105,78,60,88,99],
#     "Risk": [
#         "Moderate","Low","High","Low","Moderate","High","Low","High","Moderate","Low",
#         "High","Moderate","Low","High","Moderate","Low","High","Moderate","Low","Moderate","High"
#     ]
# }

# df = pd.DataFrame(data)

# # Encode labels
# label_map = {"Low":0, "Moderate":1, "High":2}
# reverse_map = {0:"Low", 1:"Moderate", 2:"High"}
# df["Risk"] = df["Risk"].map(label_map)

# # -----------------------------
# # MIN-MAX (GLOBAL — FIXED)
# # -----------------------------
# X_min = df[["BP", "HR"]].min()
# X_max = df[["BP", "HR"]].max()

# def min_max(col):
#     return (col - col.min()) / (col.max() - col.min())

# df["BP_s"] = min_max(df["BP"])
# df["HR_s"] = min_max(df["HR"])

# X = df[["BP_s", "HR_s"]].values
# y = df["Risk"].values

# # -----------------------------
# # TRAIN-TEST SPLIT
# # -----------------------------
# np.random.seed(42)
# indices = np.random.permutation(len(X))

# split = int(0.8 * len(X))
# train_idx = indices[:split]
# test_idx = indices[split:]

# X_train, X_test = X[train_idx], X[test_idx]
# y_train, y_test = y[train_idx], y[test_idx]

# # -----------------------------
# # DISTANCE
# # -----------------------------
# def distance(a, b):
#     return np.sqrt(np.sum((a - b) ** 2))

# # -----------------------------
# # KNN
# # -----------------------------
# def knn_predict(X_train, y_train, point, k):
#     distances = []
#     for i in range(len(X_train)):
#         d = distance(point, X_train[i])
#         distances.append((d, y_train[i]))

#     distances.sort(key=lambda x: x[0])
#     neighbors = distances[:k]

#     votes = {}
#     for _, label in neighbors:
#         votes[label] = votes.get(label, 0) + 1

#     return max(votes, key=votes.get), distances

# # -----------------------------
# # MANUAL DISTANCE SECTION (FIXED)
# # -----------------------------
# print("\n--- MANUAL DISTANCE COMPUTATION (10 TRAINING SAMPLES) ---")

# test_raw = np.array([132, 82])

# # ✅ SAME scaling method as training
# test_scaled = np.array([
#     (test_raw[0] - X_min["BP"]) / (X_max["BP"] - X_min["BP"]),
#     (test_raw[1] - X_min["HR"]) / (X_max["HR"] - X_min["HR"])
# ])

# print(f"\nTest Patient (raw): {test_raw}")
# print(f"Test Patient (scaled): {test_scaled}\n")

# for i in range(10):
#     train_point = X_train[i]
#     train_label = reverse_map[y_train[i]]

#     diff = test_scaled - train_point
#     squared = diff ** 2
#     dist = np.sqrt(np.sum(squared))

#     print(f"Sample {i+1}:")
#     print(f"  Train Point: {train_point}")
#     print(f"  Label: {train_label}")
#     print(f"  Squared diff: {squared}")
#     print(f"  Distance: {dist:.4f}\n")

# # -----------------------------
# # EVALUATION
# # -----------------------------
# def evaluate(k):
#     preds = []

#     for pt in X_test:
#         pred, _ = knn_predict(X_train, y_train, pt, k)
#         preds.append(pred)

#     preds = np.array(preds)
#     acc = np.mean(preds == y_test)

#     print(f"\nK = {k}")
#     print("Accuracy:", acc)

#     cm = np.zeros((3,3), dtype=int)
#     for a, p in zip(y_test, preds):
#         cm[a][p] += 1

#     print("Confusion Matrix:")
#     print(cm)

# for k in [3,5,7]:
#     evaluate(k)

# # -----------------------------
# # VISUALIZATION
# # -----------------------------
# k = 3

# new_raw = np.array([134, 83])

# new_point = np.array([
#     (new_raw[0] - X_min["BP"]) / (X_max["BP"] - X_min["BP"]),
#     (new_raw[1] - X_min["HR"]) / (X_max["HR"] - X_min["HR"])
# ])

# prediction, distances = knn_predict(X, y, new_point, k)

# dist_values = np.array([d[0] for d in distances])
# nearest_idx = np.argsort(dist_values)[:k]

# plt.figure()

# for label in np.unique(y):
#     pts = X[y == label]
#     plt.scatter(pts[:,0], pts[:,1], label=f"Class {reverse_map[label]}")

# plt.scatter(new_point[0], new_point[1], marker="*", s=200)

# radius = dist_values[nearest_idx[-1]]
# circle = plt.Circle((new_point[0], new_point[1]), radius, fill=False)
# plt.gca().add_patch(circle)

# plt.title(f"KNN Visualization (K={k}) - Predicted: {reverse_map[prediction]}")
# plt.xlabel("Blood Pressure (scaled)")
# plt.ylabel("Heart Rate (scaled)")
# plt.legend()

# plt.show()
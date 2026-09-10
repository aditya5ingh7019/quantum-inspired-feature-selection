#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import dimod
from sklearn.datasets import load_breast_cancer
from sklearn.feature_selection import mutual_info_classif

# 1. Load data
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names
n_features = X.shape[1]

# 2. Relevance: mutual information between each feature and the target
relevance = mutual_info_classif(X, y, random_state=0)

# 3. Redundancy: absolute correlation between each pair of features
corr_matrix = np.abs(np.corrcoef(X, rowvar=False))

# 4. Build the QUBO
# Minimize: -sum(relevance_i * x_i) + alpha * sum_{i<j}(redundancy_ij * x_i * x_j)
alpha = 1.0  # tradeoff weight; tune this later

Q = {}
for i in range(n_features):
    Q[(i, i)] = -relevance[i]

for i in range(n_features):
    for j in range(i + 1, n_features):
        Q[(i, j)] = alpha * corr_matrix[i, j]

bqm = dimod.BinaryQuadraticModel.from_qubo(Q)

# 5. Solve locally (no API token needed)
sampler = dimod.SimulatedAnnealingSampler()
sampleset = sampler.sample(bqm, num_reads=100)
best = sampleset.first.sample

selected_features = [feature_names[i] for i, v in best.items() if v == 1]
print(f"Selected {len(selected_features)} features:")
print(selected_features)


# In[6]:


from amplify import VariableGenerator, FixstarsClient, solve

# Step 2: rebuild the objective in Amplify's format
gen = VariableGenerator()
x = gen.array("Binary", n_features)

objective = sum(Q[(i, i)] * x[i] for i in range(n_features) if (i, i) in Q)
objective += sum(Q[(i, j)] * x[i] * x[j] for (i, j) in Q if i != j)

# Step 3: set your token and solve
client = FixstarsClient()
client.token = "AE/FzstSenlWNLGUG7a8MImkZRVCG7pHipV"
client.parameters.timeout = 1000  # milliseconds

result = solve(objective, client)
best = result.best
selected = [feature_names[i] for i in range(n_features) if best.values[x[i]] == 1]
print(selected)


# In[7]:


from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --- Classical baseline: SelectKBest with same k as your QUBO result ---
k = 2  # match however many features your QUBO selected

selector = SelectKBest(score_func=mutual_info_classif, k=k)
selector.fit(X, y)
classical_selected = [feature_names[i] for i in selector.get_support(indices=True)]
print("Classical (SelectKBest) selected:", classical_selected)

# --- Your QUBO-selected features (from Amplify/simulated annealing) ---
quantum_selected = ['fractal dimension error', 'worst perimeter']
print("Quantum-inspired (QUBO) selected:", quantum_selected)

# --- Downstream comparison: train a classifier on each feature set ---
def evaluate_feature_set(feature_list):
    idx = [list(feature_names).index(f) for f in feature_list]
    X_sub = X[:, idx]
    X_train, X_test, y_train, y_test = train_test_split(
        X_sub, y, test_size=0.3, random_state=42
    )
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    return accuracy_score(y_test, preds)

acc_classical = evaluate_feature_set(classical_selected)
acc_quantum = evaluate_feature_set(quantum_selected)

print(f"\nAccuracy with SelectKBest features: {acc_classical:.4f}")
print(f"Accuracy with QUBO-selected features: {acc_quantum:.4f}")


# In[8]:


# --- Try different alpha values to get different selection sizes ---
for alpha_test in [1.0, 0.3, 0.1, 0.05]:
    Q_test = {}
    for i in range(n_features):
        Q_test[(i, i)] = -relevance[i]
    for i in range(n_features):
        for j in range(i + 1, n_features):
            Q_test[(i, j)] = alpha_test * corr_matrix[i, j]

    bqm_test = dimod.BinaryQuadraticModel.from_qubo(Q_test)
    result_test = sampler.sample(bqm_test, num_reads=100)
    best_test = result_test.first.sample
    selected_test = [feature_names[i] for i, v in best_test.items() if v == 1]
    print(f"alpha={alpha_test}: {len(selected_test)} features -> {selected_test}")


# In[9]:


# --- k=3 comparison ---
k = 3
quantum_selected_k3 = ['mean radius', 'mean symmetry', 'worst area']

selector_k3 = SelectKBest(score_func=mutual_info_classif, k=k)
selector_k3.fit(X, y)
classical_selected_k3 = [feature_names[i] for i in selector_k3.get_support(indices=True)]
print("Classical (SelectKBest, k=3):", classical_selected_k3)
print("Quantum-inspired (QUBO, alpha=0.3):", quantum_selected_k3)

acc_classical_k3 = evaluate_feature_set(classical_selected_k3)
acc_quantum_k3 = evaluate_feature_set(quantum_selected_k3)

print(f"\nAccuracy with SelectKBest (k=3): {acc_classical_k3:.4f}")
print(f"Accuracy with QUBO-selected (k=3): {acc_quantum_k3:.4f}")


# In[10]:


# --- Fine-grained scan to find alpha giving exactly 4 features ---
for alpha_test in [0.28, 0.25, 0.22, 0.2, 0.18, 0.15, 0.12]:
    Q_test = {}
    for i in range(n_features):
        Q_test[(i, i)] = -relevance[i]
    for i in range(n_features):
        for j in range(i + 1, n_features):
            Q_test[(i, j)] = alpha_test * corr_matrix[i, j]

    bqm_test = dimod.BinaryQuadraticModel.from_qubo(Q_test)
    result_test = sampler.sample(bqm_test, num_reads=100)
    best_test = result_test.first.sample
    selected_test = [feature_names[i] for i, v in best_test.items() if v == 1]
    print(f"alpha={alpha_test}: {len(selected_test)} features -> {selected_test}")


# In[11]:


# --- k=4 comparison ---
k = 4
quantum_selected_k4 = ['mean perimeter', 'mean symmetry', 'concavity error', 'worst area']

selector_k4 = SelectKBest(score_func=mutual_info_classif, k=k)
selector_k4.fit(X, y)
classical_selected_k4 = [feature_names[i] for i in selector_k4.get_support(indices=True)]
print("Classical (SelectKBest, k=4):", classical_selected_k4)
print("Quantum-inspired (QUBO, alpha=0.28):", quantum_selected_k4)

acc_classical_k4 = evaluate_feature_set(classical_selected_k4)
acc_quantum_k4 = evaluate_feature_set(quantum_selected_k4)

print(f"\nAccuracy with SelectKBest (k=4): {acc_classical_k4:.4f}")
print(f"Accuracy with QUBO-selected (k=4): {acc_quantum_k4:.4f}")


# In[ ]:





# FedCADS-UCB: Client Selection in Federated Learning

**Federated CUSUM-Adaptive Discounted Shapley UCB**

A research project from AI211 — Machine Learning Theory, Indian Institute of Technology Ropar

Authors: Kush Mistry (2024AIB1368), Rishi Datt Gupta (2024AIB1377), I. Nikhil Varma (2024AIB1350)
Mentor: Shradha Sharma

---

## Overview

This repository contains the full implementation of a three-part study on intelligent client selection strategies for Federated Learning under non-IID data and concept drift. The project culminates in the design and evaluation of **FedCADS-UCB**, a novel algorithm that combines Shapley value-based bandit rewards, CUSUM-driven adaptive memory, exponential-backoff client quarantine, and fairness constraints into a single coherent framework.

The central problem this project addresses: in a federated system with 50 clients, each holding only 2 out of 10 digit classes, selecting the wrong 10 clients per round causes the global model to forget entire classes. The situation becomes dramatically worse when some clients are adversarially poisoned mid-training. Standard algorithms fail in predictable but instructive ways. FedCADS-UCB is designed to remain robust throughout.

---

## Repository Structure

```
FedCADS-UCB-Client-Selection/
│
├── src/
│   ├── data_setup.py          # MNIST loading and non-IID partitioning across 50 clients
│   ├── models.py              # Four model architectures: LR, SVM, DNN, Simple CNN
│   ├── main.py                # Client update logic and DivFL selection implementation
│   ├── shapley.py             # Monte Carlo Shapley value computation and S-FedAvg selection
│   ├── fedcads.py             # FedCADS-UCB core: CUSUM, UCB scoring, quarantine, fairness
│   └── train_task3.py         # Full Task 3 training loop comparing all three algorithms
│
├── results/
│   ├── task1_divfl_vs_sfedavg.png       # Task 1: stable environment comparison
│   ├── task2_concept_drift.png          # Task 2: both algorithms under poisoning
│   └── task3_fedcads_comparison.png     # Task 3: FedCADS-UCB vs baselines under drift
│
├── docs/
│   └── AI211_ProjectReport_Final_IEEE.pdf   # Full IEEE-format project report
│
├── LICENSE
└── README.md
```

---

## Background: Why Client Selection Matters

Federated Learning (FL) trains a shared global model across many distributed clients without ever sharing raw data. Each round, the server selects a subset of clients, sends them the current global model, receives their local weight updates (deltas), and aggregates those updates via FedAvg.

In our experimental setup there are 50 clients and only 10 can be selected per round. Each client holds data for at most 2 out of 10 digit classes — an extreme non-IID configuration. If we randomly select 10 clients in a given round, we might only cover 4 or 5 digit classes. The global model then degrades on the unrepresented classes. Intelligent client selection directly determines whether the model learns all classes efficiently.

Two additional challenges compound this:

**Non-IID data heterogeneity.** When client data distributions differ heavily, any selection algorithm must ensure broad coverage of the label space across the selected cohort. This is harder than it sounds — the server has no direct visibility into client data distributions.

**Concept drift.** Real-world FL deployments must contend with clients whose data distributions shift over time due to sensor failures, user behaviour changes, or adversarial tampering. An algorithm that works beautifully in a stable environment may collapse catastrophically when 30% of clients suddenly start sending harmful updates.

---

## Experimental Setup

| Parameter | Value |
|---|---|
| Dataset | MNIST (60,000 training, 10,000 test images) |
| Number of clients | 50 |
| Clients selected per round | 10 |
| Total communication rounds | 100 |
| Data partitioning | Extreme non-IID: 2 shards per client, 600 images per shard |
| Local training | 3 epochs, SGD, lr = 0.01, batch size = 16 |
| Shapley estimation | R = 5 Monte Carlo permutations |
| Concept drift injection | Round 50, clients 0-14 (label shift: y_new = (y_old + 1) mod 10) |
| Model architectures | Logistic Regression, SVM, DNN, Simple CNN |
| Hardware | NVIDIA GPU, PyTorch |

The non-IID partitioning works as follows: the 60,000 training images are sorted by label and cut into 100 shards of 600 images each. Each client receives exactly 2 shards, giving it 1,200 images that cover at most 2 digit classes. This forces the selection algorithm to actively manage class coverage.

---

## Task 1: Baseline Comparison — DivFL vs S-FedAvg

### DivFL (Diverse Client Selection)

DivFL selects clients whose gradient updates are maximally diverse relative to each other. The formal objective is a Facility Location problem:

```
G(S) = sum over all clients k of [ min over i in S of ||gradient_k - gradient_i|| ]
```

This quantifies how well the selected set S represents the full population of 50 clients. We want to maximize G(S), because larger values mean the selected cohort better covers the diversity of the full client pool. Exact maximization is NP-hard (there are C(50, 10) possible subsets), so DivFL uses a greedy approach that is guaranteed to achieve a (1 - 1/e) ≈ 63% optimal solution due to the submodular structure of the objective.

**In practice:** DivFL requires a probe round where each client does 1 step of training so the server can compare gradient directions. The client whose addition to the current selected set produces the largest increase in G(S) is added greedily until 10 clients are selected.

### S-FedAvg (Shapley Value-Based Federated Averaging)

S-FedAvg measures each client's true marginal contribution to validation accuracy using Shapley values from cooperative game theory. For each selected client i, the Shapley value is estimated via Monte Carlo:

```
SV_i = (1/R) * sum over R random permutations of [ v(coalition up to i) - v(coalition without i) ]
```

where v(coalition) is the validation accuracy when the average of that coalition's updates is applied to the global model. A positive Shapley value means the client helped; a negative value means it hurt.

Each client maintains a relevance score updated with an exponential moving average:

```
phi_k = 0.75 * phi_k + 0.25 * SV_k(t)
```

Clients are then selected probabilistically via softmax over relevance scores.

### Task 1 Results

In a stable environment (no drift), DivFL converges significantly faster in the first 15-20 rounds because its diversity objective immediately ensures all digit classes are represented from round 1. For the Simple CNN, DivFL reached 89.19% accuracy by round 12, while S-FedAvg reached only 84.08% at the same point.

S-FedAvg suffers an initialization delay: all relevance scores start equal, so early rounds are essentially random selection until enough Shapley estimates accumulate to differentiate clients. This takes 15-20 rounds.

By round 100, both converge to similar final accuracies: approximately 96.8% for the CNN, 87% for LR, 85% for SVM, and 90% for the DNN. In a stationary environment, both methods have enough rounds to identify the best clients.

---

## Task 2: Robustness Under Concept Drift

At round 50, clients 0 through 14 (30% of the client pool) have their labels secretly shifted:

```
y_new = (y_original + 1) mod 10
```

So digit 0 becomes 1, digit 1 becomes 2, and so on. These 15 clients now consistently send harmful updates that push the model in the wrong direction. Neither algorithm was warned about this event.

### DivFL's Failure

DivFL's core weakness becomes its downfall here. When clients are poisoned, their gradient updates diverge sharply from the global model parameters — they now point in completely wrong directions. From DivFL's perspective, this makes them the most diverse clients in the entire pool. DivFL actively selects them, perceiving their anomalous updates as valuable diversity.

For the Simple CNN, accuracy collapsed from 95.87% at round 50 to 57.46% by round 60 and never recovered, finishing at 58.43% at round 100. This is not an implementation bug — it is a fundamental algorithmic vulnerability. Diversity maximization without quality filtering is exploitable.

### S-FedAvg's Partial Recovery

S-FedAvg handles drift far better. When a poisoned client is selected, its update applied to the global model causes validation accuracy to drop. Its Shapley value turns negative. The EMA score decreases, and softmax selection gradually stops choosing that client.

However, recovery is slow. The EMA factor alpha = 0.75 means old positive scores decay at (0.75)^t per round. After 10 rounds, 6% of the old positive reputation remains. It takes roughly 10-15 rounds for a bad client to be fully suppressed. During those rounds, the poisoned client continues being selected and causing damage.

Final accuracy for the CNN: 94.90% — a good eventual recovery, but 40+ rounds of degraded performance.

**Key lesson:** S-FedAvg is structurally superior for adversarial settings because it filters by contribution quality, not diversity. But its fixed memory (fixed alpha) cannot accelerate forgetting when things go bad quickly. This gap motivates FedCADS-UCB.

---

## Task 3: FedCADS-UCB — The Proposed Algorithm

FedCADS-UCB stands for: **Federated CUSUM-Adaptive Discounted Shapley UCB**.

The algorithm integrates five components into a unified framework. Each component addresses a specific failure mode observed in Tasks 1 and 2.

### Component 1: Multi-Armed Bandit Framework

Each of the 50 clients is treated as an arm of a multi-armed bandit. The reward for pulling arm k is the Shapley value of client k's update in that round. The server must balance exploration (trying clients it hasn't selected recently, in case they've become valuable) against exploitation (repeatedly selecting the best known clients).

This framing is appropriate because the server receives no signal about clients it does not select in a given round — a classic partial observability problem that bandit algorithms are designed for.

### Component 2: Discounted UCB Scoring

For each active client k, the server maintains:
- `R_tilde_k`: discounted sum of all past Shapley rewards
- `N_tilde_k`: discounted count of selections

The UCB score is:

```
UCB_k = (R_tilde_k / N_tilde_k) + sqrt(ln(t + 1) / N_tilde_k)
```

The first term is the estimated mean reward. The second term is an exploration bonus that grows when a client has been selected rarely (small N_tilde_k) and shrinks as confidence increases. This ensures every client gets tried eventually, and clients that have been rarely explored are given the benefit of the doubt.

### Component 3: CUSUM-Driven Adaptive Discount (Core Novelty)

Every round, the discount factor gamma is recomputed based on the current level of detected drift:

```
g_t = acc_(t-2) - acc_(t-1)           # accuracy drop this round
C_t = max(0, C_(t-1) + g_t - 0.01)   # CUSUM accumulator
DriftSignal = min(C_t / 0.05, 1.0)   # normalized signal in [0, 1]
gamma(t) = 0.95 - 0.65 * DriftSignal # adaptive discount factor
```

When the environment is stable, C_t stays near 0, DriftSignal is 0, and gamma = 0.95. Historical Shapley estimates are trusted and preserved. When concept drift begins, accuracy drops, C_t rises, DriftSignal approaches 1, and gamma drops toward 0.30. Old reputation is rapidly discounted. When drift stabilizes (poisoned clients quarantined, accuracy recovering), C_t stops rising and gamma gradually returns to 0.95.

The discount is applied every round to all clients, whether selected or not:

```
R_tilde_k = gamma * R_tilde_k + SV_k  (if k was selected, else just the discount)
N_tilde_k = gamma * N_tilde_k + 1     (if k was selected, else just the discount)
```

This is the core novelty of FedCADS-UCB. S-FedAvg uses a fixed alpha = 0.75 always. CUCB with sliding windows uses a fixed window size. FedCADS-UCB reads the actual magnitude of ongoing drift in real-time and continuously tunes the forgetting speed — more drift means more forgetting, and the relationship is direct and principled.

### Component 4: Fairness via Virtual Queue

If we always select the 10 highest-scoring clients, some clients will never be chosen. In our non-IID setup, this means entire digit classes can be excluded indefinitely. To prevent this, each client k accumulates a fairness debt:

```
D_k(t) = max(0, D_k(t-1) + 0.01 - b_k(t-1))
```

where b_k(t) is 1 if client k was selected in round t, else 0. The target selection rate c_k = 0.01 means each client should be selected in at least 1% of rounds (once every 100 rounds). Unselected clients accumulate debt; selected clients have their debt reduced.

The final scoring combines UCB and fairness:

```
Score_k = 0.9 * UCB_k + 0.1 * D_k
```

This 90/10 split means performance dominates, but a client with growing fairness debt will eventually earn a selection even if its estimated UCB score is moderate.

### Component 5: Quarantine with Exponential Backoff

When a client's Shapley value is negative for two consecutive rounds it is quarantined — removed from the active selection pool for a time. The two-round threshold (rather than one) is deliberate: Monte Carlo Shapley with R=5 permutations has noise, and a single slightly-negative value can occur by chance even for a good client. Two consecutive negative values is a much stronger signal.

Quarantine duration uses exponential backoff:

```
Q_k = 10 * 2^min(n_k - 1, 3)
```

where n_k is the number of times client k has been quarantined. This gives timeouts of 10, 20, 40, and 80 rounds for the first through fourth offenses. A client quarantined for the fourth time is effectively gone for the rest of the 100-round training run.

Additionally, when a quarantined client returns and posts a positive Shapley value, its suspicion counter is not fully reset — it is only decremented by 1. This gradual decay prevents adversarial clients from gaming the system by alternating between slightly negative and slightly positive values.

### The Two-Model Architecture

Updates from quarantined clients are not discarded. They are aggregated into a separate model `h_drift`, which tracks the new data concept that the poisoned clients represent. The main global model `h_global` is trained exclusively on clean clients. This architecture is borrowed from FedDrift and allows the system to preserve potentially useful information about shifting client distributions without contaminating the primary model.

### Full Algorithm per Round

```
1. Compute accuracy drop g_t = acc_(t-2) - acc_(t-1)
2. Update CUSUM: C_t = max(0, C_(t-1) + g_t - 0.01)
3. Compute gamma(t) = 0.95 - 0.65 * min(C_t / 0.05, 1)
4. Score all active (non-quarantined) clients using UCB + fairness debt
5. Select top-10 clients by score
6. Each selected client trains for 3 epochs, returns weight delta
7. Compute Shapley values for all selected clients (R=5 permutations)
8. Update R_tilde and N_tilde for all clients (with current gamma)
9. Update fairness debt for all clients
10. Quarantine check: increment suspicion counter for negative SV; 
    quarantine if counter >= 2 (exponential backoff duration)
11. Aggregate: h_global gets updates from clients with SV >= -0.01
                h_drift  gets updates from clients with SV <  -0.01
12. Decrement quarantine timers for all quarantined clients
```

---

## Task 3 Results

| Algorithm | Pre-Drift (Round 50) | Round 60 | Final (Round 100) |
|---|---|---|---|
| DivFL | 95.87% | 57.46% | 58.43% |
| S-FedAvg | 95.20% | 78.31% | approximately 80% |
| FedCADS-UCB | 94.60% | approximately 90% | **96.95%** |

FedCADS-UCB outperforms DivFL by +38.52 percentage points and S-FedAvg by approximately +17 percentage points at round 100 under concept drift. Total training time for all three algorithms (100 rounds each): 83.2 minutes on GPU.

The behaviour across training divides into three phases:

**Phase 1 — Stable (Rounds 1 to 50).** All three algorithms perform comparably, reaching 94-96%. FedCADS-UCB's CUSUM stays near 0, gamma = 0.95. UCB exploration gradually identifies the best 35 clean clients. The fairness debt ensures all digit classes are represented.

**Phase 2 — Drift Response (Rounds 51 to 60).** This is the decisive window. DivFL crashes to 57%. S-FedAvg drops to 78%. FedCADS-UCB dips briefly but stays above 86%. The CUSUM fires within 2 rounds of the poisoning taking effect. Poisoned clients accumulate two consecutive negative Shapley values and are quarantined. Simultaneously, gamma drops toward 0.30 and their old positive history is rapidly erased.

**Phase 3 — Recovery (Rounds 61 to 100).** With poisoned clients quarantined and gamma stabilizing back toward 0.95, the global model trains on clean updates only. Accuracy rebounds to the 93-97% range. Some oscillation persists because clients occasionally return from shorter quarantine timeouts before being re-quarantined, and because Monte Carlo Shapley has inherent variance at R=5. The overall trend is stable and upward.

Sample quarantine log output from actual training:

```
[!] FedCADS ALERT: Client 3 quarantined for 20 rounds (offense #2).
[!] FedCADS ALERT: Client 1 quarantined for 40 rounds (offense #3).
[!] FedCADS ALERT: Client 13 quarantined for 80 rounds (offense #4).
```

---

## Theoretical Guarantees

Under standard FL assumptions (L-smoothness, bounded gradient norms, bounded data heterogeneity), FedCADS-UCB satisfies:

```
E[ ||w_t - w*||^2 ] <= O(1/T)  +  O(epsilon_sel)  +  O(C_t / h_alarm)
                        [opt]       [selection bias]    [drift cost]
```

The selection bias term vanishes as gamma approaches 1 (stable phase), recovering the standard Shapley-UCB convergence rate with O(sqrt(T)) regret.

In full drift (gamma = 0.30), the regret bound approaches O(T^(2/3)), matching the theoretical guarantee of CUCB-SW for non-stationary bandits.

FedCADS-UCB achieves both bounds simultaneously by continuously interpolating gamma between regimes. This two-regime optimality is not shared by any single existing algorithm: S-FedAvg achieves O(sqrt(T)) in stable settings but is suboptimal under drift due to slow forgetting; CUCB-SW achieves O(T^(2/3)) under drift but introduces unnecessary forgetting in stable phases.

---

## Installation and Usage

**Requirements**

```
Python >= 3.8
PyTorch >= 1.12
torchvision
numpy
matplotlib
```

**Install dependencies**

```bash
pip install torch torchvision numpy matplotlib
```

**Run Task 3 (full three-algorithm comparison with concept drift)**

```bash
cd src
python train_task3.py
```

This script will:
1. Download MNIST automatically (approximately 11 MB) on first run
2. Partition data into 50 non-IID clients
3. Run DivFL for 100 rounds
4. Run S-FedAvg for 100 rounds
5. Run FedCADS-UCB for 100 rounds
6. Inject label-flip poisoning at round 50 for all three algorithms
7. Save the comparison plot as `task3_fedcads_comparison.png`
8. Print final accuracy for all three algorithms

Expected runtime: approximately 80-90 minutes on an NVIDIA GPU. CPU-only training will take significantly longer.

**Run individual components**

```python
from data_setup import create_non_iid_clients, train_dataset
from models import SimpleCNN
from fedcads import FedCADS

# Create 50 non-IID clients
client_data = create_non_iid_clients(train_dataset, num_clients=50, batch_size=16)

# Initialize FedCADS server
server = FedCADS(
    num_clients=50,
    fairness_rates=[0.01] * 50,
    beta=0.10
)
```

---

## Key Hyperparameters

| Parameter | Symbol | Value | Description |
|---|---|---|---|
| Max discount factor | gamma_max | 0.95 | Memory depth in stable environment |
| Min discount factor | gamma_min | 0.30 | Memory depth under full drift |
| CUSUM slack | kappa | 0.01 | Absorbs small random accuracy fluctuations |
| CUSUM alarm threshold | h_alarm | 0.05 | Triggers drift response when exceeded |
| Quarantine threshold | delta_q | 0.01 | Shapley value below which a client is suspect |
| Base quarantine duration | T_q | 10 | Rounds for first quarantine offense |
| Fairness weight | beta | 0.10 | Weight of fairness debt in final score |
| Fairness target rate | c_k | 0.01 | Target fraction of rounds each client is selected |

---

## Comparison with Related Work

| Algorithm | Handles Non-IID | Detects Drift | Adaptive Memory | Client Isolation |
|---|---|---|---|---|
| FedAvg (random) | Poorly | No | No | No |
| DivFL | Yes (fast convergence) | No (exploitable) | No | No |
| S-FedAvg | Yes (good quality filter) | Partially | No (fixed EMA) | Partial (soft deprioritization) |
| CUCB-SW | Partial | Yes (fixed window) | No (fixed window) | No |
| CUCB-BoB | Partial | Yes (ensemble) | Partially | No |
| FedCADS-UCB (ours) | Yes | Yes (CUSUM) | Yes (continuous) | Yes (quarantine) |

---

## Acknowledgments

This project was completed as part of AI211 — Machine Learning Theory at the Indian Institute of Technology Ropar, under the guidance of Mentor Shradha Sharma.

The algorithm builds on ideas from the following prior works:

- Balakrishnan et al., "Diverse Client Selection for Federated Learning via Submodular Maximization," ICLR 2022 (DivFL)
- Nagalapatti and Narayanam, "Game of Gradients: Mitigating Irrelevant Clients in Federated Learning," AAAI 2021 (S-FedAvg)
- Fouad et al., "Combinatorial Semi-Bandit in the Non-Stationary Environment," UAI 2020 (CUSUM and CUCB framework)
- Jothimurugesan et al., "Federated Learning under Distributed Concept Drift," AISTATS 2023 (dual-model architecture)
- McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data," AISTATS 2017 (FedAvg)

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

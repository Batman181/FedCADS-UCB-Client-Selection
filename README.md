<div align="center">

#  Client Selection in Federated Learning: Diversity, Shapley Values, and Drift-Resilient Bandits

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status: Complete](https://img.shields.io/badge/Status-Complete-success.svg)

**Official PyTorch Implementation of FedCADS-UCB** *A novel, drift-resilient client selection algorithm combining CUSUM drift detection, Shapley-valued bandits, and adaptive memory.*

</div>

---

##  Abstract & Motivation

**Federated Learning (FL)** enables distributed model training across edge devices while preserving data privacy. However, real-world FL deployments face two structural challenges that cause standard algorithms (like FedAvg) to fail:
1. **Extreme Non-IID Data Distribution:** Local datasets are highly imbalanced. If client selection is naive, the global model suffers from catastrophic forgetting of minority classes.
2. **Sudden Concept Drift (Data Poisoning):** Client data distributions can shift abruptly due to sensor malfunctions or adversarial label-flipping, injecting toxic gradients into the global model.

This project provides a comprehensive empirical study of existing baselines (**DivFL** and **S-FedAvg**) under extreme conditions and introduces **FedCADS-UCB**, an adaptive bandit-based algorithm that achieves two-regime optimality—maintaining stability in stationary environments and rapidly recovering during concept drift.

---

##  Algorithmic Frameworks

### 1. DivFL (Diverse Client Selection)
* **Mechanism:** Formulates client selection as a Facility Location problem, greedily selecting a subset of clients that maximizes the diversity of gradient updates.
* **Vulnerability:** Under concept drift, poisoned gradients diverge significantly from the global model. DivFL's objective misinterprets this adversarial divergence as "valuable diversity," actively selecting malicious clients and causing total model collapse.

### 2. S-FedAvg (Shapley-Value Federated Averaging)
* **Mechanism:** Utilizes Monte Carlo approximation of Shapley Values to measure each client's exact marginal contribution to a clean validation set. Client reputations are tracked using an Exponential Moving Average (EMA).
* **Vulnerability:** The EMA relies on a fixed decay factor ($\alpha=0.75$). When a historically reliable client suddenly drifts, the fixed memory causes a dangerous lag in updating their reputation, leading to a slow and painful recovery phase.

### 3. FedCADS-UCB (Proposed Solution) 
FedCADS-UCB addresses these flaws by integrating real-time statistical process control with multi-armed bandits:
* **CUSUM Drift Detection:** Continuously monitors the global validation accuracy. If a sustained drop is detected ($C_t > h_{alarm}$), it triggers a drift state.
* **Adaptive Discounting ($\gamma$):** Instead of a fixed memory, the discount factor shrinks dynamically when drift is detected. This forces the algorithm to rapidly "forget" stale reputations and identify newly poisoned clients.
* **Exponential Quarantine:** Clients yielding consecutive negative Shapley values are quarantined with an exponential backoff timer ($10 \times 2^{n-1}$ rounds). This isolates repeat offenders without permanently banning clients whose data might represent valid new concepts.
* **Fairness Debt ($D_k$):** A virtual queue system guarantees that rarely-selected clients are eventually sampled, ensuring coverage of minority classes in highly non-IID settings.

---

##  Experimental Setup & Key Results

* **Dataset:** MNIST, artificially partitioned to extreme non-IID settings (each client holds only 2 out of 10 digit classes).
* **Models Tested:** Logistic Regression, SVM, Deep Neural Network (DNN), and Simple CNN.
* **Adversarial Setup:** Sudden label-flipping poisoning injected into 30% of the client pool exactly at **Round 50**.

### Robustness Comparison (Simple CNN)

| Algorithm | Pre-Drift Accuracy (Round 50) | Post-Drift Drop (Round 60) | Final Recovery (Round 100) |
| :--- | :---: | :---: | :---: |
| **DivFL** | 95.87% | 57.46% (Collapsed) | 58.43%  |
| **S-FedAvg** | 95.20% | ~78.00% (Struggling) | ~80.00% |
| **FedCADS-UCB** | **94.60%** | **~90.00% (Resilient)** | **96.95%**  |

**Conclusion:** FedCADS-UCB effectively quarantines malicious actors and dynamically adjusts its memory, rebounding to near-baseline accuracy and outperforming DivFL by over **+38%** and S-FedAvg by **+16%**.

---

##  Installation & Quick Start

### Prerequisites
Ensure you have Python 3.8+ installed. It is recommended to use a virtual environment.

```bash
# Clone the repository
git clone [https://github.com/Batman181/FedCADS-UCB-Client-Selection.git](https://github.com/Batman181/FedCADS-UCB-Client-Selection.git)
cd FedCADS-UCB-Client-Selection

# Install required dependencies
pip install torch torchvision numpy matplotlib


├── docs/
│   └── AI211_ProjectReport_Final_IEEE.pdf  # Comprehensive mathematical report & proofs
├── results/
│   ├── Task1_Result.jpeg                   # Baseline stable performance charts
│   ├── Task2_Result.jpg                    # Concept drift stress-test charts
│   └── Figure_1.png                        # Final comparative trajectory
├── src/
│   ├── data_setup.py                       # Non-IID data partitioning logic
│   ├── fedcads.py                          # FedCADS-UCB algorithm implementation
│   ├── main.py                             # Local training loops & DivFL selection
│   ├── models.py                           # PyTorch model architectures
│   ├── shapley.py                          # Monte Carlo Shapley value estimation
│   └── train_task3.py                      # Main execution script
└── README.md

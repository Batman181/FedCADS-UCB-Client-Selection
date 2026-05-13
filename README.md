# Client Selection in Federated Learning under Concept Drift

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg) ![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)

A research-oriented framework for robust client selection in Federated Learning (FL) under extreme non-IID data distributions and sudden concept drift. This project provides a fully transparent implementation of **FedCADS-UCB** (Federated CUSUM-Adaptive Discounted Shapley UCB) alongside benchmark implementations of diverse and contribution-based selection methods, enabling a deep conceptual understanding of drift-resilient distributed training.

---

## Overview

Federated Learning enables collaborative model training while keeping data on local devices. However, real-world deployments face two massive challenges: highly imbalanced (non-IID) client data, and non-stationary data distributions that evolve abruptly due to adversarial poisoning or sensor failures (concept drift). This repository demonstrates how intelligent client selection can overcome these hurdles. The project emphasizes a comparative analysis, benchmarking our proposed FedCADS-UCB algorithm against diversity-maximizing approaches (DivFL) and Shapley-based contribution filters (S-FedAvg) to provide a holistic view of algorithm robustness.

---

## Key Features

* **Advanced Selection Algorithms:** Includes implementations of **DivFL** (Diverse Client Selection via submodular maximization) and **S-FedAvg** (Shapley Value-based Federated Averaging) for baseline clarity, alongside our proposed **FedCADS-UCB**.
* **Real-Time Drift Adaptation:** Integrates a CUSUM drift detector to continuously modulate the memory (discount factor) of a Shapley-valued bandit, allowing the global model to rapidly "forget" stale reputations when clients turn malicious.
* **Rigorous Quantitative Evaluation:** Employs a robust set of architectures (Logistic Regression, SVM, Deep Neural Networks, Simple CNN) and injects targeted label-flip poisoning to objectively assess algorithmic recovery speed and final test accuracy.
* **Modular and Reproducible Design:** Built with a clear separation of concerns: dedicated modules for data partitioning, Shapley estimation, core algorithm logic, and a central training execution script.

---

## Project Structure

```text
FedCADS-UCB-Client-Selection/
├── README.md
├── .gitignore
├── docs/
│   └── AI211_ProjectReport_Final_IEEE.pdf  # Comprehensive mathematical report
├── results/
│   ├── Task1_Result.jpeg                   # Baseline stable performance charts
│   ├── Task2_Result.jpg                    # Concept drift stress-test comparison
│   └── Figure_1.png                        # Final comparative trajectory
└── src/
    ├── data_setup.py                       # Extreme non-IID MNIST partitioning
    ├── fedcads.py                          # FedCADS-UCB class and CUSUM logic
    ├── main.py                             # Local training loops & DivFL selection
    ├── models.py                           # PyTorch model architectures
    ├── shapley.py                          # Monte Carlo Shapley value estimation
    └── train_task3.py                      # Main execution and drift injection script
```

---

## Mathematical Core

The project's foundation relies on linking statistical process control (CUSUM) with cooperative game theory (Shapley values) and Multi-Armed Bandits (UCB).

* **Shapley Values ($SV_k$):** Measures the exact marginal contribution of client $k$ to the global model's validation accuracy. Approximated via Monte Carlo sampling over permutations $\pi$:
  $$SV_i = \frac{1}{R}\sum_{\pi} [v(\pi_{<i} \cup \{i\}) - v(\pi_{<i})]$$

* **CUSUM Drift Detection ($C_t$):** Accumulates evidence of accuracy degradation ($g_t$) beyond a noise threshold $\kappa$:
  $$C_t = \max(0, C_{t-1} + g_t - \kappa)$$

* **Adaptive Discount Factor ($\gamma$):** The core innovation. Modulates the bandit's memory based on the normalized drift signal, switching smoothly from long memory (0.95) to rapid forgetting (0.30):
  $$\gamma(t) = 0.95 - 0.65 \cdot \min\left(\frac{C_t}{h_{alarm}}, 1\right)$$

* **Fairness-Aware UCB Scoring:** Balances exploitation of high-performing clients with exploration and a fairness debt ($D_k$) to ensure minority classes are not forgotten:
  $$Score_k(t) = (1-\beta)\left[\frac{\tilde{R}_k}{\tilde{N}_k} + \sqrt{\frac{\ln(t+1)}{\tilde{N}_k}}\right] + \beta D_k(t)$$

---

## Methodology & Results

The project follows a rigorous, multi-stage methodology:

1. **Data Acquisition & Partitioning:** Fetches the MNIST dataset and heavily sorts it by label. Distributes shards so each of the 50 clients receives data for only 1 or 2 digit classes (Extreme Non-IID).
2. **Task 1 (Stable Environment Baseline):** Evaluates DivFL and S-FedAvg over 100 rounds without drift. DivFL converges faster initially due to forced diversity, but both reach comparable high accuracy (~96.8% for CNN).
3. **Task 2 (Concept Drift Vulnerability):** Injects sudden label-flip poisoning into 15 clients at round 50. DivFL fatally misinterprets the poisoned gradients as valuable diversity, collapsing to 58.43%. S-FedAvg penalizes the clients but recovers slowly due to its fixed memory.
4. **Task 3 (FedCADS-UCB Evaluation):** Our proposed algorithm detects the drift via CUSUM, shortens its memory $\gamma$, and places offending clients into exponential-backoff quarantine. **FedCADS-UCB recovers to 96.95% final accuracy**, vastly outperforming the baselines.

---

## Implementation Details

### `src/fedcads.py`

This module contains a fully self-contained `FedCADS` class written in Python with NumPy. It implements the core logic:
* `update_drift_and_gamma()` - Computes CUSUM statistics and dynamic $\gamma$.
* `get_selection_scores()` - Computes the UCB + Fairness scores for active clients.
* `update_client_stats()` - Processes Shapley values, decays historical stats, and handles exponential-backoff quarantine logic.

### `src/shapley.py`

Handles the cooperative game theory aspects:
* `compute_shapley_values()` - Implements the Monte Carlo approximation for marginal client contributions.
* `s_fedavg_select_clients()` - Implements the softmax-based selection logic for the S-FedAvg baseline.

---

## Installation

**Prerequisites:** Python 3.8 or higher.

**1. Clone the repository:**
```bash
git clone [https://github.com/Batman181/FedCADS-UCB-Client-Selection.git](https://github.com/Batman181/FedCADS-UCB-Client-Selection.git)
cd FedCADS-UCB-Client-Selection
```

**2. Install dependencies:**
It is highly recommended to use a virtual environment.
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install torch torchvision numpy matplotlib
```

---

## Usage

The project is designed to be executed via the central training script. 

Run the main experiment from the `src` directory:
```bash
cd src
python train_task3.py
```
This script will initialize 50 non-IID clients, train the models over 100 communication rounds, inject concept drift at round 50, output real-time quarantine alerts to the console, and generate the final comparative plot (`task3_fedcads_comparison.png`).

---

## References

1. Balakrishnan, R., et al. (2022). Diverse Client Selection for Federated Learning via Submodular Maximization. *ICLR*.
2. Nagalapatti, L., & Narayanam, R. (2021). Game of Gradients: Mitigating Irrelevant Clients in Federated Learning. *AAAI*, 35(10), 9046-9054.
3. Fouad, R., et al. (2020). Combinatorial Semi-Bandit in the Non-Stationary Environment. *UAI*.
4. Jothimurugesan, E., et al. (2023). Federated Learning under Distributed Concept Drift. *AISTATS*, 5834-5853.

---

## Acknowledgements

This project was developed as part of the coursework for **AI211 (Machine Learning Theory)** under the guidance and mentorship of **Dr. Shradha Sharma** at the Indian Institute of Technology (IIT) Ropar. It builds upon theoretical foundations in submodular optimization, cooperative game theory, and non-stationary multi-armed bandits. 

**Authors:** Kush Mistry (2024AIB1368), Rishi Datt Gupta (2024AIB1377), I. Nikhil Varma (2024AIB1350).

---

## License

This project is licensed under the **MIT License**.

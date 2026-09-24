# Metaheuristics for the Selective Capacitated Fair Pickup and Delivery Problem (SCF-PDP)

**Authors:** Lorena Lozano, Tina Taheri  
**Date:** November 2025  

## Overview
This repository contains the design, implementation, and analysis of various heuristic and metaheuristic approaches to solve the Selective Capacitated Fair Pickup and Delivery Problem (SCF-PDP). 

Unlike standard Vehicle Routing Problems (VRPs), the SCF-PDP requires selecting a subset of customer requests to serve from a pool, ensuring a minimum coverage requirement ($\gamma$). It uses a homogeneous fleet of vehicles with limited capacity ($C$). The composite objective function balances operational efficiency (minimizing total travel durations) with equity among drivers (penalizing workload imbalances using a Jain fairness index and a fairness weighting factor $\rho$).

## Implemented Algorithms
1. **Constructive Heuristics:**
   - Deterministic Greedy Construction
   - Randomized Greedy Construction
   - Beam Search
2. **Local Search Framework:**
   - Neighborhood 1 (N1): Intra-Route Relocation
   - Neighborhood 2 (N2): Inter-Route Relocation
   - Neighborhood 3 (N3): Swap Requests
   - Neighborhood 4 (N4): Add/Remove Requests
   - Step Functions: Best-Improvement & First-Improvement
3. **Advanced Metaheuristics:**
   - **Variable Neighborhood Descent (VND)** (Adaptive sampling for scalability)
   - **GRASP** (Greedy Randomized Adaptive Search Procedure)
   - **Tabu Search** (Dynamic tabu tenure and convergence tracking)

## Key Optimizations
- **PyTorch Vectorization:** Fast distance matrix computation using Euclidean distances.
- **Delta Evaluation:** Constant time $O(1)$ objective function recalculation for local search moves, avoiding full route re-evaluations.
- **Adaptive Neighborhood Sampling:** Dynamically reduces search space for extremely large instances (up to 10,000 requests) to satisfy strict runtime constraints.

## Full Report & Results
For a detailed explanation of the mathematical model, the implemented algorithms, parameter tuning, and a comprehensive analysis of the computational results, please refer to the **`SCF-PDP-Metaheuristics`** document included in this repository.

## Project Structure
- `VRP.py`: Core framework (Instance parser, Solution representation, Fairness computation).
- `det_cons_heu_delta.py` & `rand_cons_heu_delta.py`: Constructive heuristics.
- `beam_search.py`: Beam search constructive algorithm.
- `neighborhoods.py`: Generation logic for N1, N2, N3, and N4.
- `local_search.py`: Local search engine.
- `GRASP.py`, `VND.py`, `tabu_search.py`: Metaheuristic implementations.
- `batch_runner.py` & `competetion.py`: Benchmarking and competition execution scripts.
- `helper_report/`: Analytical scripts mapping results to CSVs and generating plots via Pandas and Matplotlib.

## Requirements
pip install numpy pandas matplotlib torch

## How to Run
To run the benchmark on competition instances using a specific algorithm (e.g., Beam Search or Tabu Search):
```bash
python competetion.py
```

To run batch evaluations across multiple instance sizes:

```bash
python batch_runner.py ./instances ./solutions greedy_tabu_best 1000
```

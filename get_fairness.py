import matplotlib.pyplot as plt
import pandas as pd
import json

def run_and_plot_tabu(inst, tabu_func, output_csv="tabu_log.csv"):
    """
    Runs a tabu_search variant function (e.g., greedy_tabu_best),
    stores objective + fairness in CSV, and plots them.
    """

    print("Running tabu search...")
    solution, data = tabu_func(inst)

    # Extract histories
    obj_hist = data["objective_history"]
    fair_hist = data["fairness_history"]
    iters = list(range(len(obj_hist)))

    # Build DataFrame
    df = pd.DataFrame({
        "iteration": iters,
        "objective": obj_hist,
        "fairness": fair_hist,
    })

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Saved log to {output_csv}")

    # ----------- PLOT OBJECTIVE OVER ITERATIONS ----------
    plt.figure(figsize=(10, 5))
    plt.plot(iters, obj_hist, linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Objective Value")
    plt.title("Tabu Search Convergence: Objective")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ----------- PLOT FAIRNESS OVER ITERATIONS ----------
    plt.figure(figsize=(10, 5))
    plt.plot(iters, fair_hist, linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("Fairness Value")
    plt.title("Tabu Search Convergence: Fairness")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return solution, df


from tabu_search_2 import greedy_tabu_first
from VRP import SCFPDPInstance

inst = SCFPDPInstance("/Users/tina/HOT-1/instances/2000/train/instance7_nreq2000_nveh40_gamma1889.txt")

solution, df = run_and_plot_tabu(inst, greedy_tabu_first)

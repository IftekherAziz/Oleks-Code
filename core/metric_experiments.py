import numpy as np
import pandas as pd

from core.distance_metric import uncertaintyDistance
from core.geometry import euclideanDistance, project_to_tangential, safe_cosine
from plots.plot_metric_experiments import plot_experiment

from sklearn.metrics import mean_squared_error


def compute_mse(gt_euclidean, obs_uncertainty, obs_euclidean, gt_cosine, obs_cosine):
    return {
        "uncertainty": mean_squared_error(gt_euclidean, obs_uncertainty),
        "observed": mean_squared_error(gt_euclidean, obs_euclidean),
        "cosine": mean_squared_error(gt_cosine, obs_cosine)
    }


def run_generic_experiment(x_values, compute_pA, compute_vA, compute_pB, compute_vB):
    gt_euclidean = []
    gt_cosine = []
    obs_cosine = []
    obs_euclidean = []
    obs_uncertainty = []

    last_positions = None
    last_velocities = None

    for x in x_values:
        pA = compute_pA(x)
        vA = compute_vA(x)
        pB = compute_pB(x)
        vB = compute_vB(x)

        vA_tan = project_to_tangential(pA, vA)
        vB_tan = project_to_tangential(pB, vB)

        gt_euclidean.append(euclideanDistance(vA, vB))
        gt_cosine.append(safe_cosine(vA, vB))
        obs_cosine.append(safe_cosine(vA_tan, vB_tan))
        obs_euclidean.append(euclideanDistance(vA_tan, vB_tan))
        obs_uncertainty.append(uncertaintyDistance(pA, vA_tan, pB, vB_tan))

        last_positions = (pA, pB)
        last_velocities = (vA, vB)

    return {
        "x": np.array(x_values),
        "gt_euclidean": np.array(gt_euclidean),
        "gt_cosine": np.array(gt_cosine),
        "obs_cosine": np.array(obs_cosine),
        "obs_euclidean": np.array(obs_euclidean),
        "obs_uncertainty": np.array(obs_uncertainty),
        "positions": last_positions,
        "velocities": last_velocities
    }


def run_all_experiments(data_dir, figures_dir):
    from core.metric_experiments_definitions import experiments

    total_mse_u = 0
    total_mse_obs = 0
    total_mse_cos = 0

    csv_rows = []

    for exp in experiments:
        name = exp["name"]
        x_vals = exp["x_values"]
        print(f"Running experiment {name}...")

        results = run_generic_experiment(
            x_values=x_vals,
            compute_pA=exp["compute_pA"],
            compute_vA=exp["compute_vA"],
            compute_pB=exp["compute_pB"],
            compute_vB=exp["compute_vB"]
        )

        mse = compute_mse(
            results["gt_euclidean"],
            results["obs_uncertainty"],
            results["obs_euclidean"],
            results["gt_cosine"],
            results["obs_cosine"])


        plot_experiment(
        results["x"],
        results["gt_euclidean"],
        results["gt_cosine"],
        results["obs_euclidean"],
        results["obs_uncertainty"],
        positions=results["positions"],
        velocities=results["velocities"],
        title=exp["title"],
        xlabel=exp["xlabel"],
        ylabel=exp["ylabel"],
        save_path=figures_dir / f"{exp['title'][:3].replace('.', '')}.png"
        )

        print(f"MSE (uncertainty-aware): {mse['uncertainty']:.3f}")
        print(f"MSE (observed euclidean): {mse['observed']:.3f}")
        print(f"MSE (cosine): {mse['cosine']:.3f}\n")

        csv_rows.append({
            "Experiment": name,
            "Uncertainty Metric": round(mse["uncertainty"], 2),
            "Observed Euclidean": round(mse["observed"], 2),
            "Observed Cosine": round(mse["cosine"], 2)
        })

        total_mse_u += mse["uncertainty"]
        total_mse_obs += mse["observed"]
        total_mse_cos += mse["cosine"]


    csv_rows.append({
        "Experiment":       "Total",
        "Uncertainty Metric":  round(total_mse_u, 2),
        "Observed Euclidean":     round(total_mse_obs, 2),
        "Observed Cosine":     round(total_mse_cos, 2)
    })

    df_summary = pd.DataFrame(csv_rows).round(3)
    data_dir.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(data_dir / "mse_summary.csv", index=False)

    print("========== Summary ==========")
    print(f"Total MSE (uncertainty-aware): {total_mse_u:.3f}")
    print(f"Total MSE (observed euclidean): {total_mse_obs:.3f}")
    print(f"Total MSE (observed cosine): {total_mse_cos:.3f}")

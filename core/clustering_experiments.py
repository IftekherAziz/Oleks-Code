import datetime

import pandas as pd
from core.clustering import clustering_evaluation
from core.config import load_experiment_configs
from synthetic_data_library import AstroDataGenerator


def clustering_experiments(config_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    configurations = load_experiment_configs(config_path)

    clustering_results = []
    noise_results = []

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    for config in configurations:
        generator = AstroDataGenerator(
            config["totalStars"], config["numberClusters"],
            config["clusterSizeMin"], config["clusterSizeMax"],
            config["noise"], config["noisePercentage"])

        df = generator.generateData()
        results = clustering_evaluation(df, synthetic_data=True)
        
        for method, result in results.items():
            clustering_results.append({
                "Case": config["name"],
                "Method": method,
                "n": config["totalStars"],
                "k": config["numberClusters"],
                "Noise [%]": round(config["noisePercentage"] * 100, 1),
                "AMI": round(result["AMI"] * 100, 2),
                "ARI": round(result["ARI"] * 100, 2)
            })

            noise_results.append({
                "Case": config["name"],
                "Method": method,
                "Prec": round(result["precision"] * 100, 2),
                "Rec": round(result["recall"] * 100, 2),
                "Noise": round(result["noise_fraction"] * 100, 2),
                "t": round(result["time"], 2)
            })

    clustering_df = pd.DataFrame(clustering_results)
    noise_df = pd.DataFrame(noise_results)
        
    desired_order = [
        "10% Noise", "20% Noise", "30% Noise", "40% Noise", "50% Noise",
        "60% Noise", "70% Noise", "80% Noise", "90% Noise",
        "Balanced Sizes", "Imbalanced Sizes", "Tiny Clusters",
        "Many Clusters + Noise", "Imbalanced + Noise"]

    clustering_pivot = clustering_df.pivot_table(
        index=["Case", "n", "k", "Noise [%]"],
        columns="Method", values=["ARI", "AMI"]
    ).reset_index()
    clustering_pivot.columns = [' '.join(col).strip() if col[1] else col[0] for col in clustering_pivot.columns]
    clustering_pivot["Case"] = pd.Categorical(clustering_pivot["Case"], categories=desired_order, ordered=True)
    clustering_pivot = clustering_pivot.sort_values("Case").round(3)

        
    noise_pivot = noise_df.pivot_table(
        index=["Case"], columns="Method", values=["Prec", "Rec", "Noise", "t"]
    ).reset_index()

    noise_pivot.columns = [' '.join(col).strip() if col[1] else col[0] for col in noise_pivot.columns]
    noise_pivot["Case"] = pd.Categorical(noise_pivot["Case"], categories=desired_order, ordered=True)
    noise_pivot = noise_pivot.sort_values("Case").round(3)
        
    clustering_pivot.to_csv(output_dir / f"clustering_summary_{timestamp}.csv", index=False)
    noise_pivot.to_csv(output_dir / f"clustering_summary_noise_{timestamp}.csv", index=False)
    return clustering_pivot, noise_pivot


    



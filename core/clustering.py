from time import time
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, pairwise_distances, precision_score, recall_score, silhouette_score
from core.distance_metric import compute_expected_distance_matrix_from_uncertainty
from core.motion import convert_proper_motions


def clustering_evaluation(df, eps=4, min_samples=4, verbose=False, synthetic_data=False):
    df = convert_proper_motions(df)

    df = df.dropna(subset=['X', 'Y', 'Z', 'v_ra_kms',
                   'v_dec_kms']).reset_index(drop=True)

    positions = df[['X', 'Y', 'Z']].values.astype(np.float64)
    vtans_2d = df[['v_ra_kms', 'v_dec_kms']].values.astype(np.float64)
    vtans_3d = np.hstack([vtans_2d, np.zeros((len(vtans_2d), 1))])
    ground_truth_velocities = df[["U", "V", "W"]].values
    if synthetic_data:
        true_labels = df["true_labels"].values

    results = {}
    methods = [
        ("d_u", compute_expected_distance_matrix_from_uncertainty),  # Uncertainty-aware
        ("GT", None),  # Ground Truth Euclidean
        ("Obs", None)  # Naive Euclidean
    ]

    for method, dist in methods:
        if (verbose):
            print(
                f"\nRunning DBSCAN using {method}, eps={eps}, min_samples={min_samples}")
        # Compute pairwise distance matrix
        start = time()
        if method == "d_u":
            D = dist(positions, vtans_3d, sigma_r=10,
                     sigma_t=0.8, use_speed_dependent_radial=True)
        elif method == "GT":
            D = pairwise_distances(ground_truth_velocities, metric='euclidean')
        elif method == "Obs":
            D = pairwise_distances(vtans_2d, metric='euclidean')
        else:
            continue

        finite_max = np.nanmax(D[~np.isnan(D)])
        D[np.isnan(D)] = finite_max

        # Clustering with DBSCAN
        labels = DBSCAN(eps=eps, min_samples=min_samples,
                        metric='precomputed').fit_predict(D)
        elapsed = time() - start


        # Metrics and stats for synthetic data
        if synthetic_data:
            mask = true_labels != -1
            ari = adjusted_rand_score(true_labels[mask], labels[mask])
            ami = adjusted_mutual_info_score(true_labels[mask], labels[mask])

            noise_gt = true_labels == -1
            noise_pred = labels == -1
            precision = precision_score(noise_gt, noise_pred, zero_division=0)
            recall = recall_score(noise_gt, noise_pred, zero_division=0)
            noise_frac = np.mean(noise_pred)
        else:
            ari = ami = precision = recall = np.nan
            noise_frac = np.mean(labels == -1)


        results[method] = {
            "labels": labels,
            "ARI": ari,
            "AMI": ami,
            "precision": precision,
            "recall": recall,
            "noise_fraction": noise_frac,
            "time": elapsed
        }
        if (verbose):
            print(f"ARI (no noise): {ari:.3f}")
            print(f"AMI (no noise): {ami:.3f}")
            print(f"Noise Detection – Precision: {precision:.2f}, Recall: {recall:.2f}")
            print(f"Noise Fraction: {noise_frac:.1%}")
            print(f"Time: {elapsed:.2f}s")
    if verbose:
        views = [(30, 45), (45, 75), (0, 45), (15, 15)]
        x, y, z = df['X'].values, df['Y'].values, df['Z'].values

        for method, result in results.items():
            print(f"\nPlotting clustering result for method: {method}")
            df["cluster"] = result["labels"]
            colors = df["cluster"].values

            fig = plt.figure(figsize=(14, 10))
            for i, (elev, azim) in enumerate(views):
                ax = fig.add_subplot(2, 2, i + 1, projection="3d")
                ax.scatter(0, 0, 0, c="black", s=15, label="Observer")

                is_noise = df["cluster"] == -1
                is_clustered = ~is_noise

                ax.scatter(x[is_clustered], y[is_clustered], z[is_clustered],
                           c=colors[is_clustered], cmap="tab10", s=5, alpha=0.9)
                ax.scatter(x[is_noise], y[is_noise], z[is_noise],
                           c="gray", s=5, alpha=0.4, label="Noise")

                ax.view_init(elev, azim)
                ax.set_xlabel("X [pc]")
                ax.set_ylabel("Y [pc]")
                ax.set_zlabel("Z [pc]")
                ax.set_title(f"{method} | Elevation={elev}°, Azimuth={azim}°")

            plt.suptitle(f"Clustering Result - Method: {method}", fontsize=16)
            plt.tight_layout()
            plt.show()

    return results


def evaluate_with_pseudo_labels(df, eps_test=4, min_samples=4, verbose=False,
                              sigma_r_values=[20], sigma_t_values=[5], beta_values=[0.1],
                              eps_gt_values=[2, 3, 4, 5, 10], min_samples_gt_values=[3, 4, 5, 7, 10]):
    df = convert_proper_motions(df)
    df = df.dropna(subset=['X', 'Y', 'Z', 'v_ra_kms', 'v_dec_kms', 'U', 'V', 'W']).reset_index(drop=True)

    positions = df[['X', 'Y', 'Z']].values.astype(np.float64)
    vtans_2d = df[['v_ra_kms', 'v_dec_kms']].values.astype(np.float64)
    vtans_3d = np.hstack([vtans_2d, np.zeros((len(vtans_2d), 1))])
    ground_truth_velocities = df[['U', 'V', 'W']].values.astype(np.float64)

    # Sweep for best GT clustering
    best_score = -np.inf
    best_labels_gt = None
    for eps_gt in eps_gt_values:
        for ms_gt in min_samples_gt_values:
            D_gt = pairwise_distances(ground_truth_velocities, metric='euclidean')
            labels = DBSCAN(eps=eps_gt, min_samples=ms_gt, metric='precomputed').fit_predict(D_gt)
            non_noise_mask = labels != -1
            if np.unique(labels[non_noise_mask]).size < 2:
                continue
            score = silhouette_score(D_gt[non_noise_mask][:, non_noise_mask], labels[non_noise_mask], metric='precomputed')
            if score > best_score:
                best_score = score
                best_labels_gt = labels
                best_eps_gt = eps_gt
                best_ms_gt = ms_gt

    true_labels = best_labels_gt
    mask = true_labels != -1

    results = {
        "GT_DBSCAN": {
            "eps": best_eps_gt,
            "min_samples": best_ms_gt,
            "silhouette": best_score,
            "labels": true_labels
        }
    }

    # Naive observed baseline
    if verbose:
        print(f"\nRunning DBSCAN using Obs, eps={eps_test}, min_samples={min_samples}")

    start = time()
    D_obs = pairwise_distances(vtans_2d, metric='euclidean')
    finite_max = np.nanmax(D_obs[~np.isnan(D_obs)])
    D_obs[np.isnan(D_obs)] = finite_max

    labels_obs = DBSCAN(eps=eps_test, min_samples=min_samples, metric='precomputed').fit_predict(D_obs)
    elapsed_obs = time() - start

    ari = adjusted_rand_score(true_labels[mask], labels_obs[mask])
    ami = adjusted_mutual_info_score(true_labels[mask], labels_obs[mask])
    noise_frac = np.mean(labels_obs == -1)

    results['Obs'] = {
        "labels": labels_obs,
        "ARI": ari,
        "AMI": ami,
        "noise_fraction": noise_frac,
        "time": elapsed_obs
    }

    if verbose:
        print(f"ARI: {ari:.3f}")
        print(f"AMI: {ami:.3f}")
        print(f"Noise Fraction: {noise_frac:.1%}")
        print(f"Time: {elapsed_obs:.2f}s")

    # Sweep 
    for sigma_r in sigma_r_values:
        for sigma_t in sigma_t_values:
            for beta in beta_values:
                label = f"d_u (sr={sigma_r}, st={sigma_t}, beta={beta})"
                if verbose:
                    print(f"\nRunning DBSCAN using {label}, eps={eps_test}, min_samples={min_samples}")

                start = time()
                D = compute_expected_distance_matrix_from_uncertainty(
                    positions, vtans_3d, sigma_r=sigma_r,
                    sigma_t=sigma_t, use_speed_dependent_radial=True, beta=beta)

                finite_max = np.nanmax(D[~np.isnan(D)])
                D[np.isnan(D)] = finite_max

                labels = DBSCAN(eps=eps_test, min_samples=min_samples, metric='precomputed').fit_predict(D)
                elapsed = time() - start

                ari = adjusted_rand_score(true_labels[mask], labels[mask])
                ami = adjusted_mutual_info_score(true_labels[mask], labels[mask])
                noise_frac = np.mean(labels == -1)

                results[label] = {
                    "labels": labels,
                    "ARI": ari,
                    "AMI": ami,
                    "noise_fraction": noise_frac,
                    "time": elapsed
                }

                if verbose:
                    print(f"ARI: {ari:.3f}")
                    print(f"AMI: {ami:.3f}")
                    print(f"Noise Fraction: {noise_frac:.1%}")
                    print(f"Time: {elapsed:.2f}s")

    return results
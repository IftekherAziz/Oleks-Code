import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def plot_experiment(x, gt_euclidean, gt_cosine, obs_euclidean, obs_uncertainty,
                    positions=None, velocities=None,
                    title="Generic Title", xlabel="X Label", ylabel="Y Label", save_path = None):

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [1, 2]})

    ax_pos = axes[0]
    if positions is not None:
        pA, pB = positions
        ax_pos.scatter(*pA[:2], color="blue", label="Star A", s=50)
        ax_pos.scatter(*pB[:2], color="darkorange", label="Star B", s=50)
        ax_pos.scatter(0, 0, color="dimgray", s=20, label="Observer")

        if velocities is not None:
            vA, vB = velocities
            ax_pos.quiver(*pA[:2], *vA[:2], color="blue",
                          angles='xy', scale_units='xy', scale=1, width=0.007)
            ax_pos.quiver(*pB[:2], *vB[:2], color="darkorange",
                          angles='xy', scale_units='xy', scale=1, width=0.007)

        ax_pos.set_xlim(-6, 6)
        ax_pos.set_ylim(-6, 6)
        ax_pos.set_aspect('equal')
        
        ax_pos.tick_params(axis='both', which='major', labelsize=14)
        ax_pos.set_title("Star Positions", fontsize=20)
        ax_pos.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
        ax_pos.legend(loc='upper right', fontsize=12)

    ax = axes[1]
    # Metrics for comparison
    ax.plot(x, gt_euclidean, color="navy", linestyle="--",
            linewidth=1.2, label="GT Euclidean")
    ax.plot(x, gt_cosine, color="skyblue", linestyle="--",
            linewidth=1, label="GT Cosine")
    ax.plot(x, obs_euclidean, color="forestgreen", linestyle="--",
            linewidth=1, label="Observed Euclidean")
    # Uncertainty Distance
    ax.plot(x, obs_uncertainty, color="red",
            linewidth=1.2, label="Uncertainty Metric")

    ax.set_xlabel(xlabel, fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.set_title(title, fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
    ax.legend(loc='upper right', fontsize=12)

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    mse_u = mean_squared_error(gt_euclidean, obs_uncertainty)
    mse_obs = mean_squared_error(gt_euclidean, obs_euclidean)
    mse_cos = mean_squared_error(gt_euclidean, gt_cosine)
    return mse_u, mse_obs, mse_cos




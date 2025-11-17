from matplotlib import pyplot as plt


def plot_gt_clusters(df, labels):
    x, y, z = df["X"].values, df["Y"].values, df["Z"].values
    df["cluster"] = labels
    colors = df["cluster"].values

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    is_noise = df["cluster"] == -1
    is_clustered = ~is_noise

    ax.scatter(x[is_clustered], y[is_clustered], z[is_clustered],
               c=colors[is_clustered], cmap="tab10", s=5, alpha=0.8)
    ax.scatter(x[is_noise], y[is_noise], z[is_noise],
               c="gray", s=5, alpha=0.3, label="Noise")

    ax.set_xlabel("X [pc]")
    ax.set_ylabel("Y [pc]")
    ax.set_zlabel("Z [pc]")
    ax.set_title("Ground Truth Pseudo-Clusters (3D Velocity-based DBSCAN)", fontsize=16)
    plt.tight_layout()
    plt.show()

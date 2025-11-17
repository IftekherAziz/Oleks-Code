from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import numpy as np

def plot_tangential_velocity_distribution(df):
    df = df.dropna(subset=['X', 'Y', 'Z', 'pmra', 'pmdec', 'parallax']).copy()

    # Distance in parsecs from Cartesian
    positions = df[['X', 'Y', 'Z']].values
    distances = np.linalg.norm(positions, axis=1)  # [pc]

    # Proper motion components [mas/yr]
    mu_ra = df['pmra'].values
    mu_dec = df['pmdec'].values
    mu_total = np.hypot(mu_ra, mu_dec)  # [mas/yr]

    # Tangential velocity in km/s
    tangential_velocities = distances * mu_total * 0.00474047

    # Standard deviation
    sigma_t = np.std(tangential_velocities)
    print(f"Standard Deviation of Tangential Velocity: {sigma_t:.2f} km/s")

    # Plotting
    fig = plt.figure(figsize=(12, 6))
    fig.suptitle("Distribution of Tangential Velocities", fontsize=18)

    # Histogram
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.hist(tangential_velocities, bins=30, edgecolor="black", alpha=0.8, color="orange")
    ax1.set_xlabel("Tangential velocity (km/s)", fontsize=14)
    ax1.set_ylabel("Number of Stars", fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=12)

    # Histogram + KDE
    kde = gaussian_kde(tangential_velocities)
    x = np.linspace(tangential_velocities.min(), tangential_velocities.max(), 200)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.hist(tangential_velocities, bins=30, density=True, edgecolor="black", alpha=0.8, color="orange")
    ax2.plot(x, kde(x), "black", linewidth=2)
    ax2.set_xlabel("Tangential velocity (km/s)", fontsize=14)
    ax2.set_ylabel("Density", fontsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    plt.show()
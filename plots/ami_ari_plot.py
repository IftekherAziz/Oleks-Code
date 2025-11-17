import pandas as pd
import matplotlib.pyplot as plt

def plot_ami_ari_vs_noise(csv_path, save_path="figures/ami_ari_comparison.png"):
    df = pd.read_csv(csv_path)
    noise_df = df[df["Case"].str.match(r"^\d+% Noise$")].copy()
    noise_df["Noise [%]"] = pd.to_numeric(noise_df["Noise [%]"], errors="coerce")
    noise_df = noise_df.sort_values("Noise [%]")

    plt.figure(figsize=(14, 6))

    # AMI plot
    plt.subplot(1, 2, 1)
    plt.plot(noise_df["Noise [%]"], noise_df["AMI GT"], label="GT", marker='o')
    plt.plot(noise_df["Noise [%]"], noise_df["AMI Obs"], label="Obs", marker='o')
    plt.plot(noise_df["Noise [%]"], noise_df["AMI d_u"], label="d_u", marker='o')
    plt.xlabel("Noise [%]", fontsize=14)
    plt.ylabel("AMI", fontsize=14)
    plt.title("AMI vs. Noise Level", fontsize=17)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=12)

    # ARI plot
    plt.subplot(1, 2, 2)
    plt.plot(noise_df["Noise [%]"], noise_df["ARI GT"], label="GT", marker='o')
    plt.plot(noise_df["Noise [%]"], noise_df["ARI Obs"], label="Obs", marker='o')
    plt.plot(noise_df["Noise [%]"], noise_df["ARI d_u"], label="d_u", marker='o')
    plt.xlabel("Noise [%]", fontsize=14)
    plt.ylabel("ARI", fontsize=14)
    plt.title("ARI vs. Noise Level", fontsize=17)
    plt.legend(fontsize=14)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

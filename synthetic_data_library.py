import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.cm as cm

from scipy.stats import gaussian_kde

class AstroDataGenerator:

    def __init__(self, totalStars = 1000, numberClusters = 5,
                 clusterSizeMin = 10, clusterSizeMax = 100,
                 noise = True, noisePercentage = 0.8):
        self.totalStars = totalStars
        self.numberClusters = numberClusters
        self.clusterSizeMin = clusterSizeMin
        self.clusterSizeMax = clusterSizeMax
        self.noise = noise
        self.noisePercentage = noisePercentage

        self.data = None
        self.colors = None
        
        self.df = None

    def __allocateClusterSizes(self, numberStars):
        C  = self.numberClusters
        mn = self.clusterSizeMin
        mx = self.clusterSizeMax

        if C*mn > numberStars or C*mx < numberStars:
            return np.random.multinomial(numberStars, [1/C]*C)

        sizes = np.random.multinomial(numberStars, [1/C]*C)

        sizes = np.clip(sizes, mn, mx)
        d = numberStars - sizes.sum()
        while d != 0:
            if d > 0:
                valid = np.where(sizes < mx)[0]
                i = np.random.choice(valid)
                sizes[i] += 1
                d -= 1
            else:
                valid = np.where(sizes > mn)[0]
                i = np.random.choice(valid)
                sizes[i] -= 1
                d += 1

        return sizes

    def __generateColors(self, labels):
        n_clusters = self.numberClusters
        unique_labels = set(labels)
        n_colors = len([label for label in unique_labels if label != -1])

        cmap = cm.get_cmap('tab20', n_colors + 1)

        color_dict = {label: mcolors.to_hex(cmap(i)) for i, label in enumerate(sorted(unique_labels)) if label != -1}
        color_dict[-1] = "#999999"

        return [color_dict[label] for label in labels]


    def generateData(self):
        points = []
        labels = []

        # Add noise stars
        if self.noise:
            n_noise = int(self.noisePercentage * self.totalStars)
            pos_noise = np.random.uniform(-100, 100, (n_noise, 3))  
            vel_noise = np.random.normal(0, 20, size=(n_noise, 3))
            
            points.append(np.column_stack((pos_noise, vel_noise)))
            labels.extend([-1] * n_noise)

            n_stars = self.totalStars - n_noise
        else:
            n_stars = self.totalStars

        
        # Add cluster stars
        clusterSizes = self.__allocateClusterSizes(n_stars)
        clusterCenters = np.random.uniform(-100, 100, (self.numberClusters, 3))
        for i, size in enumerate(clusterSizes):

            cov = np.diag(np.random.uniform(20, 50, size=3))
            coords = np.random.multivariate_normal(clusterCenters[i], cov, size)

            v_rand = np.random.normal(0, 20, 3)
            disp = np.random.normal(0, 1, (size, 3))
            vel = (v_rand + disp)
            
            clusterData = np.column_stack((coords, vel))
            points.append(clusterData)
            labels.extend([i] * size)

        self.data = np.vstack(points)
        labels = np.array(labels)
        self.colors = self.__generateColors(labels)
        
        dataTransformer = AstroDataTransformer()
        sphericalData = dataTransformer.transformData(self.data)
        
        X, Y, Z = self.data[:, 0], self.data[:, 1], self.data[:, 2]
        U, V, W = self.data[:, 3], self.data[:, 4], self.data[:, 5]
        
        dist = 1 / (sphericalData[:, 2] / 1000)
        mu_ra, mu_dec = sphericalData[:, 3], sphericalData[:, 4]
        v_a_lsr = mu_ra * dist * 4.74047
        v_d_lsr = mu_dec * dist * 4.74047
        
        df = pd.DataFrame({
            'ra': sphericalData[:,0], 'dec': sphericalData[:,1],
            'parallax': sphericalData[:,2],
            'pmra': sphericalData[:,3], 'pmdec': sphericalData[:,4],
            'radial_velocity': sphericalData[:,5],
            'X': X, 'Y': Y, 'Z': Z,
            'U': U, 'V': V, 'W': W,
            'v_a_lsr': v_a_lsr, 'v_d_lsr': v_d_lsr
        })
        
        n = len(df)
        df['ra_error'] = np.abs(np.random.normal(0.03, 0.01, n))
        df['dec_error'] = np.abs(np.random.normal(0.03, 0.01, n))
        df['parallax_error'] = np.abs(np.random.normal(0.04, 0.01, n))
        df['pmra_error'] = np.abs(np.random.normal(0.05, 0.02, n))
        df['pmdec_error'] = np.abs(np.random.normal(0.05, 0.02, n))
        df['radial_velocity_error'] = np.abs(np.random.normal(1.0, 0.5, n))
        
        correlations = [
            'ra_dec_corr','ra_parallax_corr','ra_pmra_corr','ra_pmdec_corr',
            'dec_parallax_corr','dec_pmra_corr','dec_pmdec_corr',
            'parallax_pmra_corr','parallax_pmdec_corr','pmra_pmdec_corr'
        ]
        for name in correlations:
            df[name] = np.random.uniform(-0.1, 0.1, n)
        df['radial_velocity_corr'] = 0
        
        df["true_labels"] = np.array(labels)
        self.df = df
        self.true_labels = np.array(labels)
        return df


class AstroDataTransformer:

    def __transformVelocityComponents(self, theta, phi, vel):
        T = np.array([
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)],
            [np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)],
            [-np.sin(phi), np.cos(phi), 0]
        ])
        
        v_radial = np.dot([np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), np.sin(theta)], vel)
        v_theta = np.dot(T[1], vel)
        v_phi = np.dot(T[2], vel)
        
        return v_radial, v_theta, v_phi
    
    def transformDataPointToSphericalCoordinates(self, point):
        x, y, z = point[:3]
        vel = point[3:]
        r = np.linalg.norm([x, y, z])
        
        theta = np.arccos(z/r) 
        phi = np.arctan2(y, x)
        
        v_radial, v_theta, v_phi = self.__transformVelocityComponents(theta, phi, vel)
        
        parallax = (1 / r) * 1000

        mu_ra = (v_phi / (r * np.sin(theta))) * 211
        mu_dec = (-v_theta / r) * 211

        dec = np.degrees(np.pi/2 - theta)
        ra = np.degrees(phi) % 360
        
        return np.array([ra, dec, parallax, mu_ra, mu_dec, v_radial])
    
    def transformData(self, data):
        return np.array([self.transformDataPointToSphericalCoordinates(point) for point in data])
    

from IPython.display import HTML  
class AstroDataPlotter:
    def __init__(self, data, colors):
        self.data = data
        self.colors = colors
        self.transformer = AstroDataTransformer()

    def plotStarChart(self):
        spherical = self.transformer.transformData(self.data[:, :6])
        
        ra, dec = spherical[:, 0], spherical[:, 1]
        pmra, pmdec = spherical[:, 3], spherical[:, 4]
        
        vec_norm = np.hypot(pmra, pmdec)
        u = pmra / vec_norm
        v = pmdec / vec_norm

        fig = plt.figure(figsize=(14, 7))
        ax2d = fig.add_subplot()
        
        ax2d.scatter(ra, dec, color=self.colors, s=4)
        ax2d.quiver(ra, dec, u, v, angles="xy", scale_units="xy", color=self.colors, alpha=0.6)
        ax2d.set_xlabel("Right Ascension (RA)")
        ax2d.set_ylabel("Declination (Dec)")
        ax2d.set_title("Sky map with proper motions")
        ax2d.invert_xaxis()
        ax2d.set_xlim(360, 0)
        ax2d.set_ylim(-90, 90)
        ax2d.set_aspect('equal')
        ax2d.grid()
        plt.show()


    def plotDataFromDifferentPerspectives(self):
        fig = plt.figure(figsize=(12, 16), layout="constrained")
        fig.suptitle("3D positions from multiple views", fontsize=18)
        
        x, y, z = self.data[:, 0], self.data[:, 1], self.data[:, 2]
        views = [(30,45), (45,75), (0,45), (15, 15)]
        for i, (elev, azim) in enumerate(views):
            ax = fig.add_subplot(2, 2, i + 1, projection="3d")

            ax.scatter(0, 0, 0, c="black", s=15)
            ax.scatter(x, y, z, c=self.colors, s=5)
            ax.view_init(elev, azim)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title(f"Elevation={elev}, azimuth={azim}", fontsize=14)

        plt.show()


    def plotDataAnimation(self):
        def drawFrame(n):
            ax.view_init(30, n)

        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(projection="3d")

        x, y, z = self.data[:, 0], self.data[:, 1], self.data[:, 2]
        ax.scatter(x, y, z, c=self.colors, s=7)
        ax.scatter(0, 0, 0, c="black", s=10)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("3D Star Simulation")

        plt.close(fig)
            
        anim = animation.FuncAnimation(fig, drawFrame, frames=np.arange(0, 360, 2), interval=50)
        return HTML(anim.to_html5_video())
    
    def plotSpeedDistributionCartesian(self):
        speeds = np.linalg.norm(self.data[:, 3:6], axis=1)
        kde = gaussian_kde(speeds)
        
        x = np.linspace(speeds.min(), speeds.max(), 100)
        
        # First figure
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(1, 2, 1)
        ax.hist(speeds, bins=30,edgecolor="black", alpha=0.8)
        ax.set_xlabel("Speed (km/s)")
        ax.set_ylabel("Number of Stars")
        ax.set_title("Distribution of Star Speed in XYZ")
        pdf_kde = kde(x)
        # Second figure
        ax2 = fig.add_subplot(1,2,2)
        ax2.hist(speeds, bins=30, edgecolor='black', alpha=0.8, density=True)
        ax2.plot(x, pdf_kde, "black", linewidth=2, label="KDE Estimation")
        ax2.set_xlabel("Speed (km/s)")
        ax2.set_ylabel("Speed Density")
        ax2.set_title("Speed Densities")
        
        plt.show()

    def plotSpeedDistributionSpherical(self):
        spherical = self.transformer.transformData(self.data[:, :6])
        magnitude = np.hypot(spherical[:, 3], spherical[:, 4])

        fig = plt.figure(figsize=(12, 6))
        fig.suptitle("Distribution of Proper Motion")
        
        # First figure
        ax = fig.add_subplot(1, 2, 1)
        ax.hist(magnitude, bins=30, edgecolor="black", alpha=0.8, color="green")
        ax.set_xlabel("Proper Motion (mas/year)")
        ax.set_ylabel("Number of Stars")
        # Second figure
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.hist(magnitude, bins=30, density=True, edgecolor="black", alpha=0.8, color="green")
        kde = gaussian_kde(magnitude)
        x = np.linspace(magnitude.min(), magnitude.max(), 100)
        ax2.plot(x, kde(x), "black", linewidth=2)
        
        ax2.set_xlabel("Proper Motion (mas/year)")
        ax2.set_ylabel("Density")
        
        plt.show()


    def plotSpeedDistributionTangential(self):
        cartesian = self.data[:, :6]
        spherical = self.transformer.transformData(self.data[:, :6])

        distances = np.linalg.norm(cartesian[:, :3], axis=1)
        
        parallax = spherical[:, 2]            
        mu_ra = spherical[:, 3]           
        mu_dec = spherical[:, 4]     
        
        mu_total = np.hypot(mu_ra, mu_dec)     
  
        tan_vel = distances * mu_total * 0.00474047

        sigma_t = np.std(tan_vel)

        print(f"Standard Deviation of Tangential Velocity: {sigma_t}")

        fig = plt.figure(figsize=(12, 6))
        fig.suptitle("Distribution of Tangential Velocities")
        
        # First figure
        ax = fig.add_subplot(1, 2, 1)
        ax.hist(tan_vel, bins=30, edgecolor="black", alpha=0.8, color="orange")
        ax.set_xlabel("Tangential velocity(km/s)")
        ax.set_ylabel("Number of Stars")

        # Second figure
        kde = gaussian_kde(tan_vel)
        x = np.linspace(tan_vel.min(), tan_vel.max(), 100)
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.hist(tan_vel, bins=30, density=True, edgecolor="black", alpha=0.8, color="orange")
        ax2.plot(x, kde(x), "black", linewidth=2)
        ax2.set_xlabel("Tangential velocity(km/s)")
        ax2.set_ylabel("Density")

        plt.show()
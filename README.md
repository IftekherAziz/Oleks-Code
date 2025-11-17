# Uncertainty in Density-Based Clustering for Astrophysics data
This Python project was developed as a part of my bachelor thesis. It includes:
- Generation of synthetic Gaia-like data
- Definition and implementation of uncertainty-aware metric
- Metric evaluation experiments
- Clustering experiments using DBSCAN
- Interactive GUI tool for data generation and visual exploration

## Project Structure
| Folder      | Description                                           |
|-------------|-------------------------------------------------------|
| core/       | Core logic: uncertainty-aware metric, geometry, clustering |
| plots/      | Plotting utilities                                    |
| astro_gui/  | Interactive GUI for data generation                   |
| notebooks/  | Jupyter notebooks for showcases and experiments       |
| data/       | CSV outputs, Gaia subset                              |
| figures/    | Generated plots                                       |
| config/     | YAML configuration files for experiments              |

## Requirements
Install all dependencies using
```
pip install -r requirements.txt
```

## Running
Launch one of the notebooks, those include:
- Clustering Experiments
- Metric Evaluation Experiments
- Gaia Data Demonstration
- Interactive GUI for data generation
- Cartesian to Spherical conversion (text-only)

import ipywidgets as widgets
from ipywidgets import VBox, HBox, Button, Output

totalStars_slider = widgets.IntSlider(value=1000, min=100, max=5000, step=100, description='Total Stars')
numberClusters_slider = widgets.IntSlider(value=5, min=1, max=10, step=1, description='Clusters')
noise_toggle = widgets.Checkbox(value=False, description='Noise')
noisePercentage_slider = widgets.FloatSlider(value=0.8, min=0.0, max=1.0, step=0.05, description='Noise %')
clusterSizeMin_slider = widgets.IntSlider(value=10, min=1, max=200, step=1, description='Min Cluster')
clusterSizeMax_slider = widgets.IntSlider(value=100, min=10, max=300, step=1, description='Max Cluster')

plotType_dropdown = widgets.Dropdown(
    options=["2D Chart", "3D Views", "3D Animation", "Speed XYZ", "Proper Motion", "Tangential Speed"],
    value="3D Views",
    description="Plot Type:")

generate_button = Button(description="Generate Data", button_style='success')
output_plot = Output()


def build_ui():
    return VBox([
    HBox([totalStars_slider, numberClusters_slider]),
    HBox([noise_toggle, noisePercentage_slider]),
    HBox([clusterSizeMin_slider, clusterSizeMax_slider]),
    HBox([generate_button, plotType_dropdown]),
    output_plot])

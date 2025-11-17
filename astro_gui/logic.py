from .widgets import *
from synthetic_data_library import AstroDataGenerator, AstroDataPlotter
from IPython.display import display

generated_data = {"data": None, "colors": None, "df": None, "true_labels" : None}

def update_plot(change=None):
    output_plot.clear_output()
    if generated_data["data"] is None:
        with output_plot:
            print("Please generate data first.")
        return

    plotter = AstroDataPlotter(generated_data["data"], generated_data["colors"])
    with output_plot:
        if plotType_dropdown.value == "2D Chart":
            plotter.plotStarChart()
        elif plotType_dropdown.value == "3D Views":
            plotter.plotDataFromDifferentPerspectives()
        elif plotType_dropdown.value == "3D Animation":
            display(plotter.plotDataAnimation())
        elif plotType_dropdown.value == "Speed XYZ":
            plotter.plotSpeedDistributionCartesian()
        elif plotType_dropdown.value == "Proper Motion":
            plotter.plotSpeedDistributionSpherical()
        elif plotType_dropdown.value == "Tangential Speed":
            plotter.plotSpeedDistributionTangential()

def generate_data(_):
    generator = AstroDataGenerator(
        totalStars=totalStars_slider.value,
        numberClusters=numberClusters_slider.value,
        clusterSizeMin=clusterSizeMin_slider.value,
        clusterSizeMax=clusterSizeMax_slider.value,
        noise=noise_toggle.value,
        noisePercentage=noisePercentage_slider.value
    )
    generator.generateData()
    generated_data["data"] = generator.data
    generated_data["colors"] = generator.colors
    generated_data["df"] = generator.df
    generated_data["true_labels"] = generator.true_labels
    update_plot()
    
def setup_callbacks():
    generate_button.on_click(generate_data)
    plotType_dropdown.observe(update_plot, names='value')
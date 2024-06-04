import time
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt
from PIL import Image
from gradio.themes.utils import sizes
from gradio.components.image_editor import Brush

from planet_parasol_demo.constants import *
from planet_parasol_demo.inputs import get_co2_input, get_sai_input
from planet_parasol_demo.fair_utils import run, get_dataframes
from planet_parasol_demo.plot import *


css = """
.my-accordion .svelte-s1r2yt:hover {
    color: red;
}
.my-button {
    font-weight: bold !important;
    border-radius: 0.5em;
}
.launch-container > div {
    background-color: rgba(130, 130, 140, 0.01);
    border-radius: 0.5em;
    padding: 0.3em;
}
.launch-markdown {
    margin-bottom: 0.5em;
}
.generating {
    border: none;
}
"""

js_func = """
function refresh() {
    const prefersDarkMode = window.matchMedia('(prefers-color-scheme: dark)').matches;
    
    if (prefersDarkMode) {
        const url = new URL(window.location);

        if (url.searchParams.get('__theme') == 'dark') {
            url.searchParams.set('__theme', 'light');
            window.location.href = url.href;
        }
    }
}
"""

global_plot_title = "### Global Temperature Projection (Compared to Preindustrial) with and without SAI"
global_running_status = "Getting global temperature projections..."
global_complete_status = "✔️ Global temperature projections complete!"


def set_header():
    gr.Markdown("# Global Cooling Forecasts from Stratospheric Aerosol Injection")
    # gr.Markdown("## What is stratospheric aerosol injection (SAI)?")
    gr.Markdown("\n\n")
    gr.Markdown("### Stratospheric aerosol injection (SAI) uses reflective aerosols released into the upper atmosphere to reflect sunlight and thereby cool Earth's surface.")
    gr.Markdown(
        "Despite the strong potential of SAI to lower Earth's average surface temperature and mitigate global warming's short-term effects:\n" +
        "- SAI cannot “turn back the clock” on climate change: it is not a substitute for reducing greenhouse gas emissions, which is critical to address climate change and achieve long-term sustainability\n" +
        "- There are potential risks that may arise from SAI, so more research is necessary to better understand those risks and compare them to the potential benefits "
    )
    with gr.Accordion(label="▶️ Click here to learn more about the challenges and considerations of SAI", open=False, elem_classes=["my-accordion"]):
        gr.Markdown(
            "There are many open questions relating to the effects of SAI that need further study prior to potential implementation. These include:\n" +
            "- The cooling efficiency of sulphate injections, due to uncertain: \n" +
            "  - Chemical reactions with other matter in the stratosphere \n" +
            "  - Lifetime and spread of sulfate in the atmosphere \n" +
            "  - Impact of sulfate on sunlight and heat redistribution in the stratosphere and at the surface\n" +
            "  - Relationship of the above with seasonal weather patterns\n" +
            "- Regional and side effects of SAI, such as: \n" +
            "  - Frequency of extreme temperature and precipitation events, such as  droughts and floods\n" +
            "  - Health impacts of the sulfate that eventually falls out of the atmosphere as well as impacts on morbidity and mortality with reduced heating.\n" +
            "  - Impact on food production, on spread of infectious diseases, and on animal populations \n" +
            "- Appropriate global governance in implementation, to avoid unilateral deployment, unfair deployment distribution, and other geopolitical risks \n" +
            "- Engineering needed to build infrastructure for SAI implementation\n\n" +
            "- The exact cost of SAI, although it is estimated to be well under 1% of the cost of climate change related damages.\n\n" +
            "For more detail about the science behind SAI, see [this United Nations report](https://www.unep.org/resources/report/Solar-Radiation-Modification-research-deployment)."
        )


def set_launch():
    gr.Markdown("\n\n")
    with gr.Group(elem_classes=["launch-container"]):
        # gr.Markdown(r"## Launch the emulator to get SAI cooling forecasts! 🚀")
        gr.Markdown("### Click the button below to get the global temperature projections with and without SAI!", elem_classes=["launch-markdown"])
        gr.Markdown(f"This tool uses the [FaIR climate emulator](https://github.com/OMS-NetZero/FAIR) to enable you to explore the impact of different CO$_2$ emissions and SAI scenarios. FaIR was developed by climate scientists at the University of Leeds and Oxford University.", latex_delimiters=[{"left": "$", "right": "$", "display": False}], elem_classes=["launch-markdown"])
        launch_button = gr.Button("▶️ Launch the emulator! 🚀", elem_classes=["my-button", "launch-markdown"], variant='primary')
        # it wouldn’t simply “turn back the clock” on climate change: 
        gr.Markdown(f"While SAI could reduce _global_ average temperature, there are major uncertainties in the _regional_ temperature response to SAI, where some regions could experience residual warming or overcooling.", elem_classes=["launch-markdown"])
        regional_checkbox = gr.Checkbox(label="Check this box to also generate regional temperature projections. Note we cannot generate regional projections for drawn scenarios.", elem_classes=["launch-markdown"], container=False)
        uncertainty_checkbox = gr.Checkbox(label="Check this box to turn on regional temperature uncertainty estimates, which indicate where there is substantial disagreement among different climate models.", elem_classes=["launch-markdown"], container=False)

    return launch_button, regional_checkbox, uncertainty_checkbox



def set_footer():
    gr.Markdown("\n\n")
    gr.Markdown("---")
    gr.Markdown("\n\n")
    gr.Markdown(r"## Methodology 📚")
    with gr.Accordion(label="▶️ Click here to learn more about the methodology of the global and regional emulators, including important assumptions we make.", open=False, elem_classes=["my-accordion"]):
        gr.Markdown("The code for this demo and the underlying emulators are available as open source software [here](https://github.com/stanfordmlgroup/planet_parasol_demo).")
        gr.Markdown("**Global Emulator**")
        gr.Markdown(
            r"- The global emulator uses [FaIR](https://docs.fairmodel.net/en/latest/intro.html)." + "\n\n" +
            "- The sensitivity of the climate emulator is the median of the IPCC assessed range ([source](https://gmd.copernicus.org/articles/11/2273/2018/)).\n" +
            "- We assume a forcing efficiency 0.28 W/m$^2$ per Tg SO$_2$/yr ([source1](https://www.pnnl.gov/sites/default/files/media/file/Sensitivity%20of%20Aerosol%20Distribution%20and%20Climate%20Response%20to%20Stratospheric%20SO2%20Injection%20Locations.pdf),[source2](https://www.google.com/url?q=https://acp.copernicus.org/articles/21/10039/2021/&sa=D&source=docs&ust=1715284226685975&usg=AOvVaw38Ib3Gc0XRuSme39sOh_tz)).\n" + 
            "- We use a 10-year SAI ramp-up.\n",# + 
            latex_delimiters=[{"left": "$", "right": "$", "display": False}]
        )
        gr.Markdown("**Regional Emulator**")
        gr.Markdown(
            "We use publicly available climate model data to develop the regional emulator. The emulator is equivalent to pattern scaling, and consists of two major components:\n"
            "1. A linear regression from the [FaIR](https://docs.fairmodel.net/en/latest/intro.html) global mean temperature to the [ScenarioMIP](https://gmd.copernicus.org/articles/9/3461/2016/) regional temperature. We train one linear regression for six different climate models, namely " + ", ".join(REGIONAL_MODEL_NAMES[:-1]) + f", and {REGIONAL_MODEL_NAMES[-1]}.\n"
            r"2. A linear regression from the global mean stratospheric aerosol optical depth (AOD), assuming 0.01 AOD per Tg SO$_2$, to the regional temperature difference. We train one linear regression for CESM2-WACCM using simulation data from [GeoMIP](https://climate.envsci.rutgers.edu/geomip/data.html) and [ARISE-SAI](https://www.cesm.ucar.edu/community-projects/arise-sai)." + "\n\n"
            "We try to estimate the substantial uncertainties in regional temperature forecasts using the following methods:\n" +
            "- _Climate model uncertainty_: We compute the standard deviation of the regional values across the 6 climate models.\n" +
            "- _Emulator uncertainty_: We compute the standard deviation of the regional values across 100 bootstrapped linear regression emulators.\n" +
            "- _Natural variability_: We compute the standard deviation of the regional values across 100 ensemble members from FaIR.\n\n" +
            "You can find more details about the regional emulator in [the open-source code](https://github.com/stanfordmlgroup/planet_parasol_demo).",
            latex_delimiters=[{"left": "$", "right": "$", "display": False}]
            )

    gr.Markdown(r"## Contributors 🧑‍💻")
    with gr.Accordion(label="▶️ Click here to see the contributors to this project.", open=False, elem_classes=["my-accordion"]):
        gr.Markdown("[Jeremy Irvin](https://twitter.com/jeremy_irvin16), Stanford University")
        gr.Markdown("[Daniele Visioni](https://twitter.com/DanVisioni), Cornell University")
        gr.Markdown("[Ben Kravitz](https://earth.indiana.edu/directory/faculty/kravitz-ben.html), Indiana University")
        gr.Markdown("[Dakota Gruener](https://twitter.com/dakotagruener), Reflective")
        gr.Markdown("[Chris Smith](https://twitter.com/chrisroadmap), University of Leeds")
        gr.Markdown("[Duncan Watson-Parris*](https://twitter.com/DWatsonParris), University of California San Diego")
        gr.Markdown("[Andrew Ng*](https://twitter.com/AndrewYNg), Stanford University")
        gr.Markdown("\* Co-supervisors.") 
    
    gr.Markdown(r"## Contact Us 📧")
    gr.Markdown("If you have any questions or feedback, you reach out to us at [planetparasol@cs.stanford.edu](mailto:planetparasol@cs.stanford.edu).")
    gr.Markdown("---")
    gr.Markdown(r"Copyright © 2024 Stanford Machine Learning Group")
    with gr.Accordion(label="▶️ MIT License", open=False, elem_classes=["my-accordion"]):
        gr.Markdown(
            """
            Permission is hereby granted, free of charge, to any person obtaining a copy
            of this software and associated documentation files (the "Software"), to deal
            in the Software without restriction, including without limitation the rights
            to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
            copies of the Software, and to permit persons to whom the Software is
            furnished to do so, subject to the following conditions:

            The above copyright notice and this permission notice shall be included in all
            copies or substantial portions of the Software.

            THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
            IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
            FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
            AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
            LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
            OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
            SOFTWARE.
            """
        )


# Sample data generation function for demonstration
def generate_data(label):
    np.random.seed(sum(ord(c) for c in label))  # Seed for reproducibility
    return np.cumsum(np.random.randn(100))  # Random walk data


def sketch2data(sketch, min_y, max_y, min_x, max_x, max_value):
    """
    Inputs the sketchpad data (a binary boolean mask) and return the (x, y) data points where True.

    Specifically, computes the median y coordinate for each x coordinate.
    """
    # Get the mask from the sketchpad
    mask = sketch['layers'][0]

    mask = mask[min_y:max_y+1, min_x:max_x+1]

    height, width = mask.shape
    window_size = 5
    new_width = CANVAS_END_YEAR - CANVAS_START_YEAR + 1
    smoothed_curve = np.zeros(width, dtype=int)
    for x in range(width):
        start = max(0, x - window_size // 2)
        end = min(width, x + window_size // 2 + 1)
        window_values = []
        for wx in range(start, end):
            non_zero_indices = np.where(mask[:, wx] > 0)[0]
            if non_zero_indices.size > 0:
                window_values.extend(non_zero_indices)
        if window_values:
            smoothed_curve[x] = int(np.median(window_values))
        else:
            smoothed_curve[x] = height
    original_width = len(smoothed_curve)
    original_indices = np.arange(original_width)
    new_indices = np.linspace(0, original_width - 1, new_width)
    interpolated_data = np.interp(new_indices, original_indices, smoothed_curve)

    x = np.arange(CANVAS_START_YEAR, CANVAS_END_YEAR + 1)
    y = (1 - interpolated_data / height) * max_value

    return np.column_stack((x, y))


def visible_component(co2_selection, sai_selection, custom_co2_data, custom_sai_data):
    return gr.Markdown(
        global_plot_title + "\n\n" + global_running_status,
        visible=True
    ), gr.Plot(visible=True), gr.Markdown(visible=False), gr.Image(visible=False)



def update_plot(co2_selection, sai_selection, custom_co2_data, custom_sai_data,
                regional_checkbox_value, uncertainty_checkbox):
    co2_selection = REVERSE_FANCY_SSP_TITLES_SUFFIXES[co2_selection]
    fig, ax = plt.subplots(figsize=(10, 5))

    co2_is_drawn = "Draw" in co2_selection
    sai_is_drawn = "Draw" in sai_selection
    neither_scenario_drawn = not co2_is_drawn and not sai_is_drawn
    if neither_scenario_drawn:
        _sai_selection = sai_selection.split(" (")[0].replace("Injection", "SAI")
        _co2_selection = REVERSE_FANCY_SSP_TITLES[co2_selection]
        temp_no_sai_path = TEMP_DATA_DIR / f"{_co2_selection}_{_sai_selection}_no_sai.nc"
        temp_sai_path = TEMP_DATA_DIR / f"{_co2_selection}_{_sai_selection}_sai.nc"
        _temp_no_sai = xr.open_dataarray(temp_no_sai_path)
        _temp_sai = xr.open_dataarray(temp_sai_path)
        # Define a dummy generator that just yields the precomputed temperature data
        def dummy_generator():
            # Iterate through the temperature data one time step at a time
            for time in range(2, _temp_no_sai.timebounds.size+1):
                yield _temp_no_sai[:time], _temp_sai[:time]

        runner = dummy_generator()

    else:

        if co2_is_drawn and custom_co2_data is not None:
            co2_input = sketch2data(custom_co2_data, 62, 447, 704, 1261, MAX_CO2)
        else:
            co2_value = REVERSE_FANCY_SSP_TITLES[co2_selection]
            co2_through_2100 = get_co2_input(
                df_emis,
                co2_value,
                start_year=PLOT_START_YEAR,
                end_year=PLOT_END_YEAR
            )
            co2_input = np.column_stack((range(PLOT_START_YEAR, PLOT_END_YEAR+1), co2_through_2100))
            co2_input = co2_input[co2_input[:, 0] >= CANVAS_START_YEAR]

        if sai_is_drawn and custom_sai_data is not None:
            sai_input = sketch2data(custom_sai_data, 58, 443, 701, 1258, MAX_SAI)
        else:
            sai_input = get_sai_input(sai_selection.split(" (")[0])

        if co2_is_drawn:
            # Replace scenario in the initial conditions with the base scenario
            for key in list(initial_conditions.keys()):
                initial_conditions[key].coords["scenario"] = [BASE_SCENARIO]
        else:
            # Replace scenario in the initial conditions with the ssp scenario
            for key in list(initial_conditions.keys()):
                initial_conditions[key].coords["scenario"] = [REVERSE_FANCY_SSP_TITLES[co2_selection]]

        runner = run(
            df_emis, df_configs,
            df_solar, df_volcanic,
            co2_input, sai_input,
            co2_selection,
            initial_conditions=initial_conditions
        )

    no_sai_label = "No SAI"
    if co2_is_drawn:
        no_sai_label = f"No SAI (Your CO$_2$ scenario)"
    else:
        no_sai_label = f"No SAI ({co2_selection})"

    if not co2_is_drawn:
        co2_selection = REVERSE_FANCY_SSP_TITLES[co2_selection]
    else:
        co2_selection = BASE_SCENARIO

    sai_label = "SAI"
    sai_selection_simple = sai_selection.split("Injection")[0] + "SAI"

    if not sai_is_drawn:
        sai_label = sai_label.replace("SAI", sai_selection_simple)
    elif sai_selection is not None and sai_is_drawn:
        sai_label = "Your SAI Scenario"

    label2linestyle = {
        no_sai_label: "-",
        sai_label: "--"
    }

    fig = initialize_temp_plot(fontsize+8)
    plt.close(fig)
    figure = copy.deepcopy(fig)
    yield gr.Markdown(global_plot_title + "\n\n" + global_running_status, visible=True), figure, gr.Markdown(visible=False), gr.Image(visible=False)
    if neither_scenario_drawn:
        time.sleep(2)
    ax = figure.axes[0]
    lines = {
        label: ax.plot([],[], color="black", linestyle=linestyle, label=label)[0]
        for label, linestyle in label2linestyle.items()
    }

    fill = {
        label: ax.fill_between([],[],[], color="black", alpha=0.2, linestyle=linestyle, lw=1,
                                    label="90% Confidence Interval" if label == no_sai_label else None)
        for label, linestyle in label2linestyle.items()
    }
    year = SIM_START_YEAR
    for temp_no_sai, temp_sai in runner:
        if year == PLOT_START_YEAR:
            ax.legend(fontsize=fontsize+8, loc="upper left")
        if year >= CANVAS_START_YEAR and (year % 15 == 0 or year == CANVAS_END_YEAR-1):
            for temp, label in [(temp_no_sai, no_sai_label), (temp_sai, sai_label)]:
                # timebounds, curves = get_lines(temp, label, sai_input, co2_selection, preindustrial_temp)
                timebounds, curves = get_lines(temp, co2_selection, preindustrial_temp)
                fill[label].remove()
                fill[label] = ax.fill_between(
                    timebounds,
                    curves[1],
                    curves[2],
                    color="black",
                    alpha=0.2,
                    linestyle=label2linestyle[label],
                    lw=1,
                    label="90% Confidence Interval" if label == no_sai_label else None
                )
                lines[label].set_xdata(timebounds)
                lines[label].set_ydata(curves[0])
            is_final = year == CANVAS_END_YEAR - 1
            if is_final:
                _global_complete_status = global_complete_status
                if regional_checkbox_value:
                    _global_complete_status += " You chose to generate regional projections but we cannot do so for drawn scenarios."
                yield gr.Markdown(global_plot_title + "\n\n" + _global_complete_status, visible=True), figure, gr.Markdown(visible=False), gr.Image(visible=False)
            else:
                yield gr.Markdown(global_plot_title + "\n\n" + global_running_status, visible=True), figure, gr.Markdown(visible=False), gr.Image(visible=False)
        year += 1

    if regional_checkbox_value and not co2_is_drawn and not sai_is_drawn:
        regional_plot_title = "### End of Century Regional Temperature Projection (Compared to Preindustrial) with and without SAI"
        regional_running_status = "Getting regional temperature projections..."
        regional_complete_status = "✔️ Regional temperature projections complete!\n\nThe plots below show the regional temperature changes at the end of the century (2088-2099) compared to preindustrial (1750-1800) without SAI (top left), with SAI (top right), and the difference between the two (bottom)."

        yield gr.Markdown(global_plot_title + "\n\n" + global_complete_status, visible=True), figure, gr.Markdown(regional_plot_title + "\n\n" + regional_running_status, visible=True), gr.Image(visible=False)

        gr.Markdown("The plots below show the regional temperature changes at the end of the century (2088-2099) compared to preindustrial (1750-1800) without SAI (top left), with SAI (top right), and the difference between the two (bottom).")
        
        simple_sai_scenario = " ".join(sai_selection.split(" ")[:2])

        if uncertainty_checkbox:
            path = DATA_DIR / "regional" / f"regional_{FANCY_SSP_TITLES[co2_selection]}_{simple_sai_scenario}_with_uncertainty.png"
        else:
            path = DATA_DIR / "regional" / f"regional_{FANCY_SSP_TITLES[co2_selection]}_{simple_sai_scenario}.png"

        regional_plot_img = Image.open(path)

        yield gr.Markdown(global_plot_title + "\n\n" + global_complete_status, visible=True), figure, gr.Markdown(regional_plot_title + "\n\n" + regional_complete_status, visible=True), gr.Image(regional_plot_img, visible=True)


# Create an empty plot
def create_empty_plot():
    fig = initialize_temp_plot(fontsize+8)
    return fig


if __name__ == "__main__":

    fontsize = 12
    
    df_emis, df_configs, df_solar, df_volcanic = get_dataframes()

    initial_dir = DATA_DIR / f"initial_{SIM_START_YEAR}_{NUM_MEMBERS}"
    print("Loading data from", initial_dir)
    initial_conditions = {
        path.stem: xr.load_dataarray(path)
        for path in initial_dir.iterdir()
        if path.suffix == ".nc"
    }
    preindustrial_temp = np.load(initial_dir / "preindustrial_temperature.npy")
    print("Loaded initial conditions")

    # Create the Gradio interface
    with gr.Blocks(
                title="Planet Parasol",
                css=css,
                theme=gr.themes.Default(text_size=sizes.text_lg),
                js=js_func
            ) as app:
        set_header()

        # Place the plot below the launch button
        launch_button, regional_checkbox, uncertainty_checkbox = set_launch()

        # Initialize empty status at the top
        global_plot_context = gr.Markdown(visible=False)

        # Initialize empty plot at the top
        combined_plot = gr.Plot(visible=False)

        # Initialize empty regional status at the top
        regional_plot_context = gr.Markdown(visible=False)

        # Initialize empty regional plot at the top
        regional_plot = gr.Image(visible=False)

        gr.Markdown("\n\n")
        gr.Markdown(r"## Change the Emulator Settings")
        with gr.Row():
            with gr.Column():
                gr.Markdown(r"### Set Annual CO$_2$ Emissions 📈", latex_delimiters=[{"left": "$", "right": "$", "display": False}])
                gr.Markdown(r"Climate scientists use several standard scenarios, called Shared Socioeconomic Pathways (SSPs), to explore how CO$_2$ emissions might change in the coming decades.  You can learn more about SSP scenarios [here](https://ourworldindata.org/explorers/ipcc-scenarios?facet=none&country=SSP1+-+1.9~SSP1+-+2.6~SSP2+-+4.5~SSP3+-+Baseline~SSP5+-+Baseline~SSP4+-+6.0~SSP4+-+3.4~SSP5+-+3.4&Metric=Temperature+increase&Rate=Per+capita&Region=Global).", latex_delimiters=[{"left": "$", "right": "$", "display": False}])
                co2_dropdown = gr.Dropdown(
                    label="The emulator defaults to a scenario that assumes steadily increasing emissions through the end of the century.\n\nYou can select a different scenario if you wish.",
                    choices=list(FANCY_SSP_TITLES_SUFFIXES.values()),
                    value=FANCY_SSP_TITLES_SUFFIXES["SSP3-Baseline"]
                )
                co2_value = REVERSE_FANCY_SSP_TITLES[REVERSE_FANCY_SSP_TITLES_SUFFIXES[co2_dropdown.value]]
                co2_ssp_context = gr.Markdown(visible=False)
                co2_sketch_context = gr.Markdown(
                    f"✅ You have chosen to draw your own CO$_2$ emissions scenario. We assume {FANCY_SSP_TITLES[BASE_SCENARIO]} for all other emissions (e.g. CH$_4$).\n\n" + 
                    "**Draw a trajectory to specify the amount of annual CO2 emissions through the end of the century.** Note: values specified outside the bounds of the plot will be clipped and unspecified values will be set to 0.",
                    latex_delimiters=[{"left": "$", "right": "$", "display": False}],
                    visible=False
                )
                co2_plot = gr.Plot(get_base_co2_image(df_emis, co2_value))
                co2_sketch = gr.Sketchpad(
                    value="data/co2/co2_img.png",
                    canvas_size=BASE_CANVAS_SIZE,
                    visible=False,
                    layers=False,
                    transforms=(),
                    brush=Brush(colors=["#2c003e"], color_mode="fixed", default_size=3),
                    image_mode="1"
                )
            with gr.Column():
                gr.Markdown(r"### Set Annual SAI ☁️")
                gr.Markdown(r"The [1991 Mount Pinatubo volcano eruption](https://en.wikipedia.org/wiki/1991_eruption_of_Mount_Pinatubo) is commonly used to study the climate impacts of SAI. We consider SAI scenarios as fractions of the aerosols emitted during that eruption (20 Tg SO$_2$) each year.", latex_delimiters=[{"left": "$", "right": "$", "display": False}])
                sai_dropdown = gr.Dropdown(
                    label="The emulator defaults the amount of SAI to releasing 75% of the aerosols emitted during the Pinatubo eruption.\n\nFeel free to select a different amount of aerosols to release.",
                    choices=[title + f" ({(i+1)*25}% Pinatubo Annually)" for i, title in enumerate(SAI_TITLES)] + [DRAW_SAI_SCENARIO],
                    value=LEVEL3_SAI + f" ({(SAI_TITLES.index(LEVEL3_SAI)+1)*25}% Pinatubo Annually)"
                )
                sai_value = sai_dropdown.value.split(" (")[0]
                sai_select_context = gr.Markdown(visible=False)
                sai_sketch_context = gr.Markdown(
                    f"✅ You have chosen to draw your own SAI scenario.\n\n" + 
                    "**Draw a trajectory to specify the amount of SAI through the end of the century.** Note: values specified outside the bounds of the plot will be clipped and unspecified values will be set to 0.",
                    latex_delimiters=[{"left": "$", "right": "$", "display": False}],
                    visible=False
                )
                sai_plot = gr.Plot(get_base_sai_image(sai_value))
                sai_sketch = gr.Sketchpad(
                    value="data/sai/sai_img.png",
                    canvas_size=BASE_CANVAS_SIZE,
                    visible=False,
                    layers=False,
                    transforms=(),
                    brush=Brush(colors=["#1a5276"], color_mode="fixed", default_size=3),
                    image_mode="1"
                )

        set_footer()

        def handle_co2_change(label):
            if "Draw" in label:
                return gr.Plot(visible=False), gr.Markdown(visible=False), gr.Markdown(visible=True), gr.Sketchpad(visible=True)
            else:
                co2_ssp_md =  gr.Markdown(
                    f"✅ You have selected {label} as the CO$_2$ emissions scenario.",
                    latex_delimiters=[{"left": "$", "right": "$", "display": False}],
                    visible=True
                )
                return gr.Plot(get_base_co2_image(df_emis, REVERSE_FANCY_SSP_TITLES[REVERSE_FANCY_SSP_TITLES_SUFFIXES[label]]), visible=True), co2_ssp_md, gr.Markdown(visible=False), gr.Sketchpad(visible=False)
            
        def handle_sai_change(label):
            if "Draw" in label:
                return gr.Plot(visible=False), gr.Markdown(visible=False), gr.Markdown(visible=True), gr.Sketchpad(visible=True)
            else:
                sai_select_md = gr.Markdown(
                    f"✅ You have selected {label} as the SAI scenario.",
                    latex_delimiters=[{"left": "$", "right": "$", "display": False}],
                    visible=True
                )

                return gr.Plot(get_base_sai_image(label.split(" (")[0]), visible=True), sai_select_md, gr.Markdown(visible=False), gr.Sketchpad(visible=False)

        # Bind updates
        co2_dropdown.change(handle_co2_change, inputs=[co2_dropdown], outputs=[co2_plot, co2_ssp_context, co2_sketch_context, co2_sketch], api_name=False)
        sai_dropdown.change(handle_sai_change, inputs=[sai_dropdown], outputs=[sai_plot, sai_select_context, sai_sketch_context, sai_sketch], api_name=False)
        launch_button\
            .click(visible_component, inputs=[co2_dropdown, sai_dropdown, co2_sketch, sai_sketch], outputs=[global_plot_context, combined_plot, regional_plot_context, regional_plot], api_name=False)\
            .then(update_plot, inputs=[co2_dropdown, sai_dropdown, co2_sketch, sai_sketch, regional_checkbox, uncertainty_checkbox], outputs=[global_plot_context, combined_plot, regional_plot_context, regional_plot], api_name=False)

    # Run the app
    app.queue(default_concurrency_limit=8)
    app.launch(favicon_path="img/logo.svg", server_port=80)

import copy
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.colorbar import ColorbarBase

from planet_parasol_demo.inputs import get_co2_input, get_sai_input
from planet_parasol_demo.constants import *
from planet_parasol_demo.fair_utils import run


plt.rcParams["font.family"] = "serif"


def get_base_co2_image(df_emis, ssp_scenario, figsize=BASE_FIGSIZE, fontsize=BASE_FONTSIZE):
    co2_color = '#2c003e'

    plt.figure(figsize=figsize)
    ax = plt.gca()

    if "Draw" not in ssp_scenario:
        co2_through_2100 = get_co2_input(
            df_emis,
            ssp_scenario,
            start_year=PLOT_START_YEAR,
            end_year=PLOT_END_YEAR
        )
        ax.plot(
            range(PLOT_START_YEAR, 2101),
            co2_through_2100,
            color=co2_color,
            lw=4,
            label=FANCY_SSP_TITLES[ssp_scenario]
        )
        # co2_input = np.column_stack((range(PLOT_START_YEAR, PLOT_END_YEAR+1), co2_through_2100))
        # co2_input = co2_input[co2_input[:, 0] >= CANVAS_START_YEAR]
    else:
        co2_through_2015 = get_co2_input(
            df_emis,
            BASE_SCENARIO, # Scenario doesn't matter since pre 2015
            PLOT_START_YEAR,
            2015
        )
        ax.plot(
            range(PLOT_START_YEAR, CANVAS_START_YEAR),
            np.concatenate([co2_through_2015, np.ones(CANVAS_START_YEAR - 2016) * co2_through_2015[-1] * 1.03 ** np.arange(8)]),
            color=co2_color,
            lw=4
        )
        # co2_input = ssp_scenario

    ax.set_xlim(PLOT_START_YEAR, PLOT_END_YEAR)
    ax.set_yticks(np.arange(0, MAX_CO2, 50))
    ax.set_ylim(-1, MAX_CO2)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.set_ylabel(r"Gt CO$_2$", fontsize=fontsize+2)
    ax.grid(axis='y')
    ax.axvline(2023, color='k', ls=':', lw=2)
    if "Draw" not in ssp_scenario:
        ax.legend(fontsize=fontsize-2, loc='upper left')

    return plt.gcf()


def get_base_sai_image(sai_scenario, figsize=BASE_FIGSIZE, fontsize=BASE_FONTSIZE):
    sai_color = '#1a5276'

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()

    ax.plot(
        range(PLOT_START_YEAR, CANVAS_START_YEAR),
        np.zeros(CANVAS_START_YEAR - PLOT_START_YEAR),
        color=sai_color,
        lw=4
    )
    sai_input = get_sai_input(sai_scenario)
    if "Draw" not in sai_scenario:
        ax.plot(
            sai_input[:, 0],
            sai_input[:, 1],
            color=sai_color,
            lw=4,
            label=sai_scenario
        )

    ax.set_xlim(PLOT_START_YEAR, PLOT_END_YEAR)
    ax.set_yticks(np.arange(0, 40, 10))
    ax.set_ylim(-0.5, MAX_SAI)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.set_ylabel(r"Tg SO$_2$", fontsize=fontsize+2)
    ax.grid(axis='y')
    ax.axvline(2023, color='k', ls=':', lw=2)
    if "Draw" not in sai_scenario:
        ax.legend(fontsize=fontsize-2, loc='upper left')

    return plt.gcf()


def initialize_temp_plot(fontsize):

    height = 6
    fig = plt.figure(figsize=(24, height), dpi=BASE_DPI)

    ax = plt.gca()

    timebounds = list(range(PLOT_START_YEAR, PLOT_END_YEAR + 1))

    # Color parts of the plot within certain y ranges differently
    cmap = ListedColormap(list(TEMP_COLORS.values()))
    alpha = 0.25
    for (ymin, ymax), color in TEMP_COLORS.items():
        ax.fill_between(timebounds, ymin, ymax, facecolor=color, alpha=alpha)

    # Create a colorbar
    colorbar_axes = fig.add_axes([0.68, 0.135, 0.02, 0.82])
    cbar = ColorbarBase(ax=colorbar_axes, cmap=cmap,
                        ticks=[0.125, 0.375, 0.625, 0.875], boundaries=[0, 0.25, 0.5, 0.75, 1])
    cbar.solids.set(alpha=alpha)
    cbar.ax.set_yticklabels(["\n".join(impact) for impact in IMPACTS.values()], fontsize=fontsize-2)

    ax.set_xlim(PLOT_START_YEAR, PLOT_END_YEAR)
    ax.set_yticks(np.arange(-2, 10, 2))
    ax.set_ylim(-0.5, 8)
    ax.axhline(0, color="k", ls=":", lw=0.5)
    ax.axvline(2023, color='k', ls=':', lw=2)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.set_ylabel("Warming Since\nPreindustrial (°C)", fontsize=fontsize+2)

    fig.tight_layout()

    plt.subplots_adjust(right=0.65)

    return fig


def get_lines(temp, ssp_scenario, preindustrial_temp=None):
    if preindustrial_temp is None:
        ### Temperature anomaly
        weights_51yr = np.ones(52)
        weights_51yr[0] = 0.5
        weights_51yr[-1] = 0.5

        preindustrial_temp = np.average(
            temp.loc[
                dict(scenario=ssp_scenario, timebounds=np.arange(1850, 1902), layer=0)
            ],
            weights=weights_51yr,
            axis=0
        )

    timebounds = temp.timebounds
    temp = temp.loc[dict(scenario=ssp_scenario, layer=0)]

    lines = [
        np.median(temp - preindustrial_temp, axis=1),
        np.percentile(temp - preindustrial_temp, 5, axis=1),
        np.percentile(temp - preindustrial_temp, 95, axis=1)
    ]

    return timebounds, lines


plt.ion()
class DynamicUpdate():

    def __init__(self, df_emis, df_configs, df_solar, df_volcanic, co2_input, sai_input,
                 fontsize, ssp_scenario=None, sai_scenario=None):
        initial_dir = DATA_DIR / f"initial_{SIM_START_YEAR}_{NUM_MEMBERS}"
        if IGNORE_DIR or not initial_dir.exists():
            initial_conditions = None
            self.preindustrial_temp = None
        else:
            initial_conditions = {
                path.stem: xr.load_dataarray(path)
                for path in initial_dir.iterdir()
                if path.suffix == ".nc"
            }
            if ssp_scenario is None or "Draw" in ssp_scenario:
                # Replace scenario in the initial conditions with the base scenario
                for key in list(initial_conditions.keys()):
                    initial_conditions[key].coords["scenario"] = [BASE_SCENARIO]
            else:
                # Replace scenario in the initial conditions with the ssp scenario
                for key in list(initial_conditions.keys()):
                    initial_conditions[key].coords["scenario"] = [REVERSE_FANCY_SSP_TITLES[ssp_scenario]]
            self.preindustrial_temp = np.load(initial_dir / "preindustrial_temperature.npy")
        sai_scenario_simple = sai_scenario.split("Injection")[0] + "SAI"

        self.runner = run(
            df_emis, df_configs,
            df_solar, df_volcanic,
            co2_input, sai_input,
            ssp_scenario, initial_conditions=None
        )

        self.no_sai_label = "No SAI"
        if "Draw" not in ssp_scenario:
            self.ssp_scenario = REVERSE_FANCY_SSP_TITLES[ssp_scenario]
        else:
            self.ssp_scenario = BASE_SCENARIO
        self.sai_scenario = sai_scenario
        if  "Draw" in sai_scenario:
            self.no_sai_label = f"No SAI (Your scenario)"
        else:
            self.no_sai_label = f"No SAI ({ssp_scenario})"

        self.sai_label = "SAI"

        if sai_scenario is not None and "Draw" not in sai_scenario:
            self.sai_label = self.sai_label.replace("SAI", sai_scenario_simple)
        elif sai_scenario is not None and "Draw" in sai_scenario:
            self.sai_label = "Your SAI Scenario"

        self.sai_input = sai_input

        self.fontsize = fontsize
        self.label2linestyle = {
            self.no_sai_label: "-",
            self.sai_label: "--"
        }

    def on_launch(self):
        gr.Markdown(r"##### Global Temperature Projection (Compared to Preindustrial) with and without SAI")

        fig = initialize_temp_plot(self.fontsize+8)
        self.figure = copy.deepcopy(fig)
        self.ax = self.figure.axes[0]
        self.lines = {
            label: self.ax.plot([],[], color="black", linestyle=linestyle, label=label)[0]
            for label, linestyle in self.label2linestyle.items()
        }

        self.fill = {
            label: self.ax.fill_between([],[],[], color="black", alpha=0.2, linestyle=linestyle, lw=1,
                                        label="90% Confidence Interval" if label == self.no_sai_label else None)
            for label, linestyle in self.label2linestyle.items()
        }
        self.median_fill = self.ax.fill_between([],[],[], color="red", alpha=0.2, lw=0)

    def on_running(self, temp_no_sai, temp_sai):
        for temp, label in [(temp_no_sai, self.no_sai_label), (temp_sai, self.sai_label)]:
            timebounds, lines = get_lines(temp, label, self.sai_input, self.ssp_scenario, self.preindustrial_temp)
            self.fill[label].remove()
            self.fill[label] = self.ax.fill_between(
                timebounds,
                lines[1],
                lines[2],
                color="black",
                alpha=0.2,
                linestyle=self.label2linestyle[label],
                lw=1,
                label="90% Confidence Interval" if label == self.no_sai_label else None
            )
            self.lines[label].set_xdata(timebounds)
            self.lines[label].set_ydata(lines[0])

        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()

        return self.figure

    def __call__(self):
        self.on_launch()
        year = SIM_START_YEAR
        for temp_no_sai, temp_sai in self.runner:
            if year == PLOT_START_YEAR:
                self.ax.legend(fontsize=self.fontsize+8, loc="upper left")
            if year >= CANVAS_START_YEAR and (year % 15 == 0 or year == CANVAS_END_YEAR-1):
                self.on_running(temp_no_sai, temp_sai)
                yield self.figure
            year += 1

        return temp_no_sai

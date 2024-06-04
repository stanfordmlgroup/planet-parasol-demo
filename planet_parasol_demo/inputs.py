import numpy as np
from scipy.interpolate import interp1d

from fair.structure.units import compound_convert, prefix_convert, time_convert, desired_emissions_units

from planet_parasol_demo.constants import *


def get_co2_between_years(df_emis, ssp_scenario, start_year, end_year, specie, specie_rcmip_name):
    timebounds = np.arange(1750, 2501)
    timepoints = 0.5 * (timebounds[:-1] + timebounds[1:])

    """Taken from FaIR: https://github.com/OMS-NetZero/FAIR/blob/395ab8a4f74d1438fb6075410961942019a9b58f/src/fair/fair.py#L695"""
    emis_in = (
        df_emis.loc[
            (df_emis["Scenario"] == ssp_scenario)
            & (
                df_emis["Variable"].str.endswith(
                    "|" + specie_rcmip_name
                )
            )
            & (df_emis["Region"] == "World"),
            "1750":"2500",
        ]
        .interpolate(axis=1)
        .values.squeeze()
    )
    # avoid NaNs from outside the interpolation range being mixed into
    # the results
    notnan = np.nonzero(~np.isnan(emis_in))

    # RCMIP are "annual averages"; for emissions this is basically
    # the emissions over the year, for concentrations and forcing
    # it would be midyear values. In every case, we can assume
    # midyear values and interpolate to our time grid.
    rcmip_index = np.arange(1750.5, 2501.5)
    interpolator = interp1d(
        rcmip_index[notnan],
        emis_in[notnan],
        fill_value="extrapolate",
        bounds_error=False,
    )
    emis = interpolator(timepoints)

    # We won't throw an error if the time is out of range for RCMIP,
    # but we will fill with NaN to allow a user to manually specify
    # pre- and post- emissions.
    emis[timepoints < 1750] = np.nan
    emis[timepoints > 2501] = np.nan

    # Parse and possibly convert unit in input file to what FaIR wants
    unit = df_emis.loc[
        (df_emis["Scenario"] == ssp_scenario)
        & (df_emis["Variable"].str.endswith("|" + specie_rcmip_name))
        & (df_emis["Region"] == "World"),
        "Unit",
    ].values[0]
    emis = emis * (
        prefix_convert[unit.split()[0]][
            desired_emissions_units[specie].split()[0]
        ]
        * compound_convert[unit.split()[1].split("/")[0]][
            desired_emissions_units[specie].split()[1].split("/")[0]
        ]
        * time_convert[unit.split()[1].split("/")[1]][
            desired_emissions_units[specie].split()[1].split("/")[1]
        ]
    )  # * self.timestep

    return emis[(timepoints > start_year) & (timepoints < end_year+1)]


def get_co2_input(df_emis, ssp_scenario, start_year, end_year):
    if "Draw" not in ssp_scenario:
        # Use CO2 for both specie and RCMIP name
        # Which is the sum of CO2 FFI and CO2 AFOLU
        return get_co2_between_years(
            df_emis,
            ssp_scenario,
            start_year,
            end_year,
            "CO2",
            "CO2"
        )
    else:
        return ssp_scenario
    


def get_sai_input(sai_scenario):
    if "Draw" not in sai_scenario:
        sai_input = np.zeros((PLOT_END_YEAR - 2022, 2))
        sai_input[:, 0] = range(2023, PLOT_END_YEAR + 1)
        if sai_scenario in SAI_TITLES:
            sai_input[:, 1] = SAI_VALUES[SAI_TITLES.index(sai_scenario)]
            # Instead of a constant value, linearly rampup to the value for 10 years
            sai_input[:, 1] *= np.clip((sai_input[:, 0] - 2023) / 10, 0, 1)
        else:
            raise ValueError(f"Invalid SAI scenario: {sai_scenario}")

    else:
        sai_input = sai_scenario

    return sai_input

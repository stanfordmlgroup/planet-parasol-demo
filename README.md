<h1>
  <img src="img/logo.svg" alt="Logo" height="30"> Planet Parasol Demo
</h1>

[![DOI](https://zenodo.org/badge/810113867.svg)](https://zenodo.org/doi/10.5281/zenodo.11467175)

*A tool to explore the effects of stratospheric aerosol injection (SAI) on the climate:  [planetparasol.ai](http://planetparasol.ai)*

## Release
- [2024/06/04] ☁️ We release the first version of the tool at [planetparasol.ai](http://planetparasol.ai)!

## Upcoming
- We plan to release the data, code, and models for the regional emulator that we developed for the tool.

## Background
Stratospheric aerosol injection (SAI) uses reflective aerosols released into the upper atmosphere to reflect sunlight and thereby cool Earth's surface.

Despite the strong potential of SAI to lower Earth's average surface temperature and mitigate global warming's short-term effects:
- SAI cannot “turn back the clock” on climate change: it is not a substitute for reducing greenhouse gas emissions, which is critical to address climate change and achieve long-term sustainability
- There are potential risks that may arise from SAI, so more research is necessary to better understand those risks and compare them to the potential benefits

Our tool allows you to create different CO2 and SAI scenarios and explore their effect on future temperature. Check it out here:  [planetparasol.ai](http://planetparasol.ai).

## Methodology
**Global Emulator**
- The global emulator uses [FaIR](https://docs.fairmodel.net/en/latest/intro.html)
- The sensitivity of the climate emulator is the median of the IPCC assessed range ([source](https://gmd.copernicus.org/articles/11/2273/2018/))
- We assume a forcing efficiency 0.28 W/m2 per Tg SO2/yr ([source1](https://www.pnnl.gov/sites/default/files/media/file/Sensitivity%20of%20Aerosol%20Distribution%20and%20Climate%20Response%20to%20Stratospheric%20SO2%20Injection%20Locations.pdf), [source2](https://www.google.com/url?q=https://acp.copernicus.org/articles/21/10039/2021/&sa=D&source=docs&ust=1715284226685975&usg=AOvVaw38Ib3Gc0XRuSme39sOh_tz))
- We use a 10-year SAI ramp-up

**Regional Emulator**
We use publicly available climate model data to develop the regional emulator. The emulator is equivalent to pattern scaling, and consists of two major components:
1. A linear regression from the [FaIR](https://docs.fairmodel.net/en/latest/intro.html) global mean temperature to the [ScenarioMIP](https://gmd.copernicus.org/articles/9/3461/2016/) regional temperature. We train one linear regression for six different climate models, namely CESM2-WACCM, CNRM-ESM2-1, IPSL-CM6A-LR, MPI-ESM1-2-HR, MPI-ESM1-2-LR, and UKESM1-0-LL.
2. A linear regression from the global mean stratospheric aerosol optical depth (AOD), assuming 0.01 AOD per Tg SO2, to the regional temperature difference. We train one linear regression for CESM2-WACCM using simulation data from [GeoMIP](https://climate.envsci.rutgers.edu/geomip/data.html) and [ARISE-SAI](https://www.cesm.ucar.edu/community-projects/arise-sai).
We try to estimate the substantial uncertainties in regional temperature forecasts using the following methods:
- _Climate model uncertainty_: We compute the standard deviation of the regional values across the 6 climate models.
- _Emulator uncertainty_: We compute the standard deviation of the regional values across 100 bootstrapped linear regression emulators. 
- _Natural variability_: We compute the standard deviation of the regional values across 100 ensemble members from FaIR.

## Contributors
- [Jeremy Irvin](https://twitter.com/jeremy_irvin16), Stanford University
- [Daniele Visioni](https://twitter.com/DanVisioni), Cornell University
- [Ben Kravitz](https://earth.indiana.edu/directory/faculty/kravitz-ben.html), Indiana University
- [Dakota Gruener](https://twitter.com/dakotagruener), Reflective
- [Chris Smith](https://twitter.com/chrisroadmap), University of Leeds
- [Duncan Watson-Parris*](https://twitter.com/DWatsonParris), University of California San Diego
- [Andrew Ng*](https://twitter.com/AndrewYNg), Stanford University
\* Co-supervisors

## Acknowledgements
- [FaIR](https://docs.fairmodel.net/en/latest/intro.html): Our tool is built around the FaIR emulator, a simple climate model to explore the effects of different inputs on the climate.
- [GeoMIP](https://climate.envsci.rutgers.edu/geomip/data.html): We use data from the GeoMIP experiments to develop our regional emulator.
- [ARISE-SAI](https://www.cesm.ucar.edu/community-projects/arise-sai): We use data from the ARISE-SAI experiments to develop our regional emulator.

## Citing the demo

If you use the Planet Parasol Demo in your work, please cite it as follows:

Jeremy Irvin, Daniele Visioni, Ben Kravitz, Dakota Gruener, Chris Smith, Duncan Watson-Parris, Andrew Ng. (2024). Planet Parasol Demo (Version 1.0) [Software]. Available at GitHub: https://github.com/stanfordmlgroup/planet-parasol-demo/ and on the webpage: http://planetparasol.ai/. DOI: https://doi.org/10.5281/zenodo.11467175

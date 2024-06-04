<img src="img/logo.svg" alt="Header Image" width="200" height="200">
# Planet Parasol Demo

*A tool to explore the effects of stratospheric aerosol injection (SAI) on the climate.*

 [http://planetparasol.ai](planetparasol.ai)

## Release
- [2024/06/03] We release the first version of the tool at [http://planetparasol.ai](planetparasol.ai)

## Upcoming
- We plan to release the data, code, and models for the regional emulator used in the tool that we developed.

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

## Citation

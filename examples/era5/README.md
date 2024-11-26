In this example, precipitation from [ERA5](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form) reanalysis data data is downscaled to the resolution of [CombiPrecip](https://www.meteoswiss.admin.ch/services-and-publications/service/weather-and-climate-products/combiprecip.html).

The structure of the example is:

1. Data engineering is done inside `engineering/`.

2. Downscaling using a diffusion model is done inside `diffusion/`.

3. Downscaling using quantile mapping is done inside `quantile_mapping/`.

4. We additionally provide namelist scripts for [WRF](https://www.mmm.ucar.edu/models/wrf) runs in `wrf_namelists/`.

5. Comparisons between the output from each model is done in `benchmarking/`.
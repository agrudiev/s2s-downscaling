In this example, precipitation from weather forecasts from 1 week up to 3 weeks of lead time is downscaled to the resolution of [CombiPrecip](https://www.meteoswiss.admin.ch/services-and-publications/service/weather-and-climate-products/combiprecip.html).

Inside `dirs_template.toml` you will find the paths for general directories used to store data used throughout this example. **The fields `raw` and `main` are left open for new users to fill in with the paths they want to use in their own computer/cluster. After filled in, the file should be renamed as `dirs.toml`.**

When configurations for the scripts are big, they can come in a configuration file inside `configs/` folders.

The structure of the example is:

1. Data engineering is done inside `engineering/`.

2. Downscaling using a diffusion model is done inside `diffusion/`.

3. Comparisons between the output from each model is done in `benchmarking/`.
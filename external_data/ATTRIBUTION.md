# Data attribution and processing

Weather data by [Open-Meteo.com](https://open-meteo.com/), using the [Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api) with ERA5 reanalysis. The upstream API data are provided under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); see [Open-Meteo's data licence](https://open-meteo.com/en/licence) and [service terms](https://open-meteo.com/en/terms). Credit the source, retain licence information and describe modifications when sharing or adapting the weather-derived material. No source endorsement is implied.

Modifications made for this study: select temperature, wind and WMO weather code at the representative city-level point (40.7128, -74.0060); align backward to weather valid times at least 60 minutes before each reconstructed target start; encode weather conditions as indicators; combine with reconstructed weekday/weekend labels; export aligned float32 arrays for 30-, 60- and 90-minute targets. Metadata records source requests, response-location information and preserved response hashes. Training-only standardization is performed later and is not baked into these arrays.

Reconstructed calendars use the published benchmark periods and a fixed 24-hour local wall-clock convention; original timestamp labels remain unrecovered. These CSVs provide derived alignment metadata rather than original trip records. They contain no passenger/trip-level information.

ERA5 is produced through the Copernicus Climate Change Service (C3S)/ECMWF. The source-data licence links are also provided on Open-Meteo's licence page. This distribution does not alter applicable source terms or provide a software licence for the RTAT code.

Suggested Open-Meteo citation: Patrick Zippenfenig (2023), Open-Meteo.com Weather API, https://doi.org/10.5281/zenodo.7970649.

Obtain the NYCTaxi/NYCBike processed flow benchmark from the data release linked by [ProSTformer](https://github.com/yanxiao1930/ProSTformer_code). Original trip providers are [NYC TLC](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) and [Citi Bike](https://citibikenyc.com/system-data).

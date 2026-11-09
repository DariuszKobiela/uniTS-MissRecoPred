# Dataset licenses and attribution

Licenses apply separately to the dataset components. No single license in this
repository overrides another source's terms.

| Component | Source and attribution | License |
|---|---|---|
| Boiler (`boiler_outlet_temp_univ`) | Nikita Manaenkov, [Time-Series of Industrial Boiler Operations](https://www.kaggle.com/datasets/nikitamanaenkov/time-series-of-industrial-boiler-operations) | [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) · [legal code](https://creativecommons.org/licenses/by/4.0/legalcode) |
| Vibration (`vibration_sensor_S1`) | Ziya / Ziya07, [IIoT Edge Computing Dataset](https://www.kaggle.com/datasets/ziya07/iiot-edge-computing-dataset) | [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) · [legal code](https://creativecommons.org/publicdomain/zero/1.0/legalcode) |

The redistributed source signals and their cleaned, split, degraded and
reconstructed versions retain the corresponding source license above.
Changes by Dariusz Kobiela / uniTS-MissRecoPred include signal selection,
cleaning and filling pre-existing gaps, temporal splitting, artificial missingness,
and reconstruction with the methods described in the dataset card. No endorsement
by the original providers is implied. Retain this attribution, source and license
links, and an indication of changes when redistributing Boiler-derived data.

Original evaluation metrics and release documentation contributed by
Dariusz Kobiela are provided under CC BY 4.0. The framework source code is
licensed separately in its GitHub repository; its software license does not
relicense these datasets.

Pump (`sensor_28`) is excluded from this distribution, including its source,
cleaned, split, degraded and reconstructed series and metric records. Its Kaggle
metadata reports an unknown license, so no redistribution permission is assumed.
Obtain the original data from [Pump Sensor Data by nphantawee](https://www.kaggle.com/datasets/nphantawee/pump-sensor-data)
and check permissions with the provider before using or redistributing it.

Source license metadata checked on 2026-11-09. Mixed historical archives were
removed from the current Hugging Face revision. Earlier Git revisions may still
contain withdrawn material; this notice grants no rights to that material.

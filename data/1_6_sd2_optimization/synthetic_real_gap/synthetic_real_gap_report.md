# Synthetic-to-real gap analysis

- Synthetic windows: 2000
- Real windows: 40
- Grouped classifier ROC AUC: 0.9037999999999999
- Grouped classifier balanced accuracy: 0.6890000000000001
- Cross-validation groups windows by source series to limit leakage.
- A classifier score near 0.5 indicates overlap; a score near 1.0 indicates easy separation.

## Feature summary

| feature | synthetic mean | synthetic std | real mean | real std |
| --- | ---: | ---: | ---: | ---: |
| mean | 0.00542645 | 0.568626 | 657.77 | 251 |
| std | 0.6928 | 0.394129 | 35.4582 | 100.554 |
| skewness | -0.027048 | 1.73228 | 0.0344521 | 0.804589 |
| kurtosis | 8.73055 | 28.9782 | 0.180085 | 1.88964 |
| acf_1 | 0.741546 | 0.39649 | 0.761799 | 0.353251 |
| acf_5 | 0.602384 | 0.389669 | 0.693615 | 0.400774 |
| acf_10 | 0.390811 | 0.424816 | 0.639328 | 0.424118 |
| acf_24 | 0.168341 | 0.502523 | 0.536391 | 0.446021 |
| trend_slope | -2.0491e-05 | 0.00245668 | 0.071504 | 0.446655 |
| trend_strength | 0.160494 | 0.29771 | 0.281208 | 0.322793 |
| seasonality_strength | 0.310378 | 0.274561 | 0.351142 | 0.262525 |
| spectral_entropy | 0.535433 | 0.283312 | 0.494429 | 0.297948 |
| dominant_frequency | 0.0662148 | 0.118093 | 0.0361816 | 0.0864289 |
| length | 512 | 0 | 512 | 0 |
| diff_mean | -1.12166e-05 | 0.00283738 | 0.0291296 | 0.304288 |
| diff_std | 0.219295 | 0.171428 | 12.9957 | 35.1729 |
| diff_abs_mean | 0.146596 | 0.118004 | 6.47624 | 10.4806 |
| diff_abs_p95 | 0.353273 | 0.337975 | 26.9512 | 84.2457 |
| turning_point_rate | 0.492138 | 0.19683 | 0.355931 | 0.255318 |

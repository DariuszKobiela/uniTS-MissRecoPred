import numpy as np
import pandas as pd
import pytest

from reconstruction_models.stable_diffusion_2_gaf import (
    gaf_to_series,
    series_to_gaf,
)
from reconstruction_models.stable_diffusion_2_mtf import (
    mtf_to_series,
    series_to_mtf,
)
from reconstruction_models.stable_diffusion_2_rp import (
    rp_to_series,
    series_to_rp,
)
from reconstruction_models.stable_diffusion_2_spec import (
    series_to_spectrogram,
    spectrogram_to_series,
)


@pytest.fixture
def reference_series() -> pd.Series:
    x = np.linspace(0.0, 8.0 * np.pi, 512)
    values = np.sin(x) + 0.3 * np.linspace(0.0, 1.0, 512)
    return pd.Series(values, index=pd.RangeIndex(1000, 1512))


def _rmse(actual: pd.Series, reconstructed: pd.Series) -> float:
    return float(np.sqrt(np.mean((actual.to_numpy() - reconstructed.to_numpy()) ** 2)))


def test_gaf_round_trip_uses_true_gasf_inverse(reference_series: pd.Series):
    image = series_to_gaf(reference_series)
    reconstructed = gaf_to_series(image, len(reference_series), reference_series)

    assert reconstructed.index.equals(reference_series.index)
    assert _rmse(reference_series, reconstructed) < 2e-5


def test_continuous_rp_round_trip_recovers_one_dimensional_geometry(
    reference_series: pd.Series,
):
    image = series_to_rp(reference_series)
    reconstructed = rp_to_series(image, len(reference_series), reference_series)

    assert reconstructed.index.equals(reference_series.index)
    assert _rmse(reference_series, reconstructed) < 1e-5


def test_mtf_is_a_transition_field_and_has_bounded_quantization_error(
    reference_series: pd.Series,
):
    image, metadata = series_to_mtf(reference_series, return_metadata=True)
    reconstructed = mtf_to_series(
        image,
        len(reference_series),
        reference_series,
        metadata=metadata,
    )

    # MTF is not bijective: decoding returns quantile-state representatives.
    assert len(np.unique(image)) > 2
    assert np.corrcoef(reference_series, reconstructed)[0, 1] > 0.98
    assert _rmse(reference_series, reconstructed) < 0.12


def test_spectrogram_round_trip_uses_retained_stft_phase(
    reference_series: pd.Series,
):
    image, metadata = series_to_spectrogram(reference_series, return_metadata=True)
    reconstructed = spectrogram_to_series(
        image,
        len(reference_series),
        reference_series,
        metadata=metadata,
    )

    assert reconstructed.index.equals(reference_series.index)
    assert np.corrcoef(reference_series, reconstructed)[0, 1] > 0.999
    assert _rmse(reference_series, reconstructed) < 0.01


def test_lossy_decoders_require_encoding_metadata(reference_series: pd.Series):
    with pytest.raises(ValueError, match="MTF decoding requires metadata"):
        mtf_to_series(np.zeros((8, 8)), len(reference_series), reference_series)

    with pytest.raises(ValueError, match="Spectrogram decoding requires phase"):
        spectrogram_to_series(np.zeros((8, 8)), len(reference_series), reference_series)

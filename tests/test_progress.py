from utils.progress import BAR_FORMAT, tqdm


def test_tqdm_wrapper_preserves_values_and_includes_eta_in_format():
    assert list(tqdm(range(3), desc="unit", disable=True)) == [0, 1, 2]
    assert "{l_bar}" in BAR_FORMAT
    assert "{remaining}" in BAR_FORMAT
    assert "{elapsed}" in BAR_FORMAT

# License: BSD 3 clause
from io import BytesIO
import os

import numpy as np
import pytest

import tick.dataset.download_helper as download_helper


class _DummyResponse:
    def __init__(self, payload: bytes):
        self._buffer = BytesIO(payload)
        self.length = len(payload)

    def read(self, n_bytes):
        return self._buffer.read(n_bytes)

    def close(self):
        self._buffer.close()


@pytest.mark.parametrize("dataset_path", ["toy/sample.npz", "toy/sub/array.npz"])
def test_download_tick_dataset_returns_cache_path(tmp_path, monkeypatch, dataset_path):
    payload_buffer = BytesIO()
    expected = np.arange(5)
    np.savez(payload_buffer, arr=expected)
    payload = payload_buffer.getvalue()

    opened_urls = []

    def fake_open(url):
        opened_urls.append(url)
        return _DummyResponse(payload)

    monkeypatch.setattr(download_helper, "urlopen", fake_open)

    data_home = tmp_path / "data"
    cache_path = download_helper.download_tick_dataset(
        dataset_path, data_home=str(data_home), verbose=False
    )

    assert cache_path == os.path.join(str(data_home), dataset_path)
    assert os.path.exists(cache_path)

    loaded = download_helper.load_dataset(dataset_path, data_home=str(data_home))
    np.testing.assert_array_equal(loaded, expected)

    assert opened_urls == [download_helper.BASE_URL % dataset_path]

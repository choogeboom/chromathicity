import numpy as np
from pytest import raises

from chromathicity.rgbspec import RgbSpecification, Srgb


class TestRgbSpecification:

    def test_abstract(self):
        with raises(TypeError):
            RgbSpecification()

    def test_srgb_linear_transformation(self):
        srgb = Srgb()
        actual_linear_transformation = np.array([[41.2418, 21.2653, 1.93321],
                                                 [35.7579, 71.5159, 11.9193],
                                                 [18.0469, 7.21877, 95.0471]])
        np.testing.assert_allclose(srgb.linear_transformation,
                                   actual_linear_transformation,
                                   rtol=1e-5, atol=1e-14)
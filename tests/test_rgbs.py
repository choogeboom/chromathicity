import numpy as np
from pytest import raises

from chromathicity.rgbspec import RgbSpecification, Srgb


class TestRgbSpecification:

    def test_abstract(self):
        with raises(TypeError):
            RgbSpecification()

    def test_srgb_linear_transformation(self):
        srgb = Srgb()
        actual_linear_transformation = np.array([[.412391, .212639, .019331],
                                                 [.3576, .715201, .1192],
                                                 [.1804, .07216, .950108]])
        np.testing.assert_allclose(srgb.linear_transformation,
                                   actual_linear_transformation,
                                   rtol=1e-5, atol=1e-14)
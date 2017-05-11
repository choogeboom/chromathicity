import numpy as np
from pytest import raises

from chromathicity.illuminant import Illuminant, D, A
from chromathicity.observer import Standard


class TestIlluminant:

    def test_abstract_illuminant(self):
        with raises(TypeError):
            Illuminant()

    def test_d(self):
        ill = D('D_65')
        assert ill.temperature == 6504.
        assert ill.name == 'D_65'
        assert repr(ill) == 'D(6504)'
        obs = Standard(2)
        white_point = np.array([.9504669658, 1.000, 1.088996386])
        np.testing.assert_allclose(ill.get_white_point(obs), white_point,
                                   rtol=1e-6, atol=1e-14)
        ill.temperature = 5400.
        assert ill.name == 'D_5400K'
        white_point = np.array([.958051305, 1.000, .902306958])
        np.testing.assert_allclose(ill.get_white_point(obs), white_point,
                                   rtol=1e-6, atol=1e-14)
        with raises(ValueError):
            ill.temperature = 3999
        with raises(ValueError):
            ill.temperature = 25001

    def test_a(self):
        ill = A()
        assert ill.temperature == 2848.
        assert ill.name == 'A'
        assert repr(ill) == 'A(2848.0)'
        obs = Standard(2)
        expected_white_point = np.array([1.0985033, 1.000, .35584922])
        actual_white_point = ill.get_white_point(obs)
        np.testing.assert_allclose(actual_white_point, expected_white_point,
                                   rtol=1e-6, atol=1e-5)

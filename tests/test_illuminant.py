import numpy as np
from pytest import raises

from chromathicity.illuminant import Illuminant, D, A, CustomIlluminant
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
        white_point = np.array([.950392, 1.000, 1.088639])
        np.testing.assert_allclose(ill.get_white_point(obs), white_point,
                                   rtol=1e-6, atol=1e-14)
        ill = D(5400.)
        assert ill.name == 'D_5400K'
        white_point = np.array([.957976, 1.000, .902011])
        np.testing.assert_allclose(ill.get_white_point(obs), white_point,
                                   rtol=1e-6, atol=1e-14)
        with raises(ValueError):
            D(3999)
        with raises(ValueError):
            D(25001)
        with raises(ValueError):
            D('D_47')
        with raises(AttributeError):
            ill.name = 'pizza'
        assert np.all(ill.wavelengths == np.arange(300, 835, 5))
        with raises(AttributeError):
            ill.wavelengths = 2
        with raises(AttributeError):
            ill.psd = 54

    def test_a(self):
        np.set_printoptions(precision=10)
        ill = A()
        assert ill.temperature == 2848.
        assert ill.name == 'A'
        assert repr(ill) == 'A(2848.0)'
        obs = Standard(2)
        expected_white_point = np.array([1.0984166, 1.000, .35573253])
        actual_white_point = ill.get_white_point(obs)
        print(actual_white_point)
        np.testing.assert_allclose(actual_white_point, expected_white_point,
                                   rtol=1e-6, atol=1e-10)

    def test_custom(self):
        ill = CustomIlluminant(name='test',
                               wavelengths=[250, 550, 850],
                               psd=[1., 1., 1.])
        assert ill.name == 'test'
        np.testing.assert_allclose(ill.get_white_point(), [1., 1., 1.],
                                   rtol=1e-6, atol=1e-10)


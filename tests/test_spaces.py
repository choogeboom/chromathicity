from pytest import raises
import numpy as np

import chromathicity.spaces as sp
from chromathicity.error import UndefinedColorSpaceError


def test_get_space():
    assert sp.get_space('xyz') == ('xyz', sp.XyzData)
    assert sp.get_space('XyzData') == ('xyz', sp.XyzData)
    assert sp.get_space(sp.XyzData) == ('xyz', sp.XyzData)
    with raises(UndefinedColorSpaceError):
        sp.get_space('peanut')
    with raises(UndefinedColorSpaceError):
        sp.get_space(sp.WhitePointSensitive)


def test_spectral_data():
    spectrum = sp.SpectralData([[1, 1, .75, 1],
                                [.5, .5, .65, .5],
                                [.25, .25, .55, .25]],
                               [350., 450., 550., 650.])
    assert spectrum.num_components == 4
    np.testing.assert_allclose(spectrum.components[2],
                               np.array([[.75], [.65], [.55]]))





from pytest import raises
import numpy as np

import chromathicity.manage
import chromathicity.spaces as sp
from chromathicity.error import UndefinedColorSpaceError


def test_get_space():
    assert chromathicity.manage.get_space('CIEXYZ') == ('CIEXYZ', sp.XyzData)
    assert chromathicity.manage.get_space(sp.XyzData) == ('CIEXYZ', sp.XyzData)
    with raises(UndefinedColorSpaceError):
        chromathicity.manage.get_space('peanut')
    with raises(UndefinedColorSpaceError):
        chromathicity.manage.get_space(2)
    with raises(TypeError):
        chromathicity.manage.get_space_name(2)


def test_spectral_data():
    spectrum = sp.SpectralData([[1., 1., .75, 1.],
                                [.5, .5, .65, .5],
                                [.25, .25, .55, .25]],
                               [350., 450., 550., 650.])
    assert spectrum.num_components == 4
    np.testing.assert_allclose(spectrum.components[2],
                               np.array([[.75], [.65], [.55]]))


def test_xyz_data():
    xyz = sp.XyzData([22.53731, 18.418652, 9.526464], is_scaled=True)
    lab = xyz.to('CIELAB')
    np.testing.assert_allclose(lab.data, np.array([50., 25., 25.]))

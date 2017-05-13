import numpy as np
from pytest import raises

import chromathicity.convert as convert
from chromathicity.error import UndefinedConversionError, \
    UndefinedColorSpaceError
from chromathicity.illuminant import D
from chromathicity.observer import Standard
from chromathicity.rgbspec import Srgb


class TestHsl:

    def test_rgb2hsl_1d(self):
        run_forward_reverse(convert.rgb2hsl,
                            convert.hsl2rgb,
                            np.array([1., .5, .25]),
                            np.array([20.0, 1.0, 0.6250]))

    def test_rgb2hsl_2d(self):
        run_forward_reverse(convert.rgb2hsl,
                            convert.hsl2rgb,
                            np.array([[1., .5, .25], [.25, 1., .5]]),
                            np.array([[20., 1., .625], [140., 1., 0.6250]]))

    def test_rgb2hsl_2d_rows(self):
        run_forward_reverse(convert.rgb2hsl,
                            convert.hsl2rgb,
                            np.array([[1., .25], [.5, 1.], [.25, .5]]),
                            np.array([[20., 140.], [1., 1.], [.625, .625]]))


class TestHsi:
    def test_rgb2hsi_1d(self):
        run_forward_reverse(convert.rgb2hsi,
                            convert.hsi2rgb,
                            np.array([1., .5, .25]),
                            np.array([20., .57142855, .58333333]))

    def test_rgb2hsi_2d(self):
        run_forward_reverse(convert.rgb2hsi,
                            convert.hsi2rgb,
                            np.array([[1., .5, .25],
                                      [.25, .5, 1.]]),
                            np.array([[20., .57142855, .58333333],
                                      [220., .57142857, .583333333]]))


class TestSpectra:

    def test_spectrum2xyz_axes1(self):
        spectra = np.array([[0, 1, 0], [1, 1, 0]])
        wavelengths = np.array([450, 550, 650])
        ill = D()
        obs = Standard()
        expected_xyz = np.array([[.4098627, .9408818, .008271787],
                                 [.588645, .961091, .950396]])
        actual_xyz = convert.spectrum2xyz(spectra, wavelengths, illuminant=ill,
                                          observer=obs)
        np.testing.assert_allclose(actual_xyz, expected_xyz,
                                   rtol=1e-5, atol=1e-14)

    def test_spectrum2xyz_axes0(self):
        spectra = np.array([[0, 1], [1, 1], [0, 0]])
        wavelengths = np.array([450, 550, 650])
        ill = D()
        obs = Standard()
        expected_xyz = np.array([[.4098627, .588645],
                                 [.9408818, .961091],
                                 [.008271787, .950396]])
        actual_xyz = convert.spectrum2xyz(spectra, wavelengths, illuminant=ill,
                                          observer=obs)
        np.testing.assert_allclose(actual_xyz, expected_xyz,
                                   rtol=1e-5, atol=1e-14)

    def test_spectrum2xyz_1d(self):
        spectra = np.array([1, 1, 0])
        wavelengths = np.array([450, 550, 650])
        ill = D()
        obs = Standard()
        expected_xyz = np.array([.588645, .961091, .950396])
        actual_xyz = convert.spectrum2xyz(spectra, wavelengths, illuminant=ill,
                                          observer=obs)
        np.testing.assert_allclose(actual_xyz, expected_xyz,
                                   rtol=1e-5, atol=1e-14)
        assert actual_xyz.shape == (3,)


class TestXyy:

    def test_xyy2xyz_1d(self):
        run_forward_reverse(convert.xyy2xyz,
                            convert.xyz2xyy,
                            np.array([0.25, 0.5, 50.0]),
                            np.array([25.0, 50.0, 25.0]))

    def test_xyy2xyz_axis_1(self):
        run_forward_reverse(convert.xyy2xyz,
                            convert.xyz2xyy,
                            np.array([[[0.25, 0.4],
                                       [0.5, 0.4],
                                       [50, 60]],
                                      [[0.3, 0.25],
                                       [0.4, 0.25],
                                       [60, 50]]]),
                            np.array([[[25.0, 60.0],
                                       [50.0, 60.0],
                                       [25.0, 30.0]],
                                      [[45.0, 50.0],
                                       [60.0, 50.0],
                                       [45.0, 100.0]]]))

    def test_xyy2xyz_with_0(self):
        xyy = np.array([0.25, 0.0, 0.0])
        expected_xyz = np.array([0.0, 0.0, 0.0])
        actual_xyz = convert.xyy2xyz(xyy)
        np.testing.assert_allclose(actual_xyz, expected_xyz,
                                   rtol=1e-5, atol=1e-14)

    def test_xyz2xyy_with0(self):
        run_forward_reverse(convert.xyz2xyy,
                            convert.xyy2xyz,
                            np.array([0., 0., 0.]),
                            np.array([.312729, .329052, 0.]))


class TestXyz:

    def test_xyz2xyzr(self):
        ill = D()
        obs = Standard()
        xyz = 0.5 * ill.get_white_point(obs)
        expected_xyzr = 0.5 * np.ones((3,))
        actual_xyzr = convert.xyz2xyzr(xyz, illuminant=ill, observer=obs)
        np.testing.assert_allclose(actual_xyzr, expected_xyzr, rtol=1e-5,
                                   atol=1e-14)

    def test_xyz2lab_big(self):
        ill = D()
        obs = Standard()
        xyz = np.array([.2253909, .1841865, .09529589])
        expected_lab = np.array([49.999998, 25.008156, 24.990288])
        actual_lab = convert.convert(xyz, 'xyz', 'lab',
                                     illuminant=ill, observer=obs)
        np.testing.assert_allclose(actual_lab, expected_lab, rtol=1e-5,
                                   atol=1e-14)
        actual_xyz = convert.convert(actual_lab, 'lab', 'xyz',
                                     illuminant=ill, observer=obs)
        np.testing.assert_allclose(actual_xyz, xyz, rtol=1e-5, atol=1e-14)

    def test_xyz2lab_small(self):
        ill = D()
        obs = Standard()
        xyz = np.array([0.008, 0.009, 0.007])
        expected_lab = np.array([8.1289723, -2.264535, 4.001228])
        actual_lab = convert.convert(xyz, 'xyz', 'lab',
                                     illuminant=ill, observer=obs)
        np.testing.assert_allclose(actual_lab, expected_lab, rtol=1e-5,
                                   atol=1e-14)
        actual_xyz = convert.convert(actual_lab, 'lab', 'xyz',
                                     illuminant=ill, observer=obs)
        np.testing.assert_allclose(actual_xyz, xyz, rtol=1e-5, atol=1e-14)


class TestHsv:
    def test_rgb2hsv_1d(self):
        run_forward_reverse(convert.rgb2hsv,
                            convert.hsv2rgb,
                            np.array([1., .5, .25]),
                            np.array([20., .75, 1.])
                            )

    def test_rgb2hsv_2d(self):
        run_forward_reverse(convert.rgb2hsv,
                            convert.hsv2rgb,
                            np.array([[1., .5, .25],
                                      [.25, .5, 1.]]),
                            np.array([[20., .75, 1.],
                                      [220., .75, 1.]]))


class TestHcy:
    def test_rgb2hcy_1d(self):
        run_forward_reverse(convert.rgb2hcy,
                            convert.hcy2rgb,
                            np.array([1., .5, .25]),
                            np.array([20., .75, 0.621])
                            )

    def test_rgb2hcy_2d(self):
        run_forward_reverse(convert.rgb2hcy,
                            convert.hcy2rgb,
                            np.array([[1., .5],
                                      [.5, .25],
                                      [.25, 1.]]),
                            np.array([[20., 260.],
                                      [.75, .75],
                                      [.621, .41025]]))


class TestLCHab:
    def test_lab2lchab_1d(self):
        run_forward_reverse(convert.lab2lchab,
                            convert.lchab2lab,
                            np.array([50., 25., 25.]),
                            np.array([50., np.sqrt(2 * 25. ** 2), 45.]))

    def test_lab2lchab_2d(self):
        run_forward_reverse(convert.lab2lchab,
                            convert.lchab2lab,
                            np.array([[50., 25., 25.], [30., 0., 45.]]),
                            np.array([[50., np.sqrt(2 * 25. ** 2), 45.],
                                      [30., 45., 90.]]))


class TestLRgb:
    def test_lrgb2rgb_1d(self):
        rgbs = Srgb()
        run_forward_reverse(convert.lrgb2rgb,
                            convert.rgb2lrgb,
                            np.array([0.25, .5, .75]),
                            np.array([.537099, .735357, .880825]),
                            rgbs=rgbs)
        run_forward_reverse(convert.lrgb2rgb,
                            convert.rgb2lrgb,
                            np.array([0.25, .5, .75]),
                            np.array([.537099, .735357, .880825]))

    def test_lrgb2xyz_1d(self):
        run_forward_reverse(convert.lrgb2xyz,
                            convert.xyz2lrgb,
                            np.array([.5, .75, 0.]),
                            np.array([.474396, .64272, .099065]))

    def test_lrgb2xyz_2d(self):
        run_forward_reverse(convert.lrgb2xyz,
                            convert.xyz2lrgb,
                            np.array([[.5, 0.],
                                      [.75, .75],
                                      [0., .5]]),
                            np.array([[.474396, .3584],
                                      [.64272, .572481],
                                      [.099065, .564454]]))

    def test_lrgb2xyz_caa(self):
        run_forward_reverse(convert.lrgb2xyz,
                            convert.xyz2lrgb,
                            np.array([.5, .75, 0.]),
                            np.array([.506831, .64893, .079794]),
                            illuminant=D('D_50'))


class TestConvert:
    def test_illegal_conversion(self):
        with raises(UndefinedConversionError):
            convert.convert(np.array([1., 1., 1.]), 'xyz', 'spectrum')

    def test_undefined_space(self):
        with raises(UndefinedColorSpaceError):
            convert.convert(np.ones((3,)), 'xyz', 'pizza')


def run_forward_reverse(convert_forward, convert_reverse, source, destination,
                        **kwargs):
    actual_destination = convert_forward(source, **kwargs)
    assert actual_destination.shape == destination.shape
    np.testing.assert_allclose(actual_destination, destination,
                               rtol=1e-5, atol=1e-14)
    if convert_reverse is not None:
        actual_source = convert_reverse(actual_destination, **kwargs)
        assert actual_source.shape == source.shape
        np.testing.assert_allclose(actual_source, source,
                                   rtol=1e-5,
                                   atol=1e-14)

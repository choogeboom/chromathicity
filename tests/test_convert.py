import numpy as np

import chromathicity.convert as convert
from chromathicity.illuminant import D
from chromathicity.observer import Standard


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
        expected_xyz = np.array([[40.9895, 94.0882, 0.82745],
                                 [58.8691, 96.1091, 95.0708]])
        actual_xyz = convert.spectrum2xyz(spectra, wavelengths, illuminant=ill,
                                          observer=obs)
        np.testing.assert_allclose(actual_xyz, expected_xyz,
                                   rtol=1e-5, atol=1e-14)

    def test_spectrum2xyz_axes0(self):
        spectra = np.array([[0, 1], [1, 1], [0, 0]])
        wavelengths = np.array([450, 550, 650])
        ill = D()
        obs = Standard()
        expected_xyz = np.array([[40.9895, 58.8691],
                                 [94.0882, 96.1091],
                                 [0.82745, 95.0708]])
        actual_xyz = convert.spectrum2xyz(spectra, wavelengths, illuminant=ill,
                                          observer=obs)
        np.testing.assert_allclose(actual_xyz, expected_xyz,
                                   rtol=1e-5, atol=1e-14)

    def test_spectrum2xyz_1d(self):
        spectra = np.array([1, 1, 0])
        wavelengths = np.array([450, 550, 650])
        ill = D()
        obs = Standard()
        expected_xyz = np.array([58.8691, 96.1091, 95.0708])
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
        xyz = np.array([22.53909, 18.41865, 9.529589])
        expected_lab = np.array([50.0, 25.0, 25.0])
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
        xyz = np.array([0.8, 0.9, 0.7])
        expected_lab = np.array([8.1289723, -2.267124, 4.004512])
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


def run_forward_reverse(convert_forward, convert_reverse, source, destination,
                        **kwargs):
    actual_destination = convert_forward(source, **kwargs)
    assert actual_destination.shape == destination.shape
    np.testing.assert_allclose(actual_destination, destination)
    if convert_reverse is not None:
        actual_source = convert_reverse(destination, **kwargs)
        assert actual_source.shape == source.shape
        np.testing.assert_allclose(actual_source, source)
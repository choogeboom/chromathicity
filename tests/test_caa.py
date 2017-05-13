import numpy as np
from pytest import raises

from chromathicity import convert
from chromathicity.chromadapt import ChromaticAdaptationAlgorithm, Bradford
from chromathicity.illuminant import D, A
from chromathicity.observer import Standard


class TestChromaticAdaptationAlgorithm:

    def test_abstract(self):
        with raises(TypeError):
            ChromaticAdaptationAlgorithm()

    def test_linear_transformation(self):
        obs = Standard()
        illuminant_d = D()
        illuminant_a = A()
        white_point_d = illuminant_d.get_white_point(obs)
        white_point_a = illuminant_a.get_white_point(obs)
        caa = Bradford()
        expected_crd = np.array(
            [
                [0.8951, -0.7502, 0.0389],
                [0.2664, 1.7135, -0.0685],
                [-0.1614, 0.0367, 1.0296]
            ])
        actual_crd = caa.cone_response_domain
        np.testing.assert_allclose(actual_crd, expected_crd, rtol=1e-5,
                                   atol=1e-14)
        expected_transform = np.array(
            [
                [1.2164509, 0.15332493, -0.023949391],
                [.11098616, 0.91524230, 0.035903103],
                [-.1549409, -0.055997489, 0.31469628]
            ])
        actual_transform = caa.get_linear_transformation(white_point_d,
                                                         white_point_a)
        np.testing.assert_allclose(actual_transform, expected_transform,
                                   rtol=1e-5, atol=1e-14)

    def test_xyz2xyz(self):
        obs = Standard()
        illuminant_d = D()
        illuminant_a = A()
        white_point_d = illuminant_d.get_white_point(obs)
        white_point_a = illuminant_a.get_white_point(obs)
        caa = Bradford()
        xyz_d = np.array([.25, .750, .350])
        expected_xyz_a = np.array([.33312305, .7051638, .13108368])
        actual_xyz_a = convert.xyz2xyz(xyz_d, white_point_d, white_point_a,
                                       caa=caa)
        np.testing.assert_allclose(actual_xyz_a, expected_xyz_a,
                                   rtol=1e-5, atol=1e-14)

    def test_xyz2xyz_axis(self):
        obs = Standard()
        illuminant_d = D()
        illuminant_a = A()
        white_point_d = illuminant_d.get_white_point(obs)
        white_point_a = illuminant_a.get_white_point(obs)
        caa = Bradford()
        xyz_d = np.array([[[.25, .15],
                           [.75, .55],
                           [.35, .20]],
                          [[.35, .30],
                           [.45, .40],
                           [.15, .50]]])
        expected_xyz_a = np.array([[[.333123, .212521],
                                    [.705164, .515183],
                                    [.131084, .0790936]],
                                   [[.452460, .331859],
                                    [.457123, .384096],
                                    [.0549786, .164526]]])
        actual_xyz_a = convert.xyz2xyz(xyz_d, white_point_d, white_point_a,
                                       axis=1, caa=caa)
        np.testing.assert_allclose(actual_xyz_a, expected_xyz_a,
                                   rtol=1e-5, atol=1e-14)
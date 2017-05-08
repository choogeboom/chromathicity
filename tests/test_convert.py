import numpy as np

import chromathicity.convert as convert


def test_rgb2hsl():
    run_forward_reverse(convert.rgb2hsl,
                        convert.hsl2rgb,
                        np.array([1., .5, .25]),
                        np.array([20.0, 1.0, 0.6250]))
    run_forward_reverse(convert.rgb2hsl,
                        convert.hsl2rgb,
                        np.array([[1., .5, .25], [.25, 1., .5]]),
                        np.array([[20., 1., .625], [140., 1., 0.6250]]))
    run_forward_reverse(convert.rgb2hsl,
                        convert.hsl2rgb,
                        np.array([[1., .25], [.5, 1.], [.25, .5]]),
                        np.array([[20., 140.], [1., 1.], [.625, .625]]))


def test_rgb2hsi():
    run_forward_reverse(convert.rgb2hsi,
                        convert.hsi2rgb,
                        np.array([1., .5, .25]),
                        np.array([20., .57142855, .58333333]))
    run_forward_reverse(convert.rgb2hsi,
                        convert.hsi2rgb,
                        np.array([[1., .5, .25],
                                  [.25, .5, 1.]]),
                        np.array([[20., .57142855, .58333333],
                                  [220., .57142857, .583333333]]))


def run_forward_reverse(convert_forward, convert_reverse, source, destination, **kwargs):
    actual_destination = convert_forward(source, **kwargs)
    assert actual_destination.shape == destination.shape
    np.testing.assert_allclose(actual_destination, destination)
    if convert_reverse is not None:
        actual_source = convert_reverse(destination, **kwargs)
        assert actual_source.shape == source.shape
        np.testing.assert_allclose(actual_source, source)
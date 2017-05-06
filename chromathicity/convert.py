from typing import Tuple

import numpy as np
import scipy.integrate as integrate
from chromathicity.chromadapt import (
    ChromaticAdaptationAlgorithm,
    get_default_chromatic_adaptation_algorithm)
from chromathicity.illuminant import Illuminant, get_default_illuminant
from chromathicity.rgbspec import (RgbSpecification,
                                   get_default_rgb_specification)
from chromathicity.manage import color_conversion, get_conversion_path

from chromathicity.observer import Observer, get_default_observer

# Constants used for L*a*b* conversions
LAB_EPS = 0.008856
LAB_KAPPA = 903.3


# noinspection PyUnusedLocal
@color_conversion('lab', 'xyzr')
def lab2xyzr(lab: np.ndarray, axis: int=None, **kwargs) -> np.ndarray:
    """
    Convert LAB to normalized XYZ
    
    :param lab: 
    :param axis: 
    :return: 
    """
    if axis is None:
        axis = get_matching_axis(lab.shape, 3)

    inds = _construct_component_inds(axis, len(lab.shape), 3)
    fxyz = np.zeros(lab.shape)
    fxyz[inds[1]] = (lab[inds[0]] + 16) / 116
    fxyz[inds[0]] = lab[inds[1]]/500 + fxyz[inds[1]]
    fxyz[inds[2]] = fxyz[inds[1]] - lab[inds[2]]/200

    is_small = fxyz <= (LAB_EPS ** (1.0/3.0))
    is_big = np.logical_not(is_small)
    xyzr = np.zeros(lab.shape)
    xyzr[is_big] = fxyz[is_big] ** 3.0
    xyzr[is_small] = (116*fxyz[is_small] - 16) / LAB_KAPPA
    return xyzr


# noinspection PyUnusedLocal
@color_conversion('lrgb', 'rgb')
def lrgb2rgb(lrgb: np.ndarray,
             rgbs: RgbSpecification=None, **kwargs) -> np.ndarray:
    """
    Convert linear RGB to companded RGB
    
    :param lrgb: 
    :param rgbs: 
    :return: 
    """
    if rgbs is None:
        rgbs = get_default_rgb_specification()
    return rgbs.compand(lrgb)


@color_conversion('lrgb', 'xyz')
def lrgb2xyz(lrgb: np.ndarray,
             axis: int=None,
             illuminant: Illuminant=None,
             observer: Observer=None,
             rgbs: RgbSpecification=None,
             caa: ChromaticAdaptationAlgorithm=None) -> np.ndarray:
    """
    Convert from linear RGB to XYZ
    
    :param lrgb: 
    :param axis: 
    :param illuminant: 
    :param observer: 
    :param rgbs: 
    :param caa: 
    :return: 
    """
    if axis is None:
        axis = get_matching_axis(lrgb.shape, 3)
    if illuminant is None:
        illuminant = get_default_illuminant()
    if observer is None:
        observer = get_default_observer()
    if rgbs is None:
        rgbs = get_default_rgb_specification()
    if caa is None:
        caa = get_default_chromatic_adaptation_algorithm()

    # Get the XYZ values into the correct shape for matrix multiplication
    n_dims = lrgb.ndim
    new_dims = list(range(n_dims))
    if axis != n_dims - 1:
        new_dims[-1] = axis
        new_dims[axis] = n_dims - 1
    lrgb = lrgb.transpose(new_dims)

    input_shape = lrgb.shape
    lrgb_is_not_matrix = n_dims != 2
    if lrgb_is_not_matrix:
        lrgb = lrgb.reshape((-1, 3))

    # Do the transformation
    m = rgbs.linear_transformation
    xyz = lrgb.dot(m)

    # Transform back to the original shape
    if lrgb_is_not_matrix:
        xyz = xyz.reshape(input_shape)

    source_white_point = rgbs.white_point
    destination_white_point = illuminant.get_white_point(observer)
    if not np.allclose(source_white_point, destination_white_point,
                       rtol=1e-5, atol=1e-14):
        return xyz2xyz(xyz, source_white_point, destination_white_point, axis,
                       caa)
    else:
        return xyz


# noinspection PyUnusedLocal
@color_conversion('rgb', 'lrgb')
def rgb2lrgb(rgb: np.ndarray,
             rgbs: RgbSpecification=None, **kwargs) -> np.ndarray:
    """
    Convert companded RGB to linear RGB
    
    :param rgb: 
    :param rgbs: 
    :return: 
    """
    if rgbs is None:
        rgbs = get_default_rgb_specification()
    return rgbs.inverse_compand(rgb)


# noinspection PyUnusedLocal
@color_conversion('spectrum', 'xyz')
def spectrum2xyz(spectrum: np.ndarray,
                 wavelengths: np.ndarray,
                 axis: int=None,
                 illuminant: Illuminant=get_default_illuminant(),
                 observer: Observer=get_default_observer(),
                 **kwargs) -> np.ndarray:
    """
    Convert reflectance spectrum to XYZ
    
    :param spectrum: the reflectance spectrum
    :param wavelengths: the wavelengths corresponding to the spectra
    :param axis: The axis along which the spectra lie. If this is None, 
        then the axis is the last axis with a size that matches wavelengths. 
    :param illuminant: the illuminant
    :param observer: the observer
    :return: 
    """
    if axis is None:
        axis = get_matching_axis(spectrum.shape, wavelengths.size)

    # Need to ensure that we reshape all the 1-D parts of the integral, so that
    # they can be broadcast properly
    new_shape = [1] * len(spectrum.shape)
    new_shape[axis] = -1
    wavelengths = wavelengths.reshape(new_shape)
    xbar = observer.get_xbar(wavelengths).reshape(new_shape)
    ybar = observer.get_ybar(wavelengths).reshape(new_shape)
    zbar = observer.get_zbar(wavelengths).reshape(new_shape)

    illuminant_psd = illuminant.get_psd(wavelengths).reshape(new_shape)

    norm_factor = (100
                   / integrate.trapz(illuminant_psd * ybar,
                                     x=wavelengths, axis=axis))

    phi = spectrum * illuminant_psd

    # This is the shape that each XYZ component must be
    component_shape = list(spectrum.shape)
    component_shape[axis] = 1

    def integrate_phi(bar: np.ndarray):
        area = norm_factor * integrate.trapz(bar * phi, wavelengths, axis=axis)
        # noinspection PyTypeChecker
        return np.reshape(area, component_shape)

    x = integrate_phi(xbar)
    y = integrate_phi(ybar)
    z = integrate_phi(zbar)
    return np.concatenate((x, y, z), axis=axis)


# noinspection PyUnusedLocal
@color_conversion('xyy', 'xyz')
def xyy2xyz(xyy, axis: int=None, **kwargs) -> np.ndarray:
    """
    converts from xyY to XYZ
    
    :param xyy: 
    :param axis: 
    :return: 
    """
    if axis is None:
        axis = get_matching_axis(xyy.shape, 3)

    inds = _construct_component_inds(axis, len(xyy.shape), 3)

    # Determine where y iz 0 so we don't divide by it
    nzy = np.nonzero(xyy[inds[1]])
    xyz = np.zeros(xyy.shape)

    xyz[inds[0]][nzy] = xyy[inds[0]][nzy]*xyy[inds[2]][nzy] / xyy[inds[1]][nzy]
    xyz[inds[1]][nzy] = xyy[inds[2]][nzy]
    xyz[inds[2]][nzy] = (
        (1 - xyy[inds[0]][nzy] - xyy[inds[1]][nzy]) * xyy[inds[2]][nzy]
        / xyy[inds[1]][nzy]
    )
    return xyz


@color_conversion('xyz', 'lrgb')
def xyz2lrgb(xyz: np.ndarray,
             axis: int=None,
             illuminant: Illuminant=None,
             observer: Observer=None,
             rgbs: RgbSpecification=None,
             caa: ChromaticAdaptationAlgorithm=None) -> np.ndarray:
    """
    Convert XYZ to linear RGB
    
    :param xyz: 
    :param axis: 
    :param illuminant: 
    :param observer: 
    :param rgbs: 
    :param caa: 
    :return: 
    """
    if axis is None:
        axis = get_matching_axis(xyz.shape, 3)
    if illuminant is None:
        illuminant = get_default_illuminant()
    if observer is None:
        observer = get_default_observer()
    if rgbs is None:
        rgbs = get_default_rgb_specification()
    if caa is None:
        caa = get_default_chromatic_adaptation_algorithm()

    # If the white points are not equal, we will need to convert to the
    # RGB white point
    source_white_point = illuminant.get_white_point(observer)
    destination_white_point = rgbs.white_point
    if not np.allclose(source_white_point, destination_white_point,
                       rtol=1e-5, atol=1e-14):
        xyz = xyz2xyz(xyz, source_white_point, destination_white_point, axis,
                      caa)

    # Get the XYZ values into the correct shape for matrix multiplication
    n_dims = xyz.ndim
    new_dims = list(range(n_dims))
    if axis != n_dims - 1:
        new_dims[-1] = axis
        new_dims[axis] = n_dims - 1
    xyz = xyz.transpose(new_dims)

    input_shape = xyz.shape
    xyz_is_not_matrix = xyz.ndim != 2
    if xyz_is_not_matrix:
        xyz = xyz.reshape((-1, 3))

    # Convert to linear RGB
    m = rgbs.linear_transformation
    lrgb = np.linalg.solve(m.T, xyz.T).T

    # Transform the destination data back to the original shape
    if xyz_is_not_matrix:
        lrgb = lrgb.reshape(input_shape)

    return lrgb.transpose(new_dims)


# noinspection PyUnusedLocal
@color_conversion('xyz', 'xyy')
def xyz2xyy(xyz: np.ndarray,
            axis: int=None,
            illuminant: Illuminant=None,
            observer: Observer=None, **kwargs) -> np.ndarray:
    """
    Convert XYZ to xyY
    
    :param xyz: 
    :param axis: 
    :param illuminant: 
    :param observer: 
    :return: 
    """
    if axis is None:
        axis = get_matching_axis(xyz.shape, 3)
    if illuminant is None:
        illuminant = get_default_illuminant()
    if observer is None:
        observer = get_default_observer()
    inds = _construct_component_inds(axis, xyz.ndim, 3)

    denominator = np.sum(xyz, axis)
    nzd = np.nonzero(denominator)

    xyy = np.zeros(xyz.shape)
    xyy[inds[0]][nzd] = xyz[inds[0]][nzd] / denominator[nzd]
    xyy[inds[1]][nzd] = xyz[inds[1]][nzd] / denominator[nzd]
    xyy[inds[2]] = xyz[inds[1]]
    if not np.all(nzd):
        # For any point that is pure black (X=Y=Z=0), give it the
        # chromaticity of the white point of the specified illuminant and
        # observer.
        white_point = illuminant.get_white_point(observer)
        # to prevent infinite recursion, ensure that the white point is
        # non-black
        if white_point[1] > 0:
            zd = np.logical_not(nzd)
            white_point_xyy = xyz2xyy(white_point, None, illuminant, observer)
            xyy[0][zd] = white_point_xyy[0]
            xyy[1][zd] = white_point_xyy[1]
    return xyy


def xyz2xyz(source_xyz: np.ndarray,
            source_white_point: np.ndarray,
            destination_white_point: np.ndarray,
            axis: int=None,
            caa: ChromaticAdaptationAlgorithm=None) -> np.ndarray:
    """
    Convert XYZ values between two white points
    
    :param source_xyz: 
    :param source_white_point: 
    :param destination_white_point: 
    :param axis: 
    :param caa: 
    :return: 
    """
    if axis is None:
        axis = get_matching_axis(source_xyz.shape, 3)
    if caa is None:
        caa = get_default_chromatic_adaptation_algorithm()

    # Get the source color into the correct shape for matrix multiplication
    n_dims = source_xyz.ndim

    new_dims = list(range(n_dims))
    if axis != n_dims - 1:
        new_dims[-1] = axis
        new_dims[axis] = n_dims - 1
    source_xyz = source_xyz.transpose(new_dims)

    input_shape = source_xyz.shape
    xyz_is_not_matrix = n_dims != 2
    if xyz_is_not_matrix:
        source_xyz = source_xyz.reshape((-1, 3))

    # Convert between the white points
    m = caa.get_linear_transformation(source_white_point,
                                      destination_white_point)
    destination_xyz = source_xyz.dot(m)
    print(destination_xyz)
    # Now convert the shape back to the original
    if xyz_is_not_matrix:
        print('reshaping')
        destination_xyz = destination_xyz.reshape(input_shape)
        print(destination_xyz)

    return destination_xyz.transpose(new_dims)


# noinspection PyUnusedLocal
@color_conversion('xyz', 'xyzr')
def xyz2xyzr(xyz: np.ndarray,
             axis: int=None,
             illuminant: Illuminant=get_default_illuminant(),
             observer: Observer=get_default_observer(), **kwargs) -> np.ndarray:
    """
    Convert XYZ to normalized XYZ reflectance
    :param xyz: the raw xyz values
    :param axis: the axis that the XYZ values lie along 
    :param illuminant: the illuminant
    :param observer: the observer
    :return: the xyz normalized Reflectance
    """
    if axis is None:
        axis = get_matching_axis(xyz.shape, 3)

    new_shape = [1] * len(xyz.shape)
    new_shape[axis] = -1
    white_point = illuminant.get_white_point(observer).reshape(new_shape)
    return xyz / white_point


# noinspection PyUnusedLocal
@color_conversion('xyzr', 'lab')
def xyzr2lab(xyzr: np.ndarray,
             axis: int=None, **kwargs) -> np.ndarray:
    """
    Convert from normalized XYZ to LAB
    
    :param xyzr: normalized XYZ
    :param axis: the axis along which the values lie
    :return: LAB
    """
    if axis is None:
        axis = get_matching_axis(xyzr.shape, 3)

    fxyz = np.zeros(xyzr.shape)
    is_big = xyzr > LAB_EPS
    is_small = xyzr <= LAB_EPS
    fxyz[is_big] = np.power(xyzr[is_big], 1.0/3.0)
    fxyz[is_small] = (LAB_KAPPA * xyzr[is_small] + 16) / 116

    lab = np.zeros(xyzr.shape)
    # Construct the indices for the 3 components
    inds = _construct_component_inds(axis, len(xyzr.shape), 3)
    lab[inds[0]] = 116*fxyz[inds[1]] - 16
    lab[inds[1]] = 500*(fxyz[inds[0]] - fxyz[inds[1]])
    lab[inds[2]] = 200*(fxyz[inds[1]] - fxyz[inds[2]])
    return lab


# noinspection PyUnusedLocal
@color_conversion('xyzr', 'xyz')
def xyzr2xyz(xyzr: np.ndarray,
             axis: int=None,
             illuminant: Illuminant=None,
             observer: Observer=None, **kwargs) -> np.ndarray:
    """
    Convert normalized XYZ to LAB

    :param xyzr:
    :param axis: 
    :param illuminant: 
    :param observer: 
    :return: 
    """
    if axis is None:
        axis = get_matching_axis(xyzr.shape, 3)

    if illuminant is None:
        illuminant = get_default_illuminant()

    if observer is None:
        observer = get_default_observer()

    new_shape = tuple(-1 if dim == axis else 1
                      for dim in range(len(xyzr.shape)))
    white_point = illuminant.get_white_point(observer).reshape(new_shape)
    return xyzr * white_point


def get_matching_axis(shape: Tuple, length: int) -> int:
    """
    Infers the correct axis to use
    
    :param shape: the shape of the input
    :param length: the desired length of the axis
    :return: the correct axis. If multiple axes match, then it returns the last 
             one.
    """
    # noinspection PyUnresolvedReferences
    axis_candidates = np.nonzero(np.array(shape) == length)[0]
    if len(axis_candidates) == 0:
        raise ValueError('Unable to infer axis tue to shape mismatch: ' 
                         '{} =/= {}.'.format(shape, length))
    return axis_candidates[-1]


def _construct_component_inds(axis: int,
                              n_dims: int,
                              n_components: int) -> Tuple[Tuple]:
    """
    Construct a tuple of tuples, where each element extracts the correct 
    component values.
    
    :param axis: 
    :param n_dims: 
    :param n_components: 
    :return: 
    """
    # noinspection PyTypeChecker
    return tuple(
        tuple(slice(i, i+1)
              if dim == axis
              else (slice(None) if dim < n_dims else np.newaxis)
              for dim in range(max(n_dims, 2)))
        for i in range(n_components))


def convert(data: np.ndarray, from_space: str, to_space: str, *args,
            **kwargs) -> np.ndarray:
    """ Convert data between spaces """
    conversion_path = get_conversion_path(from_space, to_space)
    for do_conversion_step in conversion_path:
        data = do_conversion_step(data, *args, **kwargs)
    return data

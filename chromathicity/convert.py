from typing import Tuple

import numpy as np
import scipy.integrate as integrate

from chromathicity.chromadapt import (
    ChromaticAdaptationAlgorithm,
    get_default_chromatic_adaptation_algorithm)
from chromathicity.illuminant import Illuminant, get_default_illuminant
from chromathicity.manage import color_conversion, get_conversion_path
from chromathicity.observer import Observer, get_default_observer
from chromathicity.rgbspec import (RgbSpecification,
                                   get_default_rgb_specification)
import chromathicity.space_names as names
# Constants used for L*a*b* conversions
from chromathicity.util import construct_component_inds, get_matching_axis

LAB_EPS = 0.008856
LAB_KAPPA = 903.3


def convert(data: np.ndarray, from_space: str, to_space: str, *args,
            **kwargs) -> np.ndarray:
    """
    Convert data between spaces


    """
    conversion_path = get_conversion_path(from_space, to_space)
    for do_conversion_step in conversion_path:
        data = do_conversion_step(data, *args, **kwargs)
    return data


@color_conversion(names.LAB, names.NORMALIZED_XYZ)
def lab2xyzr(lab: np.ndarray, *, axis: int=None) -> np.ndarray:
    """
    Convert LAB to normalized XYZ
    
    :param lab: 
    :param axis: 
    :return: 
    """
    if axis is None:
        axis = get_matching_axis(lab.shape, 3)

    inds = construct_component_inds(axis, len(lab.shape), 3)
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


@color_conversion(names.LAB, names.LCH)
def lab2lch(lab: np.ndarray, *, axis: int=None) -> np.ndarray:
    """
    Convert L*a*b* to LCh
    
    :param lab: 
    :param axis: 
    :return: 
    """
    if axis is None:
        axis = get_matching_axis(lab.shape, 3)

    inds = construct_component_inds(axis, lab.ndim, 3)

    lch = np.zeros(lab.shape)

    lch[inds[0]] = lab[inds[0]]
    lch[inds[1]] = np.sqrt(lab[inds[1]]**2 + lab[inds[2]]**2)
    lch[inds[2]] = np.mod((180/np.pi)*np.arctan2(lab[inds[2]], lab[inds[1]]),
                          360.)

    return lch


@color_conversion(names.LCH, names.LAB)
def lch2lab(lch: np.ndarray, *, axis: int=None) -> np.ndarray:
    """
    Converts LCh to L*a*b*
    
    :param lch: 
    :param axis: 
    :return: 
    """
    if axis is None:
        axis = get_matching_axis(lch.shape, 3)

    inds = construct_component_inds(axis, lch.ndim, 3)

    lab = np.zeros(lch.shape)
    lab[inds[0]] = lch[inds[0]]
    h_rad = (np.pi/180)*lch[inds[2]]
    lab[inds[1]] = lch[inds[1]]*np.cos(h_rad)
    lab[inds[2]] = lch[inds[1]]*np.sin(h_rad)

    return lab


@color_conversion(names.LINEAR_RGB, names.RGB)
def lrgb2rgb(lrgb: np.ndarray, *, rgbs: RgbSpecification=None) -> np.ndarray:
    """
    Convert linear RGB to companded RGB
    
    :param lrgb: 
    :param rgbs: 
    :return: 
    """
    if rgbs is None:
        rgbs = get_default_rgb_specification()
    return rgbs.compand(lrgb)


@color_conversion(names.LINEAR_RGB, names.XYZ)
def lrgb2xyz(lrgb: np.ndarray,
             *,
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

    if axis != n_dims - 1:
        xyz = xyz.transpose(new_dims)

    source_white_point = rgbs.white_point
    destination_white_point = illuminant.get_white_point(observer)
    if not np.allclose(source_white_point, destination_white_point,
                       rtol=1e-5, atol=1e-14):
        return xyz2xyz(xyz, source_white_point, destination_white_point, axis,
                       caa)
    else:
        return xyz


@color_conversion(names.RGB, names.LINEAR_RGB)
def rgb2lrgb(rgb: np.ndarray,
             *,
             rgbs: RgbSpecification=None) -> np.ndarray:
    """
    Convert companded RGB to linear RGB
    
    :param rgb: 
    :param rgbs: 
    :return: 
    """
    if rgbs is None:
        rgbs = get_default_rgb_specification()
    return rgbs.inverse_compand(rgb)


@color_conversion(names.RGB, names.HSL)
def rgb2hsl(rgb: np.ndarray,
            *,
            axis: int=None) -> np.ndarray:
    """
    Convert RGB to Hue Saturation Lightness
    :param rgb: 
    :param axis: 
    :return: 
    """
    if axis is None:
        axis = get_matching_axis(rgb.shape, 3)
    big_m, little_m, chroma = _compute_chroma(rgb, axis)

    inds = construct_component_inds(axis, rgb.ndim, 3)

    hsl = np.zeros(rgb.shape)
    hsl[inds[0]] = _compute_rgb_hue(rgb, big_m, little_m, chroma, axis)
    l = 0.5 * (big_m + little_m)
    hsl[inds[2]] = l

    l_lt_one = l < 1.
    if np.any(l_lt_one):
        hsl[inds[1]][l_lt_one] = (
            (big_m[l_lt_one] - little_m[l_lt_one])
            / (1. - np.abs(2*l[l_lt_one] - 1)))

    return hsl


@color_conversion(names.HSL, names.RGB)
def hsl2rgb(hsl: np.ndarray,
            *,
            axis: int=None) -> np.ndarray:
    """
    Convert from Hue Saturation Lightness (HSL) to RGB
    
    :type hsl: np.ndarray
    :param hsl: 
    :param axis: 
    :return: 
    """
    if axis is None:
        axis = get_matching_axis(hsl.shape, 3)

    inds = construct_component_inds(axis, hsl.ndim, 3)

    chroma = ((1. - np.abs(2. * hsl[inds[2]] - 1.))
              * hsl[inds[1]])  # type: np.ndarray

    h_prime = hsl[inds[0]] / 60.
    x = chroma * (1 - np.abs(np.mod(h_prime, 2) - 1))
    rgb1 = _compute_rgb1(hsl.shape, inds, h_prime, x, chroma)

    little_m = hsl[inds[2]] - chroma/2.
    if little_m.ndim > rgb1.ndim:
        # This only happens if hsl is 1-D
        return rgb1 + little_m[0]
    else:
        return rgb1 + little_m


@color_conversion(names.RGB, names.HSI)
def rgb2hsi(rgb: np.ndarray,
            *,
            axis: int=None) -> np.ndarray:
    """
    Convert RGB to Hue Saturation Intensity
    
    :param rgb: 
    :param axis: 
    :return: 
    """
    if axis is None:
        axis = get_matching_axis(rgb.shape, 3)
    big_m, little_m, chroma = _compute_chroma(rgb, axis)

    inds = construct_component_inds(axis, rgb.ndim, 3)

    hsi = np.zeros(rgb.shape)
    hsi[inds[0]] = _compute_rgb_hue(rgb, big_m, little_m, chroma, axis)
    hsi[inds[2]] = np.mean(rgb, axis=axis, keepdims=True)

    i_nz = hsi[inds[2]] != 0  # type: np.ndarray
    if little_m.ndim < i_nz.ndim:
        # This only happens in the 1D case
        little_m = little_m[slice(None), np.newaxis]
    if np.any(i_nz):
        hsi[inds[1]][i_nz] = 1 - little_m[i_nz] / hsi[inds[2]][i_nz]

    return hsi


@color_conversion(names.HSI, names.RGB)
def hsi2rgb(hsi: np.ndarray,
            *,
            axis: int=None) -> np.ndarray:
    """
    Convert Hue Saturation Intensity (HSI) to RGB
    
    :param hsi: 
    :param axis: 
    :return: 
    """
    if axis is None:
        axis = get_matching_axis(hsi.shape, 3)

    inds = construct_component_inds(axis, hsi.ndim, 3)

    h_prime = hsi[inds[0]] / 60.
    z = 1 - np.abs(np.mod(h_prime, 2.) - 1)
    chroma = 3*hsi[inds[2]]*hsi[inds[1]]/(1 + z)  # type: np.ndarray
    x = chroma * z
    rgb1 = _compute_rgb1(hsi.shape, inds, h_prime, x, chroma)
    little_m = hsi[inds[2]] * (1 - hsi[inds[1]])  # type: np.ndarray
    if little_m.ndim > rgb1.ndim:
        # This only happens in the 1D case
        return rgb1 + little_m[0]
    else:
        return rgb1 + little_m


@color_conversion(names.RGB, names.HSV)
def rgb2hsv(rgb: np.ndarray, *, axis: int=None) -> np.ndarray:
    """
    Convert from RGB to Hue Saturation Value (HSV)
    
    :param rgb: 
    :param axis: 
    :return: 
    """
    if axis is None:
        axis = get_matching_axis(rgb.shape, 3)

    big_m, little_m, chroma = _compute_chroma(rgb, axis)

    inds = construct_component_inds(axis, rgb.ndim, 3)

    hsv = np.zeros(rgb.shape)
    hsv[inds[0]] = _compute_rgb_hue(rgb, big_m, little_m, chroma, axis)
    hsv[inds[2]] = big_m

    big_m_nz = big_m != 0
    hsv[inds[1]][big_m_nz] = chroma[big_m_nz] / big_m[big_m_nz]

    return hsv


@color_conversion(names.HSV, names.RGB)
def hsv2rgb(hsv: np.ndarray, *, axis: int=None) -> np.ndarray:
    """
    Convert from HSV to RGB
    :param hsv: 
    :param axis: 
    :return: 
    """
    if axis is None:
        axis = get_matching_axis(hsv.shape, 3)

    inds = construct_component_inds(axis, hsv.ndim, 3)

    chroma = hsv[inds[1]] * hsv[inds[2]]
    h_prime = hsv[inds[0]] / 60.
    x = chroma * (1. - np.abs(np.mod(h_prime, 2.) - 1))  # type: np.ndarray
    rgb1 = _compute_rgb1(hsv.shape, inds, h_prime, x, chroma)
    little_m = hsv[inds[2]] - chroma

    if little_m.ndim > rgb1.ndim:
        # This only happens in the 1D case
        return rgb1 + little_m[0]
    else:
        return rgb1 + little_m


@color_conversion(names.RGB, names.HCY)
def rgb2hcy(rgb: np.ndarray, *, axis: int=None) -> np.ndarray:
    """
    Convert from RGB to Hue, Chroma, Luma (Y'_601)
    
    :param rgb: 
    :param axis: 
    :return: 
    """
    if axis is None:
        axis = get_matching_axis(rgb.shape, 3)

    big_m, little_m, chroma = _compute_chroma(rgb, axis)

    inds = construct_component_inds(axis, rgb.ndim, 3)

    hcy = np.zeros(rgb.shape)
    hcy[inds[0]] = _compute_rgb_hue(rgb, big_m, little_m, chroma, axis)
    hcy[inds[1]] = chroma
    hcy[inds[2]] = 0.299*rgb[inds[0]] + 0.587*rgb[inds[1]] + 0.114*rgb[inds[2]]

    return hcy


@color_conversion(names.HCY, names.RGB)
def hcy2rgb(hcy: np.ndarray, *, axis: int=None) -> np.ndarray:
    """
    
    :param hcy: 
    :param axis: 
    :return: 
    """
    if axis is None:
        axis = get_matching_axis(hcy.shape, 3)

    inds = construct_component_inds(axis, hcy.ndim, 3)

    h_prime = hcy[inds[0]] / 60.
    x: np.ndarray = hcy[inds[1]] * (1 - np.abs(np.mod(h_prime, 2) - 1))
    rgb1 = _compute_rgb1(hcy.shape, inds, h_prime, x, hcy[inds[1]])
    little_m: np.ndarray = hcy[inds[2]] - (0.299*rgb1[inds[0]]
                                           + 0.587*rgb1[inds[1]]
                                           + 0.114*rgb1[inds[2]])

    if little_m.ndim > rgb1.ndim:
        # This only happens in the 1D case
        return rgb1 + little_m[0]
    else:
        return rgb1 + little_m


def _compute_chroma(rgb: np.ndarray,
                    axis: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the chroma for an RGB triple"""
    big_m = np.max(rgb, axis=axis, keepdims=True)
    little_m = np.min(rgb, axis=axis, keepdims=True)
    chroma = big_m - little_m
    return big_m, little_m, chroma


def _compute_rgb_hue(rgb: np.ndarray, big_m, little_m, chroma,
                     axis: int) -> np.ndarray:
    """ Compute the RGB Hue. This is the same for HSV and HSL """
    inds = construct_component_inds(axis, rgb.ndim, 3)
    r = rgb[inds[0]]
    g = rgb[inds[1]]
    b = rgb[inds[2]]
    if chroma.ndim < r.ndim:
        # this will only happen when chroma is 1-D
        chroma = chroma[(slice(None), np.newaxis)]
    ch_is_nz = chroma != 0.
    m_is_r = np.logical_and(big_m == r, ch_is_nz)
    m_is_g = np.logical_and(big_m == g, ch_is_nz)
    m_is_b = np.logical_and(big_m == b, ch_is_nz)

    h_prime = np.zeros(r.shape)
    if np.any(m_is_r):
        h_prime[m_is_r] = np.mod((g[m_is_r] - b[m_is_r]) / chroma[m_is_r] + 6.,
                                 6.)
    if np.any(m_is_g):
        h_prime[m_is_g] = (b[m_is_g] - r[m_is_g]) / chroma[m_is_g] + 2.
    if np.any(m_is_b):
        h_prime[m_is_b] = (r[m_is_b] - g[m_is_b]) / chroma[m_is_b] + 4.
    return 60. * h_prime


def _compute_rgb1(shape, inds, h_prime, x, chroma) -> np.ndarray:
    rgb1 = np.zeros(shape)
    i01 = np.logical_and(0. <= h_prime, h_prime <= 1.)
    i12 = np.logical_and(1. <= h_prime, h_prime <= 2.)
    i23 = np.logical_and(2. <= h_prime, h_prime <= 3.)
    i34 = np.logical_and(3. <= h_prime, h_prime <= 4.)
    i45 = np.logical_and(4. <= h_prime, h_prime <= 5.)
    i56 = np.logical_and(5. <= h_prime, h_prime <= 6.)
    or_ = np.logical_or
    rgb1[inds[0]][or_(i01, i56)] = chroma[or_(i01, i56)]
    rgb1[inds[0]][or_(i12, i45)] = x[or_(i12, i45)]
    rgb1[inds[1]][or_(i01, i34)] = x[or_(i01, i34)]
    rgb1[inds[1]][or_(i12, i23)] = chroma[or_(i12, i23)]
    rgb1[inds[2]][or_(i23, i56)] = x[or_(i23, i56)]
    rgb1[inds[2]][or_(i34, i45)] = chroma[or_(i34, i45)]

    return rgb1


@color_conversion(names.REFLECTANCE_SPECTRUM, names.XYZ)
def spectrum2xyz(spectrum: np.ndarray,
                 wavelengths: np.ndarray,
                 *,
                 axis: int=None,
                 illuminant: Illuminant=get_default_illuminant(),
                 observer: Observer=get_default_observer()) -> np.ndarray:
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

    norm_factor = (1
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


@color_conversion(names.XYY, names.XYZ)
def xyy2xyz(xyy, *, axis: int=None) -> np.ndarray:
    """
    converts from xyY to XYZ
    
    :param xyy: 
    :param axis: 
    :return: 
    """
    if axis is None:
        axis = get_matching_axis(xyy.shape, 3)

    inds = construct_component_inds(axis, len(xyy.shape), 3)

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


@color_conversion(names.XYZ, names.LINEAR_RGB)
def xyz2lrgb(xyz: np.ndarray, *,
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


@color_conversion(names.XYZ, names.XYY)
def xyz2xyy(xyz: np.ndarray, *,
            axis: int=None,
            illuminant: Illuminant=None,
            observer: Observer=None) -> np.ndarray:
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
    inds = construct_component_inds(axis, xyz.ndim, 3)

    denominator = np.sum(xyz, axis, keepdims=True)
    nzd = denominator != 0

    xyy = np.zeros(xyz.shape)
    if np.any(nzd):
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
            white_point_xyy = xyz2xyy(white_point,
                                      illuminant=illuminant,
                                      observer=observer)
            xyy[inds[0]][zd] = white_point_xyy[0]
            xyy[inds[1]][zd] = white_point_xyy[1]
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
    # Now convert the shape back to the original
    if xyz_is_not_matrix:
        destination_xyz = destination_xyz.reshape(input_shape)

    return destination_xyz.transpose(new_dims)


@color_conversion(names.XYZ, names.NORMALIZED_XYZ)
def xyz2xyzr(xyz: np.ndarray, *,
             axis: int=None,
             illuminant: Illuminant=get_default_illuminant(),
             observer: Observer=get_default_observer()) -> np.ndarray:
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


@color_conversion(names.NORMALIZED_XYZ, names.LAB)
def xyzr2lab(xyzr: np.ndarray, *,
             axis: int=None) -> np.ndarray:
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
    inds = construct_component_inds(axis, len(xyzr.shape), 3)
    lab[inds[0]] = 116*fxyz[inds[1]] - 16
    lab[inds[1]] = 500*(fxyz[inds[0]] - fxyz[inds[1]])
    lab[inds[2]] = 200*(fxyz[inds[1]] - fxyz[inds[2]])
    return lab


@color_conversion(names.NORMALIZED_XYZ, names.XYZ)
def xyzr2xyz(xyzr: np.ndarray, *,
             axis: int=None,
             illuminant: Illuminant=None,
             observer: Observer=None) -> np.ndarray:
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

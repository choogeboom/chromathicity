from abc import ABC, abstractmethod

import numpy as np
import scipy.integrate as integrate
from bidict import bidict

from chromathicity.error import raise_not_implemented
from chromathicity.math import interp1
from chromathicity.util import SetGet
from chromathicity.observer import Observer, Standard, get_default_observer


class Illuminant(ABC):
    """
    Interface for illuminants
    
    A standard illuminant is a theoretical source of visible light with a 
    publish profile (power spectral distribution). Standard illuminants
    provide a basis for comparing images or colors recorded under different
    lighting.
    """

    @property
    def name(self) -> str:
        return self.__name__

    def __str__(self):
        return self.name

    @property
    @abstractmethod
    def wavelengths(self) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def psd(self) -> np.ndarray:
        pass

    def get_psd(self, wavelengths: np.ndarray) -> np.ndarray:
        return interp1(wavelengths, self.wavelengths, self.psd)

    def get_white_point(self, observer: Observer=None) -> np.ndarray:
        """
        Calculate the white point of the illuminant with a specified observer
        
        :param observer: The observer 
        :return: the white point
        """
        if observer is None:
            observer = get_default_observer()
        wls = observer.wavelengths
        p = self.get_psd(wls)
        x_power = observer.xbar * p
        is_valid_x = np.logical_not(np.isnan(x_power))
        x_point = integrate.trapz(x_power[is_valid_x], wls[is_valid_x])
        y_power = observer.ybar * p
        is_valid_y = np.logical_not(np.isnan(y_power))
        y_point = integrate.trapz(y_power[is_valid_y], wls[is_valid_y])
        z_power = observer.zbar * p
        is_valid_z = np.logical_not(np.isnan(z_power))
        z_point = integrate.trapz(z_power[is_valid_z], wls[is_valid_z])
        return np.array([x_point, y_point, z_point])/y_point


class D(Illuminant):
    """
    Natural daylight
    
    Derived by Judd, MacAdam, and Wyszecki, the D series of illuminants are
    constructed to represent natural daylight. They are difficult to produce
    artificially, but are easy to characterize mathematically.
    
    This is, by default, equivalent to D_65.
    """

    _WAVELENGTHS = np.arange(300, 835, 5)
    _S = (
        np.array([
            0.04, 3.02, 6.00, 17.80, 29.60, 42.45, 55.30, 56.30, 57.30, 59.55,
            61.80, 61.65, 61.50, 65.15, 68.80, 66.10, 63.40, 64.60, 65.80,
            80.30, 94.80, 99.80, 104.80, 105.35, 105.90, 101.35, 96.80, 105.35,
            113.90, 119.75, 125.60, 125.55, 125.50, 123.40, 121.30, 121.30,
            121.30, 117.40, 113.50, 113.30, 113.10, 111.95, 110.80, 108.65,
            106.50, 107.65, 108.80, 107.05, 105.30, 104.85, 104.40, 102.20,
            100.00, 98.00, 96.00, 95.55, 95.10, 92.10, 89.10, 89.80, 90.50,
            90.40, 90.30, 89.35, 88.40, 86.20, 84.00, 84.55, 85.10, 83.50,
            81.90, 82.25, 82.60, 83.75, 84.90, 83.10, 81.30, 76.60, 71.90,
            73.10, 74.30, 75.35, 76.40, 69.85, 63.30, 67.50, 71.70, 74.35,
            77.00, 71.10, 65.20, 56.45, 47.70, 58.15, 68.60, 66.80, 65.00,
            65.50, 66.00, 63.50, 61.00, 57.15, 53.30, 56.10, 58.90, 60.40, 61.90
        ]),
        np.array([
            0.02, 2.26, 4.50, 13.45, 22.40, 32.20, 42.00, 41.30, 40.60, 41.10,
            41.60, 39.80, 38.00, 40.20, 42.40, 40.45, 38.50, 36.75, 35.00,
            39.20, 43.40, 44.85, 46.30, 45.10, 43.90, 40.50, 37.10, 36.90,
            36.70, 36.30, 35.90, 34.25, 32.60, 30.25, 27.90, 26.10, 24.30,
            22.20, 20.10, 18.15, 16.20, 14.70, 13.20, 10.90, 8.60, 7.35, 6.10,
            5.15, 4.20, 3.05, 1.90, 0.95, 0.00, -0.80, -1.60, -2.55, -3.50,
            -3.50, -3.50, -4.65, -5.80, -6.50, -7.20, -7.90, -8.60, -9.05,
            -9.50, -10.20, -10.90, -10.80, -10.70, -11.35, -12.00, -13.00,
            -14.00, -13.80, -13.60, -12.80, -12.00, -12.65, -13.30, -13.10,
            -12.90, -11.75, -10.60, -11.10, -11.60, -11.90, -12.20, -11.20,
            -10.20, -9.00, -7.80, -9.50, -11.20, -10.80, -10.40, -10.50,
            -10.60, -10.15, -9.70, -9.00, -8.30, -8.80, -9.30, -9.55, -9.80
        ]),
        np.array([
            0.00, 1.00, 2.00, 3.00, 4.00, 6.25, 8.50, 8.15, 7.80, 7.25, 6.70,
            6.00, 5.30, 5.70, 6.10, 4.55, 3.00, 2.10, 1.20, 0.05, -1.10, -0.80,
            -0.50, -0.60, -0.70, -0.95, -1.20, -1.90, -2.60, -2.75, -2.90,
            -2.85, -2.80, -2.70, -2.60, -2.60, -2.60, -2.20, -1.80, -1.65,
            -1.50, -1.40, -1.30, -1.25, -1.20, -1.10, -1.00, -0.75, -0.50,
            -0.40, -0.30, -0.15, 0.00, 0.10, 0.20, 0.35, 0.50, 1.30, 2.10,
            2.65, 3.20, 3.65, 4.10, 4.40, 4.70, 4.90, 5.10, 5.90, 6.70, 7.00,
            7.30, 7.95, 8.60, 9.20, 9.80, 10.00, 10.20, 9.25, 8.30, 8.95, 9.60,
            9.05, 8.50, 7.75, 7.00, 7.30, 7.60, 7.80, 8.00, 7.35, 6.70, 5.95,
            5.20, 6.30, 7.40, 7.10, 6.80, 6.90, 7.00, 6.70, 6.40, 5.95, 5.50,
            5.80, 6.10, 6.30, 6.50
        ])
    )
    STANDARD_TEMPERATURES = bidict({
        'D_50': 5003,
        'D_55': 5503,
        'D_65': 6504,
        'D_75': 7505
    })

    def __init__(self, temperature=None):
        self._temperature = None
        if temperature is None:
            self._set_temperature('D_65')
        else:
            self._set_temperature(temperature)

    def __repr__(self):
        return 'D({:g})'.format(self.temperature)

    @Illuminant.name.getter
    def name(self):
        if self.temperature in self.STANDARD_TEMPERATURES.inv:
            return self.STANDARD_TEMPERATURES.inv[self.temperature]
        else:
            return "D_{:g}K".format(self.temperature)

    @property
    def temperature(self):
        return self._temperature

    def _set_temperature(self, t):
        if isinstance(t, str):
            if t in self.STANDARD_TEMPERATURES:
                self._temperature = self.STANDARD_TEMPERATURES[t]
            else:
                raise ValueError('Illegal illuminant temperature')
        else:
            if 4000 <= t <= 25000:
                self._temperature = t
            else:
                raise ValueError('Illuminant D must have a temperature between ' 
                                 '4000 and 25000.')

    @Illuminant.wavelengths.getter
    def wavelengths(self):
        if self._WAVELENGTHS.flags.writeable:
            self._WAVELENGTHS.flags.writeable = False
        return self._WAVELENGTHS

    def get_psd(self, wavelengths: np.ndarray):
        s = self._get_daylight_components(wavelengths)
        m = self._daylight_coefficients
        return s[0] + m[0]*s[1] + m[1]*s[2]

    def _get_daylight_components(self, wavelengths):
        return tuple(interp1(wavelengths, self.wavelengths, s) for s in self._S)

    @Illuminant.psd.getter
    def psd(self):
        # noinspection PyTypeChecker
        return self.get_psd(self.wavelengths)

    @property
    def _daylight_coefficients(self):
        x, y = self._chromaticity_coordinates
        md = 0.0241 + 0.2562*x - 0.7341*y
        return ((-1.3515 - 1.7703*x + 5.9114*y)/md,
                (0.0300 - 31.4424*x + 30.0717*y)/md)

    @property
    def _chromaticity_coordinates(self):
        t = self.temperature
        if t <= 7000:
            xd = -4.607e9/t**3 + 2.9678e6/t**2 + 0.09911e3/t + 0.244063
        else:
            xd = -2.0064e9/t**3 + 1.9018e6/t**2 + 0.24748e3/t + 0.237040

        return xd, -3*xd**2 + 2.87*xd - 0.275


class A(Illuminant):
    """
    Standard illuminant based on a Planckian Locus
    
    The CIE defines illuminant A in these terms:
    
      CIE standard illuminant A is intended to represent typical, domestic, 
      tungsten-filament lighting. Its relative spectral power distribution is 
      that of a Planckian radiator at a temperature of approximately 2856 K. 
      CIE standard illuminant A should be used in all applications of 
      colorimetry involving the use of incandescent lighting, unless there 
      are specific reasons for using a different illuminant. 
    
          —CIE, CIE Standard Illuminants for Colorimetry
    
    This class is more general than the CIE standard illuminant A in that you 
    can change the color temperature 
    """

    _DEFAULT_TEMPERATURE = 2848.0
    _WAVELENGTHS = np.arange(300.0, 831.0)
    _C_1 = 560.0
    _C_2 = 1.435e7

    def __init__(self, temperature: float=None):
        self.temperature = self._DEFAULT_TEMPERATURE
        if temperature is not None:
            self.temperature = temperature

    def __repr__(self):
        return 'A({})'.format(self.temperature)

    @Illuminant.name.getter
    def name(self):
        if self.temperature == self._DEFAULT_TEMPERATURE:
            return 'A'
        else:
            return 'A_{}K'.format(self.temperature)

    @Illuminant.wavelengths.getter
    def wavelengths(self):
        return self._WAVELENGTHS

    @Illuminant.psd.getter
    def psd(self):
        # noinspection PyTypeChecker
        return self.get_psd(self.wavelengths)

    def get_psd(self, wavelengths: np.ndarray=None) -> np.ndarray:
        if wavelengths is None:
            wavelengths = self.wavelengths
        t = self.temperature
        wl = wavelengths
        c1 = self._C_1
        c2 = self._C_2
        return (
            100 * (c1 / wl)**5.0 * (np.exp(c2 / (t*c1)) - 1)
            / (np.exp(c2/(t*wl)) - 1)
        )


class E(Illuminant):
    """
    Ideal Illuminant
    
    From `Wikipedia <https://en.wikipedia.org/wiki/
    Standard_illuminant#Illuminant_E>`_:
    
        *Illuminant E is an equal-energy radiator; it has a constant SPD 
        inside the visible spectrum. It is useful as a theoretical reference; 
        an illuminant that gives equal weight to all wavelengths, presenting 
        an even color. It also has equal CIE XYZ tristimulus values, thus its 
        chromaticity coordinates are (x,y)=(1/3,1/3). This is by design; the 
        XYZ color matching functions are normalized such that their integrals 
        over the visible spectrum are the same.*

        *Illuminant E is not a black body, so it does not have a color 
        temperature, but it can be approximated by a D series illuminant with 
        a CCT of 5455 K. (Of the canonical illuminants, D55 is the closest.) 
        Manufacturers sometimes compare light sources against Illuminant E to 
        calculate the excitation purity.* 
    """
    _WAVELENGTHS = np.arange(360, 10, 831)

    def __repr__(self):
        return 'E()'

    @Illuminant.wavelengths.getter
    def wavelengths(self):
        if self._WAVELENGTHS.flags.writeable:
            self._WAVELENGTHS.flags.writeable = False
        return self._WAVELENGTHS

    @Illuminant.psd.getter
    def psd(self):
        # noinspection PyTypeChecker
        return self.get_psd(self.wavelengths)

    def get_psd(self, wavelengths: np.ndarray):
        return np.ones((len(wavelengths),))


# noinspection PyMethodOverriding
class CustomIlluminant(Illuminant):
    """
    A class that allows dynamic creation of new illuminants
    """

    def __init__(self, *, name='', wavelengths=(), psd=()):
        self._name = name
        self._wavelengths = np.array(wavelengths)
        self._psd = np.array(psd)

    @Illuminant.name.getter
    def name(self):
        return self._name

    @Illuminant.wavelengths.getter
    def wavelengths(self):
        return self._wavelengths

    @Illuminant.psd.getter
    def psd(self):
        return self._psd


def get_default_illuminant() -> Illuminant:
    return D()

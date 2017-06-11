from chromathicity.chromadapt import Bradford
from chromathicity.illuminant import D
from chromathicity.interfaces import Illuminant
from chromathicity.observer import Standard
from chromathicity.rgbspec import Srgb


def get_default_illuminant() -> Illuminant:
    return D()


def get_default_observer():
    return Standard()


def get_default_rgb_specification():
    return Srgb()


def get_default_chromatic_adaptation_algorithm():
    return Bradford()

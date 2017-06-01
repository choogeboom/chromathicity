Color Spaces
============

Color spaces are defined, for the most part, in the
``chromathicity.spaces`` \module.

Basic Interface
---------------

.. autoclass:: chromathicity.spaces.ColorSpaceData
   :members:


Spectral Data
-------------

.. autoclass:: chromathicity.spaces.ReflectanceSpectrumData
   :members:


CIE Spaces
----------

.. autoclass:: chromathicity.spaces.XyzData

.. autoclass:: chromathicity.spaces.XyyData

.. autoclass:: chromathicity.spaces.LabData


RGB Spaces
----------

.. autoclass:: chromathicity.spaces.RgbData


Defining New Spaces
-------------------

New color spaces should extend :class:`~chromathicity.spaces.ColorSpaceData`,
and use the :func:`~chromathicity.spaces.color_space` decorator to register the
name of the color space.

.. autofunction:: chromathicity.spaces.color_space

The easiest way to create a new color space is to extend one of the following
classes, which take care of many implementation details.

.. autoclass:: chromathicity.spaces.ColorSpaceDataImpl
   :members:

.. autoclass:: chromathicity.spaces.WhitePointSensitive
   :members:

.. autoclass:: chromathicity.spaces.RgbsSensitive
   :members:


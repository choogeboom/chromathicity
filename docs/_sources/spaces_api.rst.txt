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
   :show-inheritance:


CIE Spaces
----------

.. autoclass:: chromathicity.spaces.XyzData
   :show-inheritance:

.. autoclass:: chromathicity.spaces.XyyData
   :show-inheritance:

.. autoclass:: chromathicity.spaces.LabData
   :show-inheritance:


RGB Spaces
----------

.. autoclass:: chromathicity.spaces.RgbData
   :show-inheritance:

.. autoclass:: chromathicity.spaces.HslData
   :show-inheritance:

.. autoclass:: chromathicity.spaces.HsiData
   :show-inheritance:

.. autoclass:: chromathicity.spaces.HcyData
   :show-inheritance:

.. autoclass:: chromathicity.spaces.HsvData
   :show-inheritance:


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
   :show-inheritance:

.. autoclass:: chromathicity.spaces.WhitePointSensitive
   :members:
   :show-inheritance:

.. autoclass:: chromathicity.spaces.RgbsSensitive
   :members:
   :show-inheritance:


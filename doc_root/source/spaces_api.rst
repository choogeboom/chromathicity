Color Spaces
============

Color spaces are defined, for the most part, in the
``chromathicity.spaces`` \module.

Basic Interface
---------------

.. autoclass:: chromathicity.spaces.ColorSpaceData
   :members:


.. autoclass:: chromathicity.spaces.ColorSpaceDataImpl
   :members:
   :inherited-members:


Spectral Data
-------------

.. autoclass:: chromathicity.spaces.SpectralData
   :members:
   :inherited-members:


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

New color spaces should extend :class:`ColorSpaceDataImpl`, and use the
:func:`color_space` decorator to register the name of the color space

.. autofunction:: chromathicity.spaces.color_space

.. autoclass:: chromathicity.spaces.WhitePointSensitive
   :members:

.. autoclass:: chromathicity.spaces.RgbsSensitive
   :members:


Tutorial
********

===========
Basic Usage
===========

Colorimetric measurements only make sense in the context of a color space, which
defines the coordinates of the measurement. In Chromathicity, color spaces
are represented by the various subclasses of
:class:`~chromathicity.spaces.ColorSpaceData`. Almost all of the basic usage
will be interacting with instances of these color spaces. This tutorial aims to
give a brief tour of what is possible with color spaces. For more detailed
information, head over to the :doc:`Spaces API page <spaces_api>`

Converting between spaces
-------------------------

First, create your data. Here, two points of data from the CIELAB space are
being packaged together in a single :class:`~chromathicity.spaces.LabData`
object.::

   >>> from chromathicity.spaces import LabData
   >>> lab = LabData([[45., 35., 25.], [55., -30., -40.]])
   LabData(array([[ 45.,  35.,  25.],
          [ 55., -30., -40.]]), axis=1, illuminant=D(6504), observer=Standard(2), rgbs=Srgb(), caa=Bradford(), is_scaled=False)

Now you can convert the data to another color space::

   >>> rgb = lab.to('RGB', is_scaled=True)
   RgbData(array([[ 168.30012917,   80.2952331 ,   66.72228311],
          [   0.        ,  148.73553925,  200.15555912]]), axis=1, illuminant=D(6504), observer=Standard(2), rgbs=Srgb(), caa=Bradford(), is_scaled=True)


Individual color components can be accessed through the
:attr:`~chromathicity.spaces.ColorSpaceData.components` attribute. For example,
we can access the *red* component from the RGB data above in the following way::

    >>> rgb.components[0]
    array([[ 168.30012917],
           [   0.]])


Data axis
---------

All color data is stored internally as a ``numpy.ndarray``, and the
:attr:`~chromathicity.ColorSpaceData.axis` of the data is the axis of the
array that the components of the color space lie along. Changing the axis
will permute the dimensions of the array ::

   >>> rgb.axis = 0
   RgbData(array([[ 168.30012917,    0.        ],
           [  80.2952331 ,  148.73553925],
           [  66.72228311,  200.15555912]]), axis=0, illuminant=D(6504), observer=Standard(2), rgbs=Srgb(), caa=Bradford(), is_scaled=True)

If you don't specify an ``axis`` for the data when you create the color space
object, then the axis will be determined by the last dimension with the correct
size. It is important to specify the ``axis`` when multiple dimensions could
potentially be interpreted as the correct axis for the data. For example if you have data lying in the
columns of a 3x3 array representing data from a 3-dimensional color space::

   >>> lab = LabData([[45., 60., 30.], [-35., 60., 25.], [-25., -20., -55.]])

In this case, the axis will be incorrectly determined to be ``1``, when it
should have been ``0``. Be sure to specify the axis to clear up ambiguities::

   >>> lab = LabData([[45., 60., 30.], [-35., 60., 25.], [-25., -20., -55.]], axis=0)



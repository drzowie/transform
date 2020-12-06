=========
Transform
=========

Transform implements a general-purpose coordinate transformation
framework for use with NumPy and other scientific analysis software.
It defines an object (a Transform) that represents a function mapping
R^N->R^M.  These Transform objects can be inverted, composed, and
manipulated in the same way as their mathematical counterparts. They
can be used in two important ways: directly, to manipulate vectors
and large arrays of vectors within the NumPy ndarray formalism; and
indirectly, to resample discretized data sets such as images.

Subclasses of Transform implement parameterized families of
mathematical operations; instances of each subclass represent
particular operations from the family, specified at construction
time.

The package includes the "transform" module, which defines the
main Transform class and some useful general-purpose families of
transformations including support for the World Coordinate System
transformations that have been adopted into the FITS scientific
data standard; and also, once they are released, the 
"transform.cartography" and "transform.color" modules which define 
more specific groups of transformations for specific applications

Typical usage
=============

To load the transform package including defining the Transform object
and a collection of useful Transform subclasses, do this::

    import transform as t
    import numpy as np

Vector manipulation
-------------------

Transforms can be used on vector data to manipulate them.  An example
usage is::
      
    a = t.Scale(3, dim=2)
    b = np.array( [[1,2],[4,5],[5,6]] )
    c = a.apply(b)
    print(c)

That snippet should output::

    [[ 3  6]
     [ 9 12]
     [15 18]]

as the transform "a" represents multiplication of all dimensions by 3.

Other transform instances can represent arbitrarily complex operations.

Image manipulation
------------------

Transforms can also be used on image data, to change the meaning of the
intrinsic pixel coordinate system and/or resample the image to a new
coordinate system.  To aid scientific usage, Transform can also interpret and 
manipulate the World Coordinate System (WCS) tags present in many scientific
image FITS headers.  This interpretation is managed through the 
Transform.WCS object that is included with the package.  This is useful,
e.g., for aligning images of the same subject collected with different 
instruments provided that they have WCS tags attached.


An example usage is::

     a = astropy.io.fits.open('myfile.fits')
     trans = t.Rotation(45,'deg')
     b = trans.remap(a,method='lin')

which loads 'myfile.fits' into a, and returns it rotated 45 degrees about
its scientific origin, in b.

The two methods used for image resampling this way are remap(), which
includes WCS interpretation and autoscaling; and resample(), which uses
only the intrinsic pixel coordinate system.


History
=======

This package was ported and adapted from a Perl Data Language module (PDL::Transform)
first written in 2001.

Contributors
============

Craig DeForest  - concept, architecture, development

Matt West       - development & testing

Jake Wilson     - initial prototype development

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
data standard; and also the "transform.cartography" and
"transform.color" modules which define more specific groups
of transformations for specific applications.

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
      
    a = t.Scale(3, dims=2)
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
coordinate system.

An example usage is::

    TBD


History
=======

This package was ported and adapted from a Perl Data Language module (PDL::Transform)
first written in 2001.

Contributors
============

Craig DeForest  - concept, architecture, development

Matt West       - development & testing

Jake Wilson     - initial prototype development

# -*- coding: utf-8 -*-
'''Transform - Coordinate transforms, image warping, and N-D functions

   The transform module defines a Transform object that represent N-dimensional
   mathematical coordinate transformations.  The Transform objects can be
   used to transform vectors and/or to resample images within the NumPy 
   framework of structured arrays.  The base package is supplied
   with subclasses that implement several general-purpose transformations; 
   additional subpackages supply suites of transformations for specialized 
   purposes (e.g. cartography and color manipulation)
   
   The simplest way to use a Transform object is to transform vector data
   between coordinate systems.  The "apply" method accepts an array or variable
   whose 0th dimension is coordinate index (subsequent dimensions are
   broadcast) and transforms the vectors into a different coordinate system.
   
   Transform also includes image resampling, via the "map" method.  You 
   define a Transform object, then use it to remap a structured array such as 
   an image.  The output is a resampled image.  The "map" method works well 
   with the FITS standard World Coordinate System (Greisen & Calabretta 2002, 
   A&A 395, 1061), to maintain both the natural pixel coordinate system of the 
   array and an associated world coordinate system related to the pixel location.
   
   You can define and compose several transformationas, then apply them all
   at once to an image.  The image is interpolated only once, when all the
   composed transformations are applied.
   
   Examples
   --------
       import transform as t          # Load the package
       xf = t.Scale(3,dims=2)         # Generate a 2D 3x scaling operation
       newData = xf.apply(myData)     # Apply the scaling to some vectors
       newImage = xf.map(oldImage)    # Resample an image to zoom in 3x
   
   Methods
   -------
   
   Transform objects have the following public methods:
       
       - apply: accepts an ndarray and transforms it in the forward or 
            reverse direction, using the original Transform.
       
       - invert: accepts an ndarray and inverse-transforms it (this is 
            syntactic sugar for apply(data,invert=1) ).
       
       - inverse: returns a Transform that is the functional inverse 
            of the original Transform.
            
       - compose: [UNDER CONSTRUCTION]: returns a Transform that 
            is the composition of two or more existing Transforms.
       
       - map: accepts an ndarray and resamples it using the original
            Transform.
       
        
   Built-in Subclasses
   -------------------
   
   The Transform class is a container for the subclasses that do the actual 
   mathematical work.  Each subclass represents a parameterized family of 
   mathematical operations; an instance of that subclass represents a single
   coordinate transformation selected from the family by the parameters you
   pass to the constructor.
   
   The transform module itself defines the following subclasses:
       
       - Identity: the identity transform
       
       - Inverse: the inverse of any Transform (supplied to the constructor)
           Note that you probably want to use the 'inverse' method to make inverses,
           as it produces a more streamlined object in the case that you want the 
           inverse of an Inverse.
           
       - Composition: groups together multiple Transforms into one composite Transfom.
       
       - Wrap: produces a wrapped transformation of the form B^-1 o A o B; this
           construction is common mathematically.
       
    In addition, the transform.basic module is imported automatically into transform
    and defines:
        
       - Linear: linear transformations including offsets, scales, and rotations;
           There are several further convenience subclasses:
               - Scale
               - Rotation
               - Offset
       
       - Polar: transformations to/from 2-D polar (or 3-D cylindrical) coordinates
       
       - WCS: linear transformations supporting the World Coordinate System
           specification used in FITS images to map array pixel coordinates to/from 
           real-world scientific coordinates
         
       - Projective: the family of projective transformations in 2-D; these are 
           first-order nonlinear projections with some nice properties, and
           are frequently used in perspective correction or rendering.
'''
# Load the main modules
from .core import *
from .basic import *


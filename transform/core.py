# -*- coding: utf-8 -*-

import copy
import os.path
import numpy as np
import astropy 
import astropy.io.fits
import astropy.wcs
import astropy.units
import re
ap = astropy


from transform.helpers import interpND

class Transform:
    '''Transform - Coordinate transforms, image warping, and N-D functions
    
    The transform module defines a Transform object that represents an 
    N-dimensional mathematical coordinate transformation.  Transform objects 
    can be used to transform vectors and/or to resample images within the NumPy 
    framework of structured arrays.  The base package is supplied with 
    subclasses that implement several general-purpose transformations; 
    additional subpackages supply suites of transformations for specialized 
    purposes (e.g. cartography and color manipulation).
    
    The simplest way to use a Transform object is to transform vector data
    between coordinate systems.  The "apply" method accepts an array or variable
    whose final dimension is coordinate index (prior dimensions are broadcast)
    and transforms the vectors according to the formulae embedded in the 
    object.
    
    Transform also includes image resampling, via the "map" method.  You 
    define a Transform object, then use it to remap an N-dimensional array 
    such as an image (N=2) or collection of images (N=3).  The output is a
    structured array (e.g., image) whose pixel coordinate system transformed
    relative to the orginal array. The "map" method works closely with  
    with the FITS standard World Coordinate System (Greisen & Calabretta 2002, 
    A&A 395, 1061), to maintain both the natural pixel coordinate system of the 
    array and an associated world coordinate system related to the pixel 
    location.  A "FITS" Transform object is also provided that performs WCS
    pixel-to-world coordinate transformations and their inverses.
    
    You can define and compose several transformationas, then apply or map 
    them all at once on a data set.  The data are interpolated only once, whe
    all the composed transformations are mapped.
    
    NOTE: Transform considers images to be 2-D arrays indexed in conventional, 
    sane order: the pixel coordinate system is defined so that (0,0) is at the 
    *center* of the LOWER, LEFT pixel of an image, with (1,0) being one pixel 
    to the RIGHT and (0,1) being one pixel ABOVE the origin, i.e. pixel vectors
    are considered to be (X,Y) by default. This indexing method agrees with 
    nearly the entire scientific world aside from the NumPy community, which
    indexes image arrays with (Y,X), and the SciPy community, which sometimes
    indexes images arrays with (Y,-X) to preserve handedness.  For this reason,
    if you use Transformed vectors directly to index array data (outside of
    the map method) then you must reverse the order of normal Transform vector
    components.  A handy ArrayIndex subclassed Transform is supplied, to do 
    this conversion from sane coordinates to NumPy array index coordinates.
     
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
          
        - ArrayIndex: reverses data vectors so that they can be used as array
          indices in NumPy arrays, or vice versa.
      
    In addition, the transform.basic module is imported automatically into transform
    and defines:
        
        - Linear: linear transformations including offsets, scales, and rotations;
            There are several further convenience subclasses:
                - Scale
                - Rotation
                - Offset
                
        - WCS: transformation representing the conversion from standard pixel 
            coordinates to world coordinates using the World Coordinate System
            specification by Greisen & Calabretta -- this is typically used in
            FITS-format images to map array pixel coordinates to/from 
            real-world scientific coordinates
            
        - Polar: Convert Cartesian to linear or conformal (logarithmic) polar 
            coordinates
          
        - Projective: the family of projective transformations in 2-D; these are 
            first-order nonlinear projections with some nice properties, and
            are frequently used in perspective correction or rendering.
            
            
            
    WCS Handling
    ------------
    Transform is built specifically to work well with the World Coordinate 
    System that is used to convert pixel coordinates to/from scientific world 
    coordinates in scientific images in FITS format.  The Transform.WCS subclass
    is a wrapper around the astropy.wcs.WCS object, which implements the full
    WCS standard. 
    
    WCS objects are also used by the remap() method to manage scaling of the
    pixel grid.  The first step in the WCS transformation chain (Greisen & 
    Calabretta, 2002 -- their Paper I) is a linear transformation from pixel
    coordinates to "intermediate world coordinates", and this linear transform
    is understood natively by the remap() method.  Remap() either generates or
    modifies the linear portion of the WCS transform as needed, to convey the
    relationship of the pixel grid to the scientific coordinates.
    '''
    def __init__(self):
       raise AssertionError(\
           "generic transforms must be subclassed (e.g. Transform.identity)"\
               )


           
    def __str__(self):

        '''
        __str__ - stringify a generic transform
        
        The Transform stringifier handles putting any generic descriptors
        on the output string. It wraps a more specific string that is 
        supplied by subclassed __str__s.  Because __str__ can't take any
        arguments directly, subclasses should store their output in 
        self._strtmp, then call super().__str__().  The ._strtmp gets
        deletedh ere.
        
        Some subclasses (notably Inverse and Composition) want to generate 
        strings for additional Transforms embedded inside them. Stringifying 
        those sub-objects should be more terse than a regular stringification, 
        so there's a separate ._str_not_top_tmp flag that they set to turn off 
        the contextual portion of the string.
        Returns
        -------
        str
            the string.
        '''
        try:
            s=self._strtmp
        except:
            return "Generic Transform stringifier - you should never see this"
        del self._strtmp

        # Some subclasses stringify by listing subsidiary transforms. 
        # They call __str__ on those transforms, but want the verbose
        # "Transform()" suppressed.  (That's mostly Inverse and Composition)
        # Those subclasses can set the flag in their subsidiary objects.
        # If the flag is set then we return just the string we're passed.
        # If the flag exists we always end up deleting it.
        try:
            flag = self._str_not_top_tmp
            del self._str_not_top_tmp
            if(flag):
                return s
        except:
            pass
        
        return f"Transform( {s} )"
    
    
    
    def apply(self, data, invert=False):
       
        '''
        Apply - apply a mathematical transform to a set of N-vectors
        
        Parameters
        ----------
        data : ndarray
          This is the data to which the Transform should be applied.
          The data should be a NumPy ndarray, with the final axis 
          running across vector dimension.  Earlier dimensions are broadcast
          (e.g., a WxHx2 NumPy array is treated as a WxH array of 2-vectors).
          The -1 axis must have sufficient size for the transform to work.
          If it is larger, then subsequent vector dimensions are ignored , so 
          that (for example) a 2-D Transform can be applied to a WxHx3 NumPy 
          array and the final WxH plane is transmitted unchanged.  
          
        invert : Boolean, (default False)
          This is an optional flag indicating that the inverse of the transform
          is to be applied, rather than the transform itself. 
          
        Raises
        ------
        ValueError
          Dimensional mismatch or non-array inputs cause this to be thrown.
          
        AssertionError
          This gets raised if the Transform won't work in the intended direction.
          That can happen because some mathematical transforms don't have inverses;
          Transform objects contain a flag indicating validity of forward and 
          reverse application.
          
        Returns
        -------
        numpy.ndarray
            The transformed vector data are returned as a numpy.ndarray.  Most 
            Transforms maintain the dimensionality of the source vectors.  Some 
            embed (increase dimensionality of the vectors) or project (decrease
            dimensionality of the vectors); additional input dimensions, if 
            present, are still appended to the output vectors in all any case.
        '''
        
        # Start by making sure we have an ndarray
        if( not isinstance(data, np.ndarray) ):
            data = np.array(data)
                
        if(invert):    
            if(self.no_reverse):
                raise AssertionError(f"This {self.__str__()} is invalid in the reverse direction")

            if( data.shape[-1] < self.odim ):
                raise ValueError(f"This {self.__str__()} requires {self.odim} dimensions; data have {data.shape[-1]}")
 
            if(self.odim > 0  and  data.shape[-1] > self.odim ):
                data0 = data[...,0:self.odim]
                data0 = self._reverse(data0)
                data = np.append( data0, data[...,self.idim:], axis=-1 )
            else:
                data = self._reverse(data)
                
            return data

        else:  
            if(self.no_forward):
                raise AssertionError("This Transform ({self.__str__()}) is invalid in the forward direction")
            
            if( data.shape[-1] < self.idim ):
                raise ValueError(f"This {self.__str__()} requires {self.idim} dimensions; data have {data.shape[-1]}")

            if ( self.idim > 0  and  data.shape[-1] > self.idim ):
                data0 = data[...,0:self.idim]
                data0 = self._forward(data0)
                sl0 = data[...,self.idim:]
                data = np.append( data0, data[...,self.idim:], axis=-1 )
            else:
                data = self._forward(data)
            
            return data
        
        
        
    def invert(self, data, invert=False):
        '''
        invert - syntactic sugar to apply the inverse of a transform (see apply)
        
        Parameters
        ----------
        data : ndarray
            The data to be transformed
        invert : Boolean, optional
            This works just like the "invert" flag for apply(), but in the reverse
            sense:  if False (the default), the reverse transform is applied. 
            
        Returns
        -------
        ndarray
            The transformed vector data are returned as a NumPy ndarray.
        '''
        return self.apply(data,invert=not(invert))   
    
    
    def composition(self, target=None):
        '''
        composition - generate the composition of this Transform with another
        
        The compose method is syntactic sugar for Composition, which is a 
        subclass of Transform. It is functionally equivalent to the 
        constructor Composition([this, that, ...])
        
        Parameters
        ----------
        target : Transform or list of Transforms

        Returns
        -------
        None.

        '''
        
        if( isinstance(target, list) or isinstance(target, tuple)):
            lst = list(target)
            lst.insert(0,self)
            return Composition(lst)
        else:
            return Composition([self,target])
        
    
    def inverse(self):
        '''
        inverse - generate the functional inverse of a Transform
        
        For most Transform objects, <obj>.inverse() is functionally 
        equivalent to transform.Inverse(<obj>).  But the method is
        overloaded in the Inverse subclass, to produce a cleaner
        output.  So if you want the inverse of a generic Transform,
        you should use its inverse method rather than explicitly 
        constructing a transform.Inverse().
        
        Returns
        -------
        Transform
            The return value is a Transform object that represents the
            mathematical inverse of the supplied Transform.  
        '''
        return(Inverse(self))

    
    def _forward(self, data):
        '''
        _forward - execute the forward transform
        
        This private method does the actual transformation.  It must be
        subclassed, and this method in Transform itself just raises an
        exception.
        
        Parameters
        ----------
        data : ndarray
            This is the data to transform (see apply()).
            
        Raises
        ------
        AssertionError
            The Transform._forward method always raises an exception.  Subclasses
            should overload the method.
            
        Returns
        -------
        None
            Subclassed _forward methods should return the manipulated ndarray.
        '''             
        raise AssertionError(\
            "Transform._forward should always be overloaded by a subclass."\
            )
        
        
    def _reverse(self, data):
        '''
        _reverse - execute the reverse transform
        
        This private method does the actual inverse transformation.  it must
        be subclassed, and this method in Transform itself just raises an
        exception.
        
        Parameters
        ----------
        data : ndarray
            This is the data to transform (see apply()).
            
        Raises
        ------
        AssertionError
            The Transform._reverse method always raises an exception.  Subclasses
            should overload the method.
            
        Returns
        -------
        None
            Subclassed _reverse methods should return the manipulated ndarray.
        '''
        raise AssertionError(\
            "Transform._reverse should always be overloaded by a subclass."\
            )
          
 
    def resample(self, data, /, 
            method='n',
            bound='t',
            phot='radiance',
            shape=None 
            ):
        '''
        resample - use a transform to resample a data array in pixel coordinates
        
        This method implements resampling of gridded data by applying the 
        Transform to the implicit coordinate system of the sampled data,
        and resampling the data to a new pixel grid matching the transformed
        coordinates. The output data have their size determined by a supplied
        shape vector, or matched to the input array if no shape is supplied.
        
        The method works by using the inverse Transform to map *from* the 
        output space back *to* the input space. The output data samples
        are then interpolated from the locations in the input space.
        
        For most scientific remapping you don't want resample, you want map(),
        which handles WCS and autoscaling.
        
        Parameters
        ----------
        
        data : ndarray
            This is the gridded data to resample, such as an image.  It must 
            have at least as many dimensions as the idim of the Transform 
            (self).
            
        /method : string (default 'sample')
            This string indicates the interpolation method to use.  Only
            the first character is checked.  Possible values are those 
            used for transform.helpers.interpND, plus the anti-aliasing
            filters "Gaussian" and "Hanning".  Items marked with "(*)"
            don't preserve the original value even on pixel centers under 
            ideal sampling conditions.
                
                'nearest' - use the value of the nearest-neighbor pixel in
                    the input space.  Very fast, but produces aliasing.
                    
                'linear' - use <N>-linear interpolation. Marginaly bettter than
                    sampling, but still produces phase and amplitude aliasing
                
                'cubic' - use <N>-cubic interpolation. This produces a smoother
                    output than linear for enlargements
                
                'fourier' - use discrete Fourier coefficients. to interpolate
                    between points.  This is useful for periodic data.
                    
                'sinc'    - sinc-function weighting in the input plane; this
                    is equivalent to a hard frequency cutoff in Fourier space in
                    the input plane. The sinc function has zeroes at integer 
                    input-pixel offsets, and is enumerated for 6 input pixels in 
                    all directions.  This limited enumeration introduces small
                    sidelobes in Fourier space.
                    
                'zlanczos' - Lanczos-function weighting in the input plane;
                    this is equivalent to a trapezoidal filter in Fourier space
                    in the input space.  The inner sinc function has zeros at
                    integer input-pixel offsets, and the a parameter is 3, so
                    the kernel extends for 3 input pixels in all directions.
                
                'gaussian' (*) - N-gaussian window weighted sampling in the input plane;
                    the gaussian has a full width half maximum (FWHM) of 1 pixel and
                    is enumerated for 3 input pixels in all directions
                        
                'hanning' - hanning-window weighted sampling in the input plane
                    Hanning window (sin^2) weighting in the input plane
                    
                'rounded' - hanning-like window weighted sampling, with a narrower
                    (1/2 pixel) crossover at the pixel boundaries
                                    
                'Gaussian' (*) - use locally optimized Jacobian-driven filtering,
                    with a Gaussian filter profile in the output plane; the Gaussian
                    has a full width half maximum (FWHM) of 1 output pixel and is
                    enumerated for 3 output pixels in all directions [note 
                    capital 'G'].
                    
                'Hanning' - use locally optimized Jacobian-driven filtering,
                    with a Hanning window profile in the output plane [note 
                    capital 'H']
                    
                'Rounded' - use a 1/4-pixel-wide Hanning window with locally-
                    optimized Jacobian-driven filtering in the output plane
                    
                'ZLanczos' - use a Lanczos filter in the output plane
            
            Most The first seven interpolation methods use the supplied "interpND"
            general purpose interpolator and are subject to aliasing and other
            effects outlined in a paper by DeForest (2004; Solar Physics 219, 3).
            The last two use the numerical Jacobian derivative matrix (local 
            linearization) of the coordinate transform to produce a variable, 
            optimized filter function that reduces or eliminates aliasing.  Gaussian
            sampling uses a Gaussian filter function with nice Fourier properties;
            Hanning resampling uses a Hanning-like filter function that
            is more local than the Gaussian filter.
            

        /phot : string (default 'radiance')
            This string indicates the style of photometry to preserve in the
            data. The default value is useful for most image data.
            Only the first character is tested. Allowable values are:
                
                'radiance' or 'intensive' - output values approximate the local
                value of the input data.
                
                'flux' or 'extensive' - output values are scaled to preserve
                summed/integrated value over each region.
                
            This option is not yet implemented and only intensive treatment
            is supported at present.
            
        /shape : list or tuple None (default None)
            If present, this is the shape of the output data grid, in regular
            array index format (directly comparable to the .shape of the 
            output).  The elements of shape, if specified, should be in 
            (...,Y,X) order just like the .shape attribute of a numpy array
            (vs. the (X,Y,...) order of vectors and indices)

        
        Returns
        -------
        
        The resampled data
        '''
        ##### Make sure we separate the data fork if we get an ImageHDU
        
        if( isinstance( data, np.ndarray ) ):
            data0 = data
        elif( hasattr(data, 'data') and isinstance( data.data, np.ndarray ) ):
            data0 = data.data
        else:
            raise ValueError('Transform.map requires a numpy array or an object containing one')
            
        methodChar = method[0]
        
        ##### Set the output array size
        if( shape is None ):
            shape = data0.shape
        
        ##### Check input, output, and Transform dimensions all agree.
        ##### Okay to pass in *more* dimensions (and let them get broadcast).
        ##### Not okay to pass in *fewer* dimensions.
        if( len(shape) < self.odim ):
            raise ValueError('map: Transform odim must match output data shape')
        if( len(data0.shape) < self.idim ):
            raise ValueError('map: Transform idim must match input data shape')
        if( len(data0.shape) - self.idim  != len(shape) - self.odim ):
            raise ValueError('map: shape and source dimensions must match')
        
        # Enumerate every pixel ( coords[...,Y,X,:] gets [X,Y,...] )
        icoords = self.invert(
            np.mgrid[ 
                tuple( map( lambda i:range(shape[-i-1]), range(len(shape)) ))
                ].transpose()
            )

        # Figure the interpolation.
        if(methodChar in {'G','H','R','Z'}):
            assert("Transform.resample: anti-aliased methods are not yet supported")
            
        output = interpND(data0, icoords, method=methodChar, bound=bound)
        
        return(output)
    
        
    def remap(self, data, /, 
            method='n',
            bound='t',
            phot='radiance',
            shape=None,
            template=None,
            input_range=None,
            output_range=None,
            justify=False,
            rectify=True,
            wcs=None
            ):
        '''
        remap - use a transform to remap scientific data
        
        This method implements resampling of gridded data by applying the 
        Transform to the underlying scientific coordinate system of the 
        scientific data.  The incoming data are either a NumPy array or an 
        object with attributes 'data' (which must be a NumPy array) and 
        either 'header' (which, if present, must contain a FITS header 
        with WCS info) or 'wcs' (which must be an astropy.wcs WCS object).
        You can also supply either a dictionary with those attributes, or 
        a tuple with data and header in that order.
        
        The return value has the same attributes (data and either header or 
        wcs) that were passed in.  If the data object is a recognized 
        and supported class, then that class is returned.  Otherwise 
        a DataWrapper object is returned.  Note that you can force the code 
        to output a DataWrapper, by feeding in a tuple of the correct attributes
        from whatever object you supply.
        
        DataWrappers contain a .data  attribute with the actual data, a 
        .header attribute with a FITS header, and a .wcs attribute with an 
        AstroPy WCS object.
        
        Output data are scaled or autoscaled according to the keyword 
        arguments as described below.  If input_range, output_range, or a full 
        WCS specification (as a FITS header or WCS object) are supplied, then 
        the data are scaled accordingly.  If not, they are autoscaled to
        fit the shape of the output pixel array.
        
        The actual resampling is carried out with the resample() method.
        
        Parameters
        ----------
        
        data : ndarray or object or dictionary
            This is the gridded data to resample, such as an image.  It must 
            have at least as many dimensions as the idim of the Transform 
            (self).  It can be an ndarray or any object that contains one in 
            a "data" attribute or dictionary entry.  If the object or dictionary
            has a "header" attribute or entry, it is parsed as a FITS header
            containing WCS metadata.   
            
        /wcs : FITS header or astropy.wcs.WCS object
            If this is present, it is used as metadata instead of any metadata
            contained in the main "data" parameter.
        
        /method : string (default 'sample')
            This string indicates the interpolation method to use.  Only
            the first character is checked.  Possible values are those 
            used for transform.helpers.interpND, plus the anti-aliasing
            filters "Gaussian" and "Hanning".  Items marked with "(*)"
            don't preserve the original value even on pixel centers under 
            ideal sampling conditions.
                
                'nearest' - use the value of the nearest-neighbor pixel in
                    the input space.  Very fast, but produces aliasing.
                    
                'linear' - use <N>-linear interpolation. Marginaly bettter than
                    sampling, but still produces phase and amplitude aliasing
                
                'cubic' - use <N>-cubic interpolation. This produces a smoother
                    output than linear for enlargements
                
                'fourier' - use discrete Fourier coefficients. to interpolate
                    between points.  This is useful for periodic data.
                    
                'sinc'    - sinc-function weighting in the input plane; this
                    is equivalent to a hard frequency cutoff in Fourier space in
                    the input plane. The sinc function has zeroes at integer 
                    input-pixel offsets, and is enumerated for 6 input pixels in 
                    all directions.  This limited enumeration introduces small
                    sidelobes in Fourier space.
                    
                'zlanczos' - Lanczos-function weighting in the input plane;
                    this is equivalent to a trapezoidal filter in Fourier space
                    in the input space.  The inner sinc function has zeros at
                    integer input-pixel offsets, and the a parameter is 3, so
                    the kernel extends for 3 input pixels in all directions.
                
                'gaussian' (*) - N-gaussian window weighted sampling in the input plane;
                    the gaussian has a full width half maximum (FWHM) of 1 pixel and
                    is enumerated for 3 input pixels in all directions
                        
                'hanning' - hanning-window weighted sampling in the input plane
                    Hanning window (sin^2) weighting in the input plane
                    
                'rounded' - hanning-like window weighted sampling, with a narrower
                    (1/2 pixel) crossover at the pixel boundaries
                                    
                'Gaussian' (*) - use locally optimized Jacobian-driven filtering,
                    with a Gaussian filter profile in the output plane; the Gaussian
                    has a full width half maximum (FWHM) of 1 output pixel and is
                    enumerated for 3 output pixels in all directions [note 
                    capital 'G'].
                    
                'Hanning' - use locally optimized Jacobian-driven filtering,
                    with a Hanning window profile in the output plane [note 
                    capital 'H']
                    
                'Rounded' - use a 1/4-pixel-wide Hanning window with locally-
                    optimized Jacobian-driven filtering in the output plane
                    
                'ZLanczos' - use a Lanczos filter in the output plane
            
            Most The first seven interpolation methods use the supplied "interpND"
            general purpose interpolator and are subject to aliasing and other
            effects outlined in a paper by DeForest (2004; Solar Physics 219, 3).
            The last two use the numerical Jacobian derivative matrix (local 
            linearization) of the coordinate transform to produce a variable, 
            optimized filter function that reduces or eliminates aliasing.  Gaussian
            sampling uses a Gaussian filter function with nice Fourier properties;
            Hanning resampling uses a Hanning-like filter function that
            is more local than the Gaussian filter.
            

        /phot : string (default 'radiance')
            This string indicates the style of photometry to preserve in the
            data. The default value is useful for most image data.
            Only the first character is tested. Allowable values are:
                
                'radiance' or 'intensive' - output values approximate the local
                value of the input data.
                
                'flux' or 'extensive' - output values are scaled to preserve
                summed/integrated value over each region.
                
            This option is not yet implemented and only intensive treatment
            is supported at present.
            
        /template : WCS object or FITS header or FITS HDU (default None)
            If present, template sets the WCS conversion from science
            coordinates to output pixel coordinates.  It also sets the dimension
            of the output array, and overrides the shape, output_range, and 
            input_range specifiers.
            
        /shape : list or tuple or None (default None)
            If present, this is the shape of the output data grid, in regular
            array index format (directly comparable to the .shape of the 
            output).  The elements of shape, if specified, should be in 
            (...,Y,X) order just like the .shape attribute of a numpy array
            (vs. the (X,Y,...) order of vectors and indices)
            
        /output_range: Nx2 NumPy array, or None (default None)
            If present, this is the range of science coordinates to support
            in the output array.  The ...,0 element is the minimum and the 
            ...,1 element is the maximum for each of N dimensions.  N must
            match the odim of the Transform being used to remap.  The range
            runs from edge to edge of the image, so the minimum value is 
            the minimum at the lowest-value edge or corner of the limiting
            pixel.  The maximum is at the highest-value edge or corner of the
            limiting pixel.
            
        /input_range: Nx2 NumPy array, or None (default None)
            If present, this is the range of science coordinates to map
            from input space to the output array.  The output array is 
            autoscaled to contain the range, by forward-transforming a few
            vectors from the input space to the output space.  The input
            range runs from edge to edge of the image, just like output_range,
            subject to the limit that the output range is determined from 
            only limited sample of test points.
            
        /justify: Boolean or float (default False)
            If present and true, this causes all science coordinates to have
            the same pixel scale in the output plane during autoscaling;
            it is overridden by the template, output_range, or input_range.
            In the special case of 2D, you can input a number and it is the
            ratio of the two output pixel scales, with larger numbers
            corresponding to larger scale along the Y axis.
            
        /rectify: Boolean (default True)
            If true, this causes autoscaling to rectify the output coordinates,
            so that the pixel axes are aligned with the scientific coordinates.
            If set to false, then any CROTA, CDij, or PCij matrix in the input
            data WCS header is retained, leaving the output coordinates 
            distorted in the same way as the input ones.
            
        Returns
        -------
        
        The resampled data.  During the operation, the data are wrapped in at
        Transform DataWrapper object; and on export they are exported.  In 
        common use cases, this returns the data and header in a compatible form
        to the input.  The return value is a combination of a NumPy array
        (the data) and a FITS header (the header).  These can come back as an
        object, a tuple, or a dictionary depending on the form of the input data.
        '''
        
        # Regularize the input data
        data = DataWrapper(data,template=wcs)
        
        if not isinstance(data.data, np.ndarray):
            raise ValueError("remap: data must be a NumPy array")
        
        if(data.wcs is not None):
            input_trans = WCS(data)
        else:
            input_trans = Identity()
            
        if(self.idim==0):
            idim = len(data.data.shape)
        else:
            idim = self.idim
            
        if(self.odim==0):
            odim = idim
        else:
            odim = self.odim
        
        # Parse and regularize template
        if(template is not None):
            out_template = DataWrapper(this=template)
        else:
            out_template = None
        
        # Set "shape" to be the desired shape of the output.
        # This is the passed-in shape, or the implicit shape in the
        # template, or the shape of the input array.
        if(shape is None):
            if(out_template is None) :
                # No shape and no output template: shape is same as input data
                shape = data.data.shape
            elif(
               out_template.wcs is None or 
               out_template.wcs.pixel_shape is None):
                # Output template exists but has no shape: match output template
                # shape or, failing that, the input data shape
                if(out_template.data is None):
                    shape = data.data.shape
                else:
                    shape = out_template.data.shape
            else:
                # output template exists and has a WCS shape: copy that.
                # shape is (...,Y,X); wcs pixel_shape is (X,Y,...), so 
                # the pixel_shape needs to be reversed.
                shape = list(reversed(out_template.wcs.pixel_shape))
        
        # Figure if autoscaling is necessary and, if it is, then do it by 
        # generating a new out_template.
        if(  (not hasattr(out_template,'wcs'))  or  out_template.wcs is None ):
            
            if(output_range is not None):
                # output_range is present; just validate it.  Must be odim x 2,  
                # where odim is the Transform dimension and 2 runs over (max,min)
                if(not isinstance(output_range,np.ndarray)):
                    output_range = np.array(output_range)
                or_shape = output_range.shape
                if (len(or_shape) != 2  or  
                        (self.odim != 0 and or_shape[0] != self.odim)):
                        raise ValueError("remap: output_range must be Nx2 and match Transform dims")     
                # 
            else:
                # No output_range is present, so we need to autoscale based on 
                # either input_range or the input pixel edges.  isamp gets a sample
                # of input science coordinates; they get transformed forward
                # to find the corresponding output coordinates, which we use to 
                # find an output_range.
                if(input_range is not None):
                    
                    try:
                        ishape = input_range.shape
                    except:
                        raise ValueError('remap: input_range parameter must be an Nx2 NumPy array')
                        
                    if (len(ishape) != 2  or  
                          (ishape[0] != idim)
                          ):
                        raise ValueError("remap: input_range must be Nx2 and match Transform dims")
                    n=11
                    isamp = np.mgrid[ [ range(n) for i in idim ] ].T / (n-1.0)
                    isamp = isamp * (input_range[...,1] - input_range[...,0]) + input_range[...,0]
                else:
                    # No range was specified.  Transform points from the original pixel
                    # grid into original science coordinates to make isamp
                    n = 11
                    isamp = np.mgrid[ [ range(n) for i in range(idim) ] ].T / (n-1.0)
                    isamp = isamp * data.data.shape
                    isamp = isamp - 0.5
                    isamp = WCS(data).apply(isamp)

                # by now isamp has an array of test vectors in the input scientific space.
                # so we apply the main (self) Transform to them to get to the output
                # scientific space.
                osamp = self.apply(isamp)
                
                #Find min and max of each vector component, across all datapoints
                data_axes = tuple(range(len(osamp.shape)-1))
                omin = np.amin(osamp,axis=data_axes)
                omax = np.amax(osamp,axis=data_axes)
                output_range = np.stack((omin,omax),1)

            # TODO: Need to insert code here to pad output_range if it's short 
            # (in the case where we are broadcasting over dimensions beyond 
            # those present in odim).  Instead for now we just assert that we 
            # are not broadcasting.  
            assert(len(shape) == output_range.shape[0])

            # Now we have an output_range, either from a parameter or from autoscaling.
            # Generate a WCS object and stuff it into the out_template.                  
            otwcs = ap.wcs.WCS(naxis=len(shape))
            otwcs.wcs.crpix = [ shape[i]/2 + 0.5 for i in range(len(shape)) ]
            otwcs.wcs.crval = (output_range[...,0]+output_range[...,1])*0.5
            otwcs.wcs.cdelt = [ (output_range[i,1]-output_range[i,0])/(shape[i]) for i in range(len(shape)) ]

            # TODO: Need to sort out types and units here   
            # This commented-out code (3 lines) is not right for broadcasting.
            # Might need to add some sort of method to individual transforms 
            # for units - or if that's too hard, just wing it.
            # Doing nothing (as we do now) is probably wrong but mqybe not 
            # all *that* wrong.
            #if(self.odim > 0):
            #    otwcs.wcs.ctype = copy.copy(self.otype)
            #    otwcs.wcs.cunit = copy.copy(self.ounit)
                
            otwcs.pixel_shape = list(shape).reverse()
            
            out_template = DataWrapper(data.header,template=otwcs)
            
        ## Now we have an out_template -- either via autoscaling or 
        ## via parameter.
        output_trans = out_template.WCS().inverse()
    
        ## Finally ... dispatch the actual resampling
        total_trans = Composition([output_trans, self, input_trans])
        data_resampled = total_trans.resample(data.data, method=method, bound=bound, phot=phot, shape=shape)

        output = DataWrapper(
            (data_resampled,data.header),
            template = out_template
        )
        return output
    
       

###################################
###################################
###
### DataWrapper - wrapper class to contain array data with a FITS header and/or 
### WCS object.  Used by WCS Transform and by remap().  It is NOT a subclass of
### Transform, because it's not a coordinate transformation.

class DataWrapper():
    '''
    DataWrapper - class to manage data with headers
    
    This wrapper class encapsulates objects with a data fork and a WCS info
    fork.  DataWrapper objects have three important attributes: the 
    .data attribute contains a NumPy array (e.g. an image but could have
    other dimensionality).  The .header attribute contains a FITS header 
    (in astropy.io.fits form) that contains WCS information.  The .WCS 
    attribute contains a Transform.WCS object.
    
    DataWrapper objects can be accessed as tuples (in which case they 
    are in (data, header, WCS) order), as dictionaries (in which case 
    the fields are 'data', 'header', or 'WCS'), or as objects with 
    their regular attributes.
    
    The constructor accepts a variety of objects:
            - a DataWrapper
            - a tuple in DataWrapper order with at least one of (data, header,
                WCS different from None)
            - a dictionary with at least one of 'data', 'header' and 'wcs' keys
            - an astropy.io.FITS PrimaryHDU object
            - an astropy.io.FITS header
            - an astropy.io.FITS HDUList, with a PrimaryHDU as its 0 element
        
    DataWrappers are intended to give easy access to their internal structure.
    You can manipulate the header and/or the WCS yourself, but then you need
    to call either the "hdr2wcs" or "wcs2hdr" method, as appropriate, to 
    synchronize the two things.
    
    For ease of plotting images, there is an "extent" method that returns
    an extent in a matplotlib compatible format, for plotting data in 
    scientific coordinates.

    Future:  export methods to:
            - Astropy.io.FITS PrimaryHDU
            - SunPy map
            - ??
    '''
    def __init__( self, this, template=None ):
        data = None
        header = None
        wcs = None
        
        # If it's already a DataWrapper, then return it
        if isinstance(this, DataWrapper):
            return this
        
        # If it's an HDUList then grab the primary HDU - DWIMming for
        # the results of opening a FITS file.
        if isinstance(this, ap.io.fits.hdu.hdulist.HDUList):
            this = this[0]

        ## Parse:
        # Try to get the desired and required fields out of the input object.
        # Afterward, patch stuff up.  This means we look for data, header, and
        # a WCS object.  At the end we have to sort out which ones we got and
        # whether they're any good.

        # Tuple - (data, header, wcs).  Be generous and take a list also
        if (isinstance(this,tuple) or isinstance(this,list)):
            try:
                data = copy.copy(this[0])
                header = copy.copy(this[1])
            except:
                raise ValueError("DataWrapper: tuple requies both data and header")
            try:
                wcs = copy.copy(this[2])
            except:
                pass

        # If it's a PrimaryHDU then parse appropriately.
        elif isinstance(this, ap.io.fits.hdu.image.PrimaryHDU):
            data = copy.copy(this.data)
            header = copy.copy(this.header)
            wcs = ap.wcs.WCS(header)
        
        # If it's just a FITS header, that is okay.
        elif isinstance(this, ap.io.fits.header.Header):
            header = copy.copy(this)
            
        # If it's a WCS object, that is okay also.
        elif( isinstance( this, ap.wcs.wcs.WCS ) ):
            wcs = copy.copy(this)
            
        # If it's a NumPy object, that is also okay.
        elif( isinstance( this, np.ndarray ) ):
            data = copy.copy(this)
            
        # If it's a dict then it must either have data, header, or wcs 
        # tags -- or else look like a FITS header
        elif ( isinstance(this, dict) ):
            if(   ('data' in this) or 
                  ('header' in this ) or
                  ('wcs' in this) ):
                if('data' in this):
                    data = this['data']
                if('header' in this):
                    header = this['header']
                if('wcs' in this):
                    wcs = this['wcs']
            # Doesn't have tags -- check if it looks like a FITS header
            elif( ('NAXIS' in this) or ('SIMPLE' in this)):
                header = copy.copy(this)
            else:
                raise ValueError(
                    "DataWrapper: dict requires either: FITS header tags; or "
                    "'data', 'header' and/or 'wcs'"
                )      
        
        # If it has data, header, or wcs attributes then pull those
        elif( hasattr( this, 'data') or 
              hasattr( this, 'header') or
              hasattr( this, 'wcs') ):
            if(hasattr(this, 'data')):
                data = copy.copy(this.data)
            if(hasattr(this, 'header')):
                header = copy.copy(this.header)
            if(hasattr(this, 'wcs')):
                wcs = copy.copy(this.wcs)
        
        else:
            raise ValueError("DataWrapper: couldn't parse meaningful data "
                             f"from the supplied {this.__class__}.")

    
        ############################
        # Now we've gathered one or more of data, header, and wcs. 
        # Make sense of what we've got:
        if( header is None):
            
            # No header but we got a valid WCS object
            if( wcs is not None):
                header = wcs.to_header()
                header['NAXIS'] = wcs.naxis
                if(wcs.pixel_shape is None and data is None):
                    raise ValueError("DataWrapper: couldn't determine "
                                     "size of the pixel canvas (no header; "
                                     "wcs object has no pixel_bound)")
                if( data is not None ):
                    # array shape is reversed (...,Y,X)
                    for ii in range(len(data.shape)):
                        header[f"NAXIS{len(data.shape)-ii}"] = data.shape[ii]
                else:
                    # pixel_shape field is forward (X,Y,...)
                    for ii in range(len(wcs.naxis)):
                        header[f"NAXIS{ii+1}"] = wcs.pixel_shape[ii]
                        
            # No header and no valid WCS object, but we have a data file:
            # make a basic FITS header describing pixel coordinates.
            elif( data is not None ):
                if( not isinstance(data, np.ndarray)):
                    raise ValueError("DataWrapper: data must be a numpy array")
                
                h = {
                    'NAXIS':len(data.shape)
                    }
                for ii in range(len(data.shape)):
                    axis = len(data.shape)-ii
                    h[f"NAXIS{axis}"]=data.shape[ii]
                    h[f"CDELT{axis}"]=1.0
                    h[f"CUNIT{axis}"]="Pixel"
                    if(axis==1):
                        ctype='X axis'
                    elif(axis==2):
                        ctype='Y axis'
                    elif(axis==3):
                        ctype='Z axis'
                    else:
                        ctype=f"axis {axis}"
                    h[f"CTYPE{axis}"]=ctype
                    h[f"CRPIX{axis}"]=1
                    h[f"CRVAL{axis}"]=0
                h["COMMENT"]="Header autogenerated by transform.DataWrapper"
                header = FITSHeaderFromDict(h)
            
            else:
                raise ValueError("DataWrapper: requires ",
                                 "data, header, and/or WCS object.")
        
        # Got a header
        else:
            if( isinstance(header, dict) ):
                header = FITSHeaderFromDict(header)
            elif( not isinstance(header, ap.io.fits.header.Header) ):
                header = ap.io.fits.header.Header(header)
            
            # We got and parsed a header.  Now, if necessary, make a an
            # astropy WCS object to match it. 
            if(wcs is None):
                wcs = ap.wcs.WCS(header)    
            
            # Got both a header and wcs -- assume the user knows what s/he is
            # doing
            else:
                pass
        
        ### Parsed!  
        ### Data can be None; both header and wcs should now be populated.
        
        self.wcs = wcs
        self.header = header
        self.data = data
        self._WCS_ = None       # cached WCS Transform object
        
        if( template is not None ):
            if( isinstance(template, DataWrapper)):
                template = template.header
                
            if(isinstance(template,ap.wcs.wcs.WCS)):
                self.wcs = template
                self.wcs2head()
            
            # We got a template and it's not a WCS.  Try to make it into one.
            else:
                if ( isinstance(template, ap.io.fits.header.Header) ):
                    self.wcs = ap.wcs.WCS(template)

                elif ( isinstance(template,dict) ):
                    self.wcs = ap.wcs.WCS(FITSHeaderFromDict(template))

                # A list or tuple is an *array* shape (...,Y,X order)                    
                elif( isinstance(template, list) or 
                      isinstance(template, tuple) or
                      ( isinstance(template, np.ndarray) and 
                        len(template.shape)==1 
                        )
                      ):
                    if( len(template) != self.header['NAXIS'] ):
                        raise ValueError(
                            "DataWrapper: shape format template "
                            "must match data axis count"
                            )
                    for i in range(self.header['NAXIS']):
                        self.header[f"NAXIS{i}"] = template[-1-i]
                    self.wcs.pixel_shape = [ 
                           template[-1-i] 
                           for i in range(self.header['NAXIS']) 
                           ]
                else:
                    raise ValueError("DataWrapper: template must be a "
                                     "FITS header, WCS object, or shape")
                
                self.wcs2head()
                
        
    def WCS(self):
        '''
        DataWrapper.WCS() - return a WCS Transform associated with this object
        The Transform is also cached in the object itself for faster action 
        if re-used.

        '''
        if(self._WCS_ is None):
            self._WCS_ = _WCSTrans_(self)
        return self._WCS_
        
    def head2wcs(self):
        self.wcs = ap.wcs.WCS(self.header)
        # Clear cached WCS Transfom obejct
        if(self._WCS_ is not None):
            self._WCS_ = None
        
    def wcs2head(self):
        header = self.header

        if(header is None):
            self.header = self.wcs.to_header()
            header = self.header
        else:
            hdr = self.wcs.to_header()
    
            # Make sure that keys that might
            # conflict with the WCS pointing info are deleted
            for i in range(hdr['WCSAXES']):
                if f"CDELT{i+1}" in header:
                    del header[f"CDELT{i+1}"]
                if f"CRPIX{i+1}" in header:
                    del header[f"CRPIX{i+1}"]
                if f"CRVAL{i+1}" in header:
                    del header[f"CRVAL{i+1}"]
                for j in range(hdr['WCSAXES']):
                    if f"CD{i+1}_{j+1}" in header:
                        del header[f"CD{i+1}_{j+1}"]
                    if f"PC{i+1}_{j+1}" in header:
                        del header[f"PC{i+1}_{j+1}"]
            if "CROTA2" in header:
                del header["CROTA2"]
                
            # TODO: Find nonlinear terms and remove them also
    
            # Copy keys from WCS header to the main FITS header
            for ky in hdr.keys():
                header.set(ky,hdr[ky])
                
        # Copy NAXIS limitsfrom WCS into the header - this is necessary if 
        # reconstituting a header entirely from the WCS object.
        if not 'NAXIS' in header:
            header['NAXIS'] = self.wcs.naxis
        
        if(self.wcs.pixel_shape is not None):
            for i in range(self.wcs.naxis):
                axis = f'NAXIS{i+1}'
                # Note: wcs pixel_shape is in axis order, not array index order
                header[axis] = self.wcs.pixel_shape[i]
    
    def __getitem__(self,index):
        '''
        __getitem__ - fake list/tuple and dict activity for DataWrappers
        '''
        if index==0 or index=='data':
            return self.data
        elif index==1 or index=='header':
            return self.header
        elif index==2 or index=='wcs':
            return self.wcs
        elif index==3 or index=='WCS':
            return self.WCS
        else:
            raise IndexError(index)
            
def FITSHeaderFromDict(HeaderDict):
    '''
    FITSHeaderFromDict - internal routine to work around astropy bug
    
    This internal routine converts a dict (containing a valid FITS header)
    into an astropy.io.fits.header.Header.  The problem to be worked around
    is that, although three types of card ('', 'HISTORY', and 'COMMENT') can be 
    multiline cards and common practice (and even the astropy.io.fits.header
    dict export method) implements them as either multiline strings or lists
    of strings, the astropy.io.fits.header.Header constructor barfs when it
    receives one of those.
    
    Since all-and-only those three types of field are allowed to be multi-
    card fields by the FITS standard (as of 2020), we remove those fields
    if present, then construct a Header, then re-add them after.

    
    Parameters
    ----------
    HeaderDict : a dict-like object
        This should contain the FITS header to be astropyified.

    Returns
    -------
    An astropy.io.fits.header.Header object constructed from HeaderDict
    '''
    if(isinstance(HeaderDict,astropy.io.fits.header.Header)):
        return HeaderDict
    
    if(  ('' in HeaderDict) or 
         ('HISTORY' in HeaderDict) or 
         ('COMMENT' in HeaderDict)
         ):
        h = copy.copy(HeaderDict)
        holder = {}
        for field in ['','HISTORY','COMMENT']:
            if(field in h):
                holder[field]=h[field]
                del h[field]
        
        header = ap.io.fits.header.Header(h)
        
        for key in holder.keys():
            field = holder[key]

            if isinstance(field,str):
                field = field.split("\n")

            if isinstance(field,
                          ( list,
                            astropy.io.fits.header._HeaderCommentaryCards                               
                           )
                          ):
                for line in field:
                    header.set(key,line)
            else:
                header.set(key,field)
        
        return header
    
    else:
        
        return ap.io.fits.header.Header(HeaderDict)
          
    
    

            
#######################################################################   
#######################################################################
#######################################################################
#    
# Core subclasses of Transform
#   - Identity     - demo and/or test class
#   - Inverse      - inverse of an arbitrary Transform
#   - Composition  - composition of two or more transformst
#
#   - Wrap         - shorthand for ( W o T o W^-1 )
#   - PlusOne_     - test class (non idempotent)
#   - ArrayIndex   - reverse the order of vectors from (x,y,...) to (...,y,x)
#
#   - WCS          - pixel-to-scientific for WCS-associated arrays
    
class Identity(Transform):
    '''
    transform.Identity -- identity transform
    
    Identity() produces a Transform that does nothing -- not very interesting.
    But it is a template for other types of Transform.  The constructor 
    accepts an arbitrary set of keyword arguments, to demonstrate stashing 
    arguments in an internal parameters field - but the parameters are not 
    used for anything.
    
    Identity() also demonstrates explicit idempotence:  its inverse() mthod
    returns the original transform.
    
    Parameters
    ----------
    **kwargs : arbitrary
        This is an optional list of keyword args that get stashed in the object,
        if specified.
    '''
    
    def __init__(self,**kwargs):
        '''
        Identity constructor - demonstrate accepting and stashing args
        
            
        '''
        self.idim = 0
        self.odim = 0
        self.no_forward = 0
        self.no_reverse = 0
        self.iunit = ""
        self.ounit = ""
        self.itype = ""
        self.otype = ""
        self.params = kwargs
    
    def _forward(self, data):
        return(data)
    
    def _reverse(self, data):
        return(data)
    
    def __str__(self):
        # This demonstrates generating an arbitrary string that gets passed
        # up to the superclass.  The base string is "Identity" but if 
        # some keyword parameters were passed in we report how many there were.
        # End by handing the string up to the superclass via "self._strtmp".
        s = "Identity"
        if(len(self.params)): 
            plural = "" if(len(self.params)==1) else "s"
            s = s + f" ({len(self.params)} param{plural})"
        self._strtmp = s
        return super().__str__()
    
    # Identity is idempotent so override the inverse method
    def inverse(self):
        return(self)
    
        
class PlusOne_(Transform):
    '''
    PlusOne_ -- non-configurable non-idempotent transform for testing
    
    PlusOne_ is a 1-d transform that just adds one to its argument.
    It's useful for testing because it is non-idempotent (its inverse
    is different from itself, unlike Identity)
    '''
    def __init__(self):
        self.idim=1
        self.odim=1
        self.no_forward = False
        self.no_reverse = False
        self.iunit = None
        self.ounit = None
        self.itype = None
        self.otype = None
        self.params = {}
    
    def _forward(self,data):
        data0 = data
        data0[...,0] += 1
        return data0
    
    def _reverse(self,data):
        data0 = data
        data0[...,0] -= 1
        return data0
        
    def __str__(self):
        self._strtmp="_PlusOne"
        return super().__str__()
    
    
class Inverse(Transform):
    '''
    transform.Inverse -- invert a Transform
    
    Inverse is a wrapper object that implements the inverse operation.
    It wraps around the supplied Transform and reverses the direction of
    execution.  Although Inverse can be used to generate a functional inverse
    of any Transform, it is deprecated for that use.  You should use the 
    inverse() method instead, which calls the Inverse constructor only 
    when necessary.
    
    Parameters
    ----------
    t : Transform
        This is the Transform to invert.
    
    '''
    def __init__(self,t):
        '''
        Inverse constructor - invert a Transform
        Parameters
        ----------
        t : Transform object
            The Transform to invert
        Returns
        -------
        Transform
            The 
        
        '''
        self.idim       = t.odim
        self.odim       = t.idim
        self.no_forward = t.no_reverse
        self.no_reverse = t.no_forward
        self.iunit      = t.ounit
        self.ounit      = t.iunit
        self.itype      = t.otype
        self.otype      = t.itype
        self.params     = {'transform':copy.copy(t)}
    
    def _forward(self, data):
        return(self.params['transform'].apply(data,invert=True))
    
    def _reverse(self, data):
        return(self.params['transform'].apply(data,invert=False))
    
    def __str__(self):
        self.params['transform']._str_not_top_tmp = True
        s = self.params['transform'].__str__()
        self._strtmp = "Inverse " + s
        return(super().__str__())
    
    def inverse(self):
        return(self.params['transform'])
    


class Composition(Transform):
    '''
    transform.Composition -- compose a list of one or more Transforms 
    
    Composition implements composite transforms.  It stores a copy of each of
    the supplied Transforms in an internal list, and when applied it invokes 
    them in sequence as a compound operation. 
    
    Parameters
    ----------
    translist : List of Transform(s)
        The args are a collection of Transforms to compose.  They are composed
        in mathematical order (i.e. the last one in the arglist gets applied
        first to data in an apply() operation).
    '''
    
    def __init__(self, translist):
        if(not isinstance(translist,list)):
            raise ValueError("Composition requires a list of Transforms")
    
        if(len(translist)<1):
            raise ValueError("Composition requires at least one Transform")
            
        complist = []
        
        ### Copy the args into the composelist, unwrapping compositions if
        ### we find them.
        idim = 0
        odim = 0
        for trans in translist:
            if(not(isinstance(trans,Transform))):
                raise AssertionError("transform.Composition: got something that's not a Transform")
            ### Track input and output dimensions- keep first nonzero dim
            ### in the list (in either direction), or use zero if there
            ### aren't any.
            if( idim==0 ):
                idim = trans.idim
            if( trans.odim != 0 ):
                odim = trans.odim
            if( isinstance(trans,Composition)):
                complist.extend(copy.copy(trans.params['list']))
            else:
                complist.append(copy.copy(trans))
        
        
        
        self.idim       = idim
        self.odim       = odim
        self.no_forward = any(map( (lambda arg: arg.no_forward), complist))
        self.no_reverse = any(map( (lambda arg: arg.no_reverse), complist))
        self.iunit      = complist[-1].iunit
        self.ounit      = complist[0].ounit
        self.itype      = complist[-1].itype
        self.otype      = complist[0].otype
        self.params     = {'list':complist}
        
    def _forward(self,data):
        for xf in self.params['list'][::-1]:
            data = xf.apply(data)
        return data
    
    def _reverse(self, data):
        for xf in self.params['list']:
            data = xf.invert(data)
        return data
    
    def __str__(self):
        #######
        # Assemble shortened versions of all the member transforms' 
        # strings, then merge them using ASCII 'o' to indicate
        # function composition
        strings = []
        for xf in self.params['list']:
            xf._str_not_top_tmp = True
            strings.append(xf.__str__())
    
        self._strtmp = '( (' + ') o ('.join( strings ) + ') )'
        return (super().__str__())
    
    
    
class Wrap(Composition):
    '''
    transform.Wrap -- wrap a Transform around another one
    
    Wrap generates transforms of the form:
        
        W^-1 o T o W
    
    where T is a Transform and W is a wrapper Transform. This is
    a common construct that permits T to work with a more convenient
    representation of the data.  Note that W is executed first and 
    W^-1 is executed last.
    
    Parameters
    ----------
    
    *Main - Transform
        The Transform to be wrapped
    
    *Wrapper - Transform
        The wrapping transform
    '''
    
    def __init__(self, T, W):
        super().__init__([W.inverse(), T, W])
    
    ## No __str__ for Wrap since the Composition stringifier works just fine
    


class ArrayIndex(Transform):
    '''
    transform.Arrayindex -- convert a vector to a NumPy array index
    
    This just reverses the order of the components of the vector, so that
    they can be used to index an array with NumPy's wonky (...,Y,X) indexing.
    Input and output dimensions are irrelevant -- the entire vector is always
    reversed.
    
    ArrayIndex is idempotent.
    '''
    def __init__(self):
        self.idim = 0
        self.odim = 0
        self.no_forward = False
        self.no_reverse = False
        self.iunit = None
        self.ounit = None
        self.itype = None
        self.otype = None
        self.params = {}
        
    def _forward(self,data):
        return(data[::-1])
    
    def _reverse(self,data):
        return(data[::-1])
    
    def inverse(self):
        return self
    
    def __str__(self):
        self._strtmp = "ArrayIndex"
        return (super().__str__())
   
           
def _WCSTrans_(dingus):
    '''
    _WCSTrans_ - just a top-level trampoline.  This is needed for DataWrapper:
    one of its methods generates a transform.WCS object but suffers a namespace
    collision in that scope.  
    '''
    return WCS(dingus)

class WCS(Transform):
    '''
    transform.WCS - World Coordinate System translation
    
    WCS Transforms implement the World Coordinate System that is used in 
    the FITS image standard that's popular among scientists.  (WCS: Greisen & 
    Calabretta 2002; "http://arxiv.org/abs/astro-ph/0207407") WCS includes 
    both linear and nonlinear components; both are implemented, via the 
    astropy.wcs library.
    
    WCS Transforms convert vectors in standard (X,Y) image pixel 
    coordinates (in which (0,0) is the center of the pixel at lower left of 
    the image, X runs right, and Y runs up), to world coordinates using the
    WCS information embedded in a FITS header. The inverse does the inverse.
    
    Parameters
    ----------
    
    object: a FITS header or astropy.wcs.WCS object, or an object having a 
    FITS header or astropy.wcs.WCS object as a "header" or "wcs" attribute, 
    respectively
    
    /dim: an optional limiting dimension
    '''
    def __init__(self, dingus):

        # Construct a WCS object -- that's what does the real work.
        # Very important to detect and not wrap DataWrappers, since the
        # DataWrapper constructor calls this.
        if isinstance(dingus,DataWrapper):
            if(not isinstance(dingus.wcs,ap.wcs.wcs.WCS)):
                dingus.head2wcs()
            wcs_obj = dingus.wcs        
        elif isinstance(dingus,ap.wcs.wcs.WCS):
            wcs_obj = dingus
        else:
            dingus = DataWrapper(dingus)
            wcs_obj = dingus.wcs

        # Generate input axis names and units - these are standard
        # image axis names and "Pixels", since we're mapping from the
        # pixel grid to the WCS system.  First three axes (0, 1, and 2)
        # are called "X", "Y", and "Z".  Later axes are called "Coord <foo>"
        # where <foo> is the index number starting at 3.
        inames = ['X','Y','Z']
        itype = inames[ 0 : (  min( len(inames), wcs_obj.wcs.naxis )  ) ]
        while len(itype) < wcs_obj.wcs.naxis:
            itype.append(f'Coord {len(itype)}')
        iunit = ['Pixels'] * wcs_obj.wcs.naxis
        
        # Populate the object
        # The WCS fields come in with exotic object types, so hammer
        # them into numbers (idim/odim) or lists of strings (ounit/otype).
        self.idim = wcs_obj.wcs.naxis + 0
        self.odim = wcs_obj.wcs.naxis + 0
        self.iunit = iunit
        self.ounit = [ f"{a}" for a in wcs_obj.wcs.cunit ]
        self.itype = itype
        self.otype = [ f"{a}" for a in wcs_obj.wcs.ctype ]
        self.no_forward = False
        self.no_reverse = False
        self.params = {
            'wcs': wcs_obj
        }
    
    def __str__(self):
        self.strtmp = "WCS"
        return super().__str__()
    
    def _forward(self, data):
        sh = data.shape
       
        if(len(sh)>2):
            data = np.reshape( data, [ np.prod(sh[:-1]),sh[-1] ], order='C' )
        elif(len(sh)==1):
            data = np.expand_dims(data,0)
            
        data = self.params['wcs'].all_pix2world( data, 0 )
        
        if(len(sh)>2 or len(sh)==1):
            data = np.reshape( data, sh, order='C' )
        
        return(data)
    
    def _reverse(self, data):
        sh = data.shape
        
        if(len(sh)>2):
            data = np.reshape( data, [ np.prod(sh[:-1]),sh[-1] ], order='C' )
        elif(len(sh)==1):
            data = np.expand_dims(data,0)
        
        data = self.params['wcs'].all_world2pix( data, 0 )
        
        if(len(sh)>2 or len(sh)==1):
            data = np.reshape( data, sh, order='C' )
        
        return(data)
    





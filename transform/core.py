# -*- coding: utf-8 -*-

import copy
import os.path
import numpy as np
import astropy 
import astropy.io.fits
import astropy.wcs
import astropy.units
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
        
        remap can therefore accept and manipulate various objects including 
        astropy.io.fits HDUs, dictionaries, and SunPy map objects.  
        
        The return value has the same attributes (data and either header or 
        wcs) that were passed in.  If the data object is a recognized 
        class (e.g. astropy.io.fits image HDU, or SunPy map) and can be 
        easily exported, then the same type of object is returned.  Otherwise 
        a DataWrapper object is returned.  DataWrappers contain a .data 
        attribute with the actual data, a .header attribute with a FITS header,
        and a .wcs attribute with an AstroPy WCS object.
        
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
        out_template = DataWrapper(template)
        
        # Set "shape" to be the desired shape of the output.
        # This is the passed-in shape, or the implicit shape in the
        # template, or the shape of the input array.
        if(shape is None):
            if(out_template.wcs is None or out_template.wcs.pixel_shape is None):
                if(out_template.data is None):
                    shape = data.data.shape
                else:
                    shape = out_template.data.shape
            else:
                # shape is (...,Y,X); wcs pixel_shape is (X,Y,...).
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
            # those present in odim)
            assert(len(shape) == output_range.shape[0])

            print(f"output range is {output_range}")
            # Now we have an output_range, either from a parameter or from autoscaling.
            # Generate a WCS object and stuff it into the out_template.                  
            otwcs = astropy.wcs.WCS(naxis=len(shape))
            otwcs.wcs.crpix = [ shape[i]/2 + 0.5 for i in range(len(shape)) ]
            otwcs.wcs.crval = (output_range[...,0]+output_range[...,1])*0.5
            otwcs.wcs.cdelt = [ (output_range[i,1]-output_range[i,0])/(shape[i]) for i in range(len(shape)) ]

            # TODO: Need to sort out types and units here   
            # This commented-out code (3 lines) is not right for broadcasting.
            # Might need to add some sort of method to individual transforms 
            # for units - or if that's too hard, just wing it.
            # Doing nothing (as we do now) is probably wrong but worked for 
            # 15 years for the legacy PDL::Transform code so it's not all *that*
            # wrong.
            #if(self.odim > 0):
            #    otwcs.wcs.ctype = copy.copy(self.otype)
            #    otwcs.wcs.cunit = copy.copy(self.ounit)
                
            otwcs.pixel_shape = tuple(shape)
            out_template.wcs = otwcs
            out_template.wcs2head()            

            
        ## Now we have an out_template -- either via autoscaling or 
        ## via parameter.
        
        out_template.data = None
        output_trans = WCS(out_template).inverse()
    
        ## Finally ... dispatch the actual resampling
        total_trans = Composition([output_trans, self, input_trans])
        data_resampled = total_trans.resample(data.data, method=method, bound=bound, phot=phot, shape=shape)
        output = DataWrapper(
            data_resampled,
            template = out_template
        )
        # Copy the original source_object in, so we export to the same style.
        output.source_object = data.source_object
        return output.export()
    

###################################
###################################
###
### DataWrapper - wrapper class to contain array data with a FITS header and/or 
### WCS object.  Used by WCS Transform and by remap().  It is NOT a subclass of
### Transform, because it's not a coordinate transformation.

class DataWrapper():
    '''
    _DataheaderWrapper - class to manage data with headers
    
    This wrapper class encapsulates a variety of objects with 
    both a data fork and a WCS info fork that could be a FITS header
    or a WCS transformation.  
    
    The constructor accepts an arbitrary object and checks it to see
    if it:
            - is an astropy WCS or Header object
            - has a .wcs, .header, and/or .data attribute
            - is a dictionary containing wcs, header, and/or data fields
            - is a dictionary containing a FITS hedader
            - is a list or tuple, the 0 element of which has a .data, .header, and/or .wcs attribute
            - is a list or tuple, the 0 element of which is a NumPy array and the 1 element of which contains a FITS header
    
    All of these get encapsulated in an object with attributes: .data, .header, 
    and .wcs where .data is a numpy array or None, .header is an astropy.io.fits header
    object, and .wcs is an astropy.wcs.WCS object.  The following methods are
    supplied:
            - head2wcs - update the WCS object using information in the header
            - wcs2head - update the FITS header using infromation in the WCS object
            - export   - export the info as the original object (see below)
            
    The 'export' method stuffs the .data, .header, and .wcs attributes (as
    appropriate) into the original object type passed in, in the event that the
    original object could support them.  If the original object was a dictionary,
    a copy of that dictionary gets passed back with the data, header, and wcs
    fields in it.  If the original object was a numpy array or FITS header,
    the _DataWrapper object itself it returned (with data, header, and wcs 
    attributes intact).
    '''
    def __init__(self, SourceObject, /, template=None):
        InputsToSearch = [template, SourceObject]
        data = None
        header = None
        wcs = None
        
        for this in InputsToSearch:

            # Make sure the argument exists
            if this is None:
                continue
            
            if isinstance(this, DataWrapper):
                if( data is None   and hasattr(this,'data')   ):
                    data = this.data
                if( header is None  and  wcs is None ):
                    if (hasattr(this,'header')):
                        header = this.header
                    elif( hasattr(this,'wcs')):
                        wcs = this.wcs
                continue
            
            # If the item is a string, treat it as a filename of a FITS file
            if isinstance(this,str):
                this = astropy.io.fits.open(this)
                
            # HDULists come from FITS files -- we take the first HDU.
            if isinstance(this,astropy.io.fits.hdu.hdulist.HDUList):
                this = this[0]
            
            # Look for the obvious attributes
            if wcs is None and isinstance(this, ap.wcs.WCS):
                wcs = this
                continue
            
            if wcs is None and hasattr(this,'wcs'):
                if isinstance(this.wcs, ap.wcs.WCS):
                    wcs = this.wcs
                else:
                    raise ValueError("DataWrapper: 'wcs' attribute must be an Astropy WCS object")
            
            if   ( header is None and 
                  hasattr(this,'header') and 
                  this.header is not None ):
                try:
                    if( ('SIMPLE' in this.header and 'NAXIS' in this.header) or ('WCSAXES' in this.header) ):
                        header = this.header
                    else:
                        raise ValueError("DataWrapper: 'header' attribute must have a FITS header in it")
                except AttributeError:
                    raise ValueError("DataWrapper: 'header' attribute must be dict-like")
                
            if data is None:
                if isinstance(this,np.ndarray):
                    data = this
                elif hasattr(this,'data'):
                    if isinstance(this.data, np.ndarray):
                        data = this.data
                    elif this.data is not None:
                        raise ValueError(f"DataWrapper: 'data' attribute must be a Numpy array (got a {this.data.__class__})")
            
            # look for attributes as keys in a dictionary instead
            if isinstance(this,dict):
                
                if (wcs is None) and ('wcs' in this):
                    if (isinstance(this['wcs'],ap.wcs.WCS)):
                        wcs = this['wcs']
                    elif(this['wcs'] is not None):
                        raise ValueError("DataWrapper: 'wcs' dict entry must be an Astropy WCS object")
                    
                if (header is None) and ('header' in this):
                    try:
                        if(  ('SIMPLE' in this['header'] and 'NAXIS' in this['header']) or ('WCSAXES' in this['header']) ):
                            header = this['header']
                        elif(this['header'] is not None): 
                            raise ValueError("DataWrapper: 'header' dict entry must have a FITS header in it.")
                    except AttributeError:
                        raise ValueError("DataWrapper: 'header' dict entry must be dict-like.")
                    
                if (data is None) and 'data' in this:
                    if isinstance(this['data'],np.ndarray):
                        data = this['data']
                    else:
                        raise ValueError("DataWrapper: 'data' dict entry must be a NumPy array")
                        
                # Maybe this dict it itself a FITS header?  If so, use it.
                if (header is None) and isinstance(this,dict) and ('NAXIS' in this) and ('SIMPLE' in this):
                    header = this
        
        
        if( wcs is None ):
            if( header is not None ):
                # Work around bug in astropy.io.fits with commentary cards:
                # make a copy of the header, delete possibly-offending cards,
                # and make a WCS out of *that*.
                h = copy.copy(header)
                if( 'HISTORY' in h ):
                    del h['HISTORY']
                if( 'COMMENT' in h ):
                    del h['COMMENT']
                wcs = ap.wcs.WCS(h)    
        
        self.wcs = wcs
        self.header = header
        self.data = data
        
        # Stash the source object into this object, for reconstitution later.
        # FITS files yield a list of HDUs; we want the first one only.
        if(isinstance(SourceObject, astropy.io.fits.hdu.hdulist.HDUList)):
            SourceObject = SourceObject[0]
        self.source_object = SourceObject
    
    def head2wcs(self):
        self.wcs = astropy.wcs.WCS(self.header)
            
    def wcs2head(self):
        if(self.header is None):
            self.header = {}
        hdr = self.wcs.to_header()
        for ky in hdr.keys():
            self.header[ky]=hdr[ky]
        
            
    def export(self):
        '''
        DataWrapper.export() -- return data in original form if possible
        
        Export returns wrapped-up data in its original form -- if you 
        built the object from a dictionary, for example, it comes back as
        a dictionary.  If you built it from an object that has .data and
        .header or .wcs attributes, you get back a copy of that object with
        the current data, .header, and/or .wcs attriutes in it.
        
        If you started with a numpy array only, a header only, or a WCS
        object only, you get back the wrapper object itself (and the
        data and header can be accessed as .data and .header).
        '''
        
        # Make sure the WCS info is definitive in case the user wants the 
        # FITS header
        if(self.wcs is not None):
            self.wcs2head()
        
        if(self.source_object is None):
            return self
        elif( isinstance(self.source_object,astropy.io.fits.hdu.image.PrimaryHDU) ):
            Out = copy.copy(self.source_object)
            Out.data = self.data
            for key in self.header.keys():
                # Work around problematic WCS implementation that preserves but breaks
                # commentary cardsets; and skip copying CD or PC matrices (or CDELT
                # fields)
                if(key=='HISTORY' or key=='COMMENT' or key=='' or
                   key[0:1]=='CD' or key[0:1]=='PC'
                   ):
                    continue
                Out.header.set(key, self.header[key])
            return Out
        elif( isinstance(self.source_object,dict) ):
            Out = { 'data':self.data, 'header':self.header, 'wcs':self.wcs }
            return Out
        ## TODO: Recognize other object types here and return them if possible
        ## (e.g. HDU; SunPy map)
   
        return self
 
            
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
    
    
    # Non-idempotent test subclass
    
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
        dw = DataWrapper(dingus)
        wcs_obj = dw.wcs
        if(wcs_obj is None):
            dw.head2wcs()
            wcs_obj = dw.wcs
            if(wcs_obj is None):
                raise(ValueError,"Transform::WCS - No WCS object found")
        
        # Test to make sure the object works.
        test_coord = np.zeros([1,wcs_obj.wcs.naxis])
        try:
            wcs_obj.wcs_world2pix(test_coord,0)
            self.no_forward = 0
        except: self.no_forward = 1
        
        try:
            wcs_obj.wcs_pix2world(test_coord,0)
            self.no_reverse = 0
        except: self.no_reverse = 1
        
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
    
    def __str__(self):
        self._strtmp = "WCS pixel-to-science"
        return super().__str__()
    





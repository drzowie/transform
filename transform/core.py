# -*- coding: utf-8 -*-

import copy
import numpy as np
import astropy

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
                
        - FITS: transformation representing the conversion from standard pixel 
            coordinates to world coordinates using the World Coordinate System
            specification by Greisen & Calabretta -- this is typically used in
            FITS-format images to map array pixel coordinates to/from 
            real-world scientific coordinates
            
        - Polar: Convert Cartesian to linear or conformal (logarithmic) polar 
            coordinates
          
        - Projective: the family of projective transformations in 2-D; these are 
            first-order nonlinear projections with some nice properties, and
            are frequently used in perspective correction or rendering.
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
          
 
    def map(self, data, /, 
            method='s',
            bound='none',
            phot='radiance',
            template=None
            ):
        '''
        map - use a transform to remap a pixel array
        
        This method implements resampling of gridded data by applying the 
        Transform to the implicit coordinate system of the sampled data,
        and resampling the data to a new pixel grid matching the transformed
        coordinates. The output data have their size determined by a supplied
        template, or matched to the input array if no template is supplied.
        
        The method works by using the inverse Transform to map *from* the 
        output space back *to* the input space. The output data samples
        are then interpolated from the locations in the input space.
        
        Parameters
        ----------
        
        data : ndarray
            This is the gridded data to resample, such as an image.  It must 
            have at least as many dimensions as the idim of the Transform 
            (self).
            
        /method : string (default 'sample')
            This string indicates the interpolation method to use.  Only
            the first character is checked.  Possible values are:
                
                'sample' - use the value of the nearest-neighbor pixel in
                    the input space.  Very fast, but produces aliasing.
                    
                'linear' - use <N>-linear interpolation. Marginaly bettter than
                    sampling, but still produces phase and amplitude aliasing
                
                'cubic' - use <N>-cubic interpolation. This produces a smoother
                    output than linear for enlargements
                
                'fourier' - use discrete Fourier coefficients. to interpolate
                    between points.  This is useful for periodic data.
                    
                'gaussian' - use locally optimized Jacobian-driven filtering,
                    with a Gaussian filter profile
                    
                'hanning' - use locally optimized Jacobian-drivn filtering,
                    with a Hanning window profile
            
            Most The first four interpolation methods use the supplied "interpND"
            general purpose interpolator and are subject to aliasing and other
            effects outlined in a paper by DeForest (2004; Solar Physics 219, 3).
            The last two use the numerical Jacobian derivative matrix (local 
            linearization) of the coordinate transform to produce a variable, 
            optimized filter function to reduce or aliminate aliasing.  Gaussian
            sampling uses a Gaussian filter function with nice Fourier properties;
            Hanning resampling sampling uses a Hanning-like filter function that
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
        
        /template : list or tuple or HDU or FITS-header or None (default None)
            If present, this is the shape of the output data grid.  It must 
            have dimensions that agree with the odim of the transform.  If the
            input grid has the same dimensionality as the imput dimension of 
            the Transform, the output must match the output dimensiln.  If the
            input grid dimensionality is higher, then the output grid must
            exceed the output dimension of the Transform by the same amount.
        
        Returns
        -------
        
        The resampled data
        '''
        ##### Make sure we separate the data fork of we get an ImageHDU
        if( isinstance( data, np.ndarray ) ):
            data0 = data
        elif( isinstance( data, astropy.ImageHDU) ):
            data0 = data.data
        else:
            raise ValueError('Transform.map requires a numpy array or an astropy ImageHDU')
            
        methodChar = method[0]
        
        ##### Set the output array size
        if( template is None ):
            template = data0.shape
            
        raise AssertionError("map: not implemented yet")
        
        ##### Check input, output, and Transform dimensions all agree.
        ##### Okay to pass in *more* dimensions (and let them get broadcast).
        ##### Not okay to pass in *fewer* dimensions.
        if( len(template) < self.odim ):
            raise ValueError('map: Transform odim must match output data shape')
        if( len(data0.shape) < self.idim ):
            raise ValueError('map: Transform idim must match input data shape')
        if( len(data0.shape) - self.idim  != len(template) - self.odim ):
            raise ValueError('map: template and source dimensions must match')
        
        # Enumerate every pixel ( coords[...,Y,X,:] gets [X,Y,...] )
        coords = np.mgrid[list( map( lambda i:range(i), iter(template)))].transpose()
        
        # Transform back to the input grid
        icoords = self.invert(coords)
        
        # Figure the interpolation.
        if(methodChar == 's'):
            pass
        
        assert("map: still needs interpolators!")
        
        
        
            
            
            
    
#######################################################################
#######################################################################
#    
# Basic subclasses for Identity, Inverse, and Composition
#
    
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
        data[...,0] = data[...,0]+1
    
    def _reverse(self,data):
        data[...,0] = data[...,0]-1
        
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
            print (f"trans is {trans}")
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
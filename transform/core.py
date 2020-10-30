# -*- coding: utf-8 -*-

import copy

class Transform:
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
        Apply - apply a matheatical transform to a set of N-vectors
        
        Parameters
        ----------
        data : ndarray
          This is the data to which the Transform should be applied.
          The data should be a NumPy ndarray, with the 0 dim running across
          vector dimension.  Subsequent dimensions are broadcast (e.g., 
          a 2xWxH NumPy array is treated as a WxH array of 2-vectors).
          The 0 dim must have sufficient size for the transform to work.
          If it is larger, then subsequent vector dimensions are ignored , so 
          that (for example) a 2-D Transform can be applied to a 3xWxH NumPy 
          array and the final WxH plane is transmitted unchanged.  
          
        invert : Boolean, optional
          This is an optional flag indicating that the inverse of the transform
          is to be applied, rather than the transform itself.  

        Raises
        ------
        AssertionError
          This gets raised if the Transform won't work in the intended direction.
          That can happen because some mathematical transforms don't have inverses;
          Transform objects contain a flag indicating validity of forward and 
          reverse application.
          
        Returns
        -------
        ndarray
            The transformed vector data are returned as a NumPy ndarray.  Most 
            Transforms maintain the dimensionality of the source vectors.  Some 
            embed (increase dimensionality of the vectors) or project (decrease
            dimensionality of the vectors); additional input dimensions, if 
            present, are appended to the output unchanged in either case.

        '''
        if(invert):    
            if(self.no_reverse):
                raise AssertionError("This Transform ({self.__str__()}) is invalid in the reverse direction")
            return self._reverse(data)
            
            if(self.no_forward):
                raise AssertionError("This Transform ({self.__str__()}) is invalid in the forward direction")
            return self._forward(data)
        
        else:
            return self._reverse(data)
        
        
        
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
        constructing a transform.Inverse of it.

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
            The Transform._forward method always throws an error.  Subclasses
            should overload the operator and carry out the actual math there.

        Returns
        -------
        None
            Subclassed _forward methods should return the manipulated ndarray.
        '''             
        raise AssertionError(\
            "Transform._forward should always be overloaded."\
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
            The Transform._reverse method always throws an error.  Subclasses
            should overload the operator and carry out the actual math there.

        Returns
        -------
        None
            Subclassed _reverse methods should return the manipulated ndarray.
        '''
        raise AssertionError(\
            "Transform._reverse should always be overridden."\
            )
          
 
#######################################################################
#######################################################################
#    
# Basic subclasses for Identity, Inverse, and Composition
#
    
class Identity(Transform):
    '''
    transform.Identity -- identity transform
    
    Identity produces a Transform that does nothing -- not very interesting.
    But it is a template for other types of Transform.  The constructor 
    accepts an arbitrary set of keyword arguments, to demonstrate stashing 
    arguments in an internal parameters field - but the parameters are not 
    used for anything.
    
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
        self.idim = 1
        self.odim = 1
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
        self.params     = {'transform':copy.deepcopy(t)}
    
    def _forward(self,data):
        return(self.params['transform'].apply(data,invert=True))
    
    def _reverse(self,data):
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
    the supplied Transforms in an internal list, and invokes them appropriately
    to generate a compound operation. 
    
    Parameters
    ----------
    *args : Transform(s)
        The args are a collection of Transforms to compose.  They are composed
        in mathematical order (i.e. the last one in the arglist gets applied
        first to data in an apply() operation).
    '''
    
    def __init__(self,*args):
        if(len(args)<1):
            raise AssertionError("Composition requires at least one Transform")
            
        for xf in args:
            if(not(isinstance(xf,Transform))):
                raise AssertionError("transform.Composition: expected a collection of Transforms")
            
        self.idim       = args[-1].idim
        self.odim       = args[0].odim
        self.no_forward = any(map( (lambda arg: arg.no_forward), args))
        self.no_reverse = any(map( (lambda arg: arg.no_reverse), args))
        self.iunit      = args[-1].iunit
        self.ounit      = args[0].ounit
        self.itype      = args[-1].itype
        self.otype      = args[0].otype
        self.params     = {'list':copy.deepcopy(args)}
        
    def _forward(self,data):
        for xf in reverse(self.params['list']):
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
    
        self._strtmp = '(' + ') o ('.join( strings ) + ')'
        return (super().__str__())
    

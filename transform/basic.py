#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transform subclasses for basic coordinate transforms
"""
import numpy as np
import math as math
from .core import Transform
from astropy import units


class Linear(Transform):
    '''
    transform.Linear - linear transforms
    
    Mathematical linear transformations consist of an offset and a matrix 
    operation.  Linear implements that class directly with a matrix 
    multiplication and an offset.  Two separate offsets (pre and post multiply)
    are tracked, for convenience. Pre offsets work in the input (pre-matrix)
    coordinate system; post offsets work in the output (post-matrix) system.
    The formula is:
    
        data_out = (post) + (matrix x (data + pre))
        
    where post, pre, data, and data_out are all column vectors (or broadcast
    arrays of column vectors), and matrix is an NxM matrix.  The Transform
    then accepts N-vectors and returns M-vectors.
    
    The parameters are pre, post, and matrix; they are all optional.  The
    input and output dimensionality of the transform are calculated at 
    construction time from whichever parameters you supply -- offsets and/or
    matrix.  They must agree dimensionally (e.g., matrix column count must
    match pre-offset vector size).
    
    The inverse transform is valid if and only if the matrix is invertible. 
    
    As with other Transforms, additional dimensions in the input data vectors 
    are ignored -- so applying a 2-D Linear transform to data comprising 
    3-vectors results in the first two components being transformed, and the 
    third component being passed through unchanged.
    
    Notes
    -----
    
    Several subclasses enable simpler specification of common linear 
    transformations such as scaling, rotation, and simple offsets.
    
    Linear transformation is overspecified by including both pre and post,
    but in practice it's convenient to be able to work in either the matrix-
    input coordinates or matrix-output coordinates when specifying an offset.
    
    
    Parameters
    ----------
    *pre : numpy.ndarray (optional; default = 0)
        Optional; if present must be a vector.  Offset vector to be added
        to the input data before hitting with the matrix
        
    *post : numpy.ndarray (optional; default = 0)
        Optional; if present must be a vector.  Offset vector to be added
        to the return data after hitting with the matrix
        
    *matrix : numpy.ndarray (optional; default = identity)
        Optional; if present must be a 2-D array.  As always in numpy, the
        matrix is addressed in (row,column) format -- so if you use 
        nested-brackets notation for np.array(), then the innermost vectors
        are rows of the matrix.
    '''
    
    def __init__(self, *,                      
                 pre    = None, post   = None, 
                 matrix = None,                
                 iunit  = None, ounit  = None, 
                 itype  = None, otype  = None  
                ):
        
        idim = None
        odim = None
        
     
        ### Check if we got a matrix.  if we did, make sure it's 2-D and use
        ### it to set the idim and odim.
        if( matrix is not None ):   
            if( isinstance( matrix, np.ndarray ) ):
                if( len( matrix.shape ) == 2 ):
                    odim,idim = matrix.shape
                else:
                    raise ValueError("Linear: matrix must be 2D")
            else:
                raise ValueError("Linear: matrix must be a 2D numpy.ndarray or None")
        
        ### Now validate the pre and post, if present
        idim = self._parse_prepost( idim, pre,  'Linear', 'Pre-offset',  'idim' )
        odim = self._parse_prepost( odim, post, 'Linear', 'Post-offset', 'odim' )
                
        if( matrix is not None ):
            if( odim is None ):
                odim = idim
            elif( idim is None ):
                idim = odim
            elif( idim != odim ):
                raise ValueError("Linear: idim and odim must match if no matrix is supplied")
      
        ### Finally - if no idim and odim, default to 2
        if( idim is None ):
            idim = 2
        if( odim is None ):
            odim = 2

   
        ### Now check for inverses.
        ### linalg.inv will throw an exception if the matrix isn't square or
        ### can't be inverted.
        matinv = None
        if( np.all( matrix != None ) ):
            try:
                matinv = np.linalg.inv(matrix)
            except:
                pass
            
        self.idim       = idim             
        self.odim       = odim
        self.no_forward = False
        self.no_reverse = (np.all(matinv == None)  and  np.all(matrix != None))
        self.iunit      = iunit
        self.ounit      = ounit
        self.itype      = itype
        self.otype      = otype
        self.params = {
            'pre'    :  pre,
            'post'   :  post,
            'matrix' :  matrix,
            'matinv' :  matinv,
            }
            
    def __str__(self):
        if(not hasattr(self,'_strtmp')):
            self._strtmp = 'Linear'
        return super().__str__()
    
    def _forward( self, data: np.ndarray ):

        ## Handle pre-offset
        if( self.params['pre'] is not None ):
            data = data + self.params['pre']
        
        ## Handle matrix
        m = self.params['matrix']
        if( m is not None ):
            data = np.expand_dims(data,-1)  # convert vectors to Mx1
            data = np.matmul( m, data ) 
            data = data[...,:,0]            # convert back to vectors
        
        ## Handle post-offset
        if( self.params['post'] is not None ):
            data = data + self.params['post']
        
        return( data )

    def _reverse( self, data: np.ndarray ):           
        ## Handle reversing the post-offset
        if( self.params['post'] is not None ):
            data = data - self.params['post']
        
        ## Handle inverse-matrix
        m = self.params['matinv']
        if( m is not None ):
            data = np.expand_dims(data,-1)
            data = np.matmul( m, data )
            data = data[...,:,0]
            
        ## Handle reversing the pre-offset
        if( self.params['pre'] is not None ):
            data = data - self.params['pre']
       
        return( data )
        


    def _parse_prepost( self, dimspec, vec, objname, prepostname, dimname ) :
        '''
        _parse_prepost - internal transform.Linear method to parse a pre-offset 
        or post-offset vector's dims during object construction.  This is 
        used by most of the Linear subclasses as well as by Linear itself.
        
        Parameters
        ----------
        dimspec : int or None
            DESCRIPTION.
        vec : np.ndarray or None
            DESCRIPTION.
        objname : string
            the name of the object doing the parsing (for exceptions)
        prepostname : string
            'pre' or post' - vector name
        
        Returns
        -------
        the dimensionality from the dimspec or vec size.
        
        '''
        if( dimspec is not None and not isinstance(dimspec, np.ndarray) ):
            dimspec = np.array(dimspec)
            
        if( dimspec is not None and dimspec.size>1 ):
            raise ValueError(f"{objname}: {dimname} must be a scalar")
        
        if( vec is not None ):
            if( not (isinstance( vec, (np.ndarray,list,tuple) ))):
                raise ValueError(f"{objname}: {prepostname} must be a 1-D numpy.ndarray vector or None")
            
            if( len( np.shape(vec) ) != 1 ):
                raise ValueError(f"{objname}: {prepostname} must be a 1-D vector")
            if( (dimspec is not None)  and (dimspec != np.shape(vec)[0])):
                raise ValueError(f"{objname}: {prepostname} size must match {dimname}")
            dimspec = np.shape(vec)[0]
    
        return dimspec
    
    
##########
# Subclassses of Linear (e.g. scale, rot)
 
class Scale(Linear):
    '''
    transform.Scale - linear transform that just stretches vectors
    
    Scale transforms implement linear transforms of the form:
        
        data_out = (post) + (dmatrix x (data + pre))
    
    where post, pre, data, and data_out are column vectors, and dmatrix is a 
    diagonal square matrix.  You specify the trace of dmatrix.  
    
    As a special case, you can specify a scalar for dmatrix so long as
    you also specify a dimension some other way.  Unlike Linear itself (which
    selects its dimensionality implicitly) you can specify the dimension
    of the operator with the "d" parameter.
    
    In the special case where you specify a single scale variable and 
    you do not specify either of the offsets, the dimensionality defaults
    to 2 -- this is for simple scaling of image data.  To obtain a 1-D
    simple scale, specify dim=1 in the construction parameters.
    
    Scale is a subclass of Linear.
    
    Examples
    --------
    
       a = t.Scale( 3, 2 )            # triple the length of a vector in 2-D
       a = t.Scale( 3, dim=2 )        # same but maybe clearer
       a = t.Scale( np.array([1,3]) ) # Triple the Y component of a vector
    
    Parameters
    ----------
    
    scale : numpy.ndarray or scalar (positional or keyword)
        This is either an ndarray vector representing the trace of the 
        scale matrix, or a scalar for a uniform scale.
        
    dim : int (optional; positional or keyword)
        This is an optional dimension specifier in case you pass in a scalar
        and no other dimensionality hints.
    
    /pre : numpy.ndarray (optional; default = 0)
        Optional; if present must be a vector.  Offset vector to be added to 
        the input data before hitting with the scale matrix
        
    /post : numpy.ndarray (optional; default = 0)
        Optional; if present must be a vector.  Offset vector to be added to
        the return data after hitting with the scale matrix
    ''' 
    def __init__(self,                                     
                 /, scale: np.ndarray,  dim=None, 
                 *, post=None, pre=None,
                 iunit=None, ounit=None,
                 itype=None, otype=None
                 ):
       
        dim = self._parse_prepost( dim, pre,  'Scale', 'Pre-offset',  'dim' )
        dim = self._parse_prepost( dim, post, 'Scale', 'Post-offset', 'dim' )
       
        # Make sure these are NumPy arrays
        if( pre is not None ):
            pre =  pre + np.array(0)
        if( post is not None ):
            post = post+ np.array(0)
     
        if( not( isinstance( scale, np.ndarray ) ) ) :
            if isinstance(scale,(list,tuple)):
                scale = np.array(scale)
            else:
                scale = np.array([scale])
        if(len(scale.shape) > 1):
           raise ValueError('Scale: scale parameter must be scalar or vector')
           
        if( dim is not None ):
            if(scale.shape[0] != dim  and  scale.shape[0] != 1):
                raise ValueError('Scale d must agree with size of scale vector')
            if(scale.shape[0] == 1):
                scale = scale + np.zeros(dim) # force broadcast to correct size
        else:
            if( scale.shape[0]==1 ):
                dim = 2
                scale = scale + np.zeros(2)
            else:
                dim = scale.shape[0]
            
        m = np.zeros([dim,dim])
        for i in range(dim):
            m[i,i]=scale[i]
            
        m1 = np.zeros([dim,dim])
        if(all(scale!=0)):
            for i in range(dim):
                m1[i,i]=1.0/scale[i]
        
        self.idim       = dim
        self.odim       = dim
        self.no_forward = False
        self.no_reverse = (any(scale == 0))
        self.iunit      = iunit
        self.ounit      = ounit
        self.itype      = itype
        self.otype      = otype
        self.params = {    
            'pre': pre,    
            'post': post,  
            'matrix' : m,  
            'matinv' : m1,
            'scale': scale
            }
    
    def __str__(self):
        self._strtmp = f"Linear/Scale ({self.params['scale']})"
        return super().__str__()
    
class Rotation(Linear):
    '''
    transform.Rotation - linear transform that just rotates vectors
    
    Rotation transforms implement operations of the form:
        
        data_out = (post) + (rmatrix x (data + pre))
        
    where post, pre, data, and data_out are column ectors, and rmatrix
    is a square rotation matrix.  You specify the rotation angles between
    pairs of input coordinates.  
    
    You can specify one or more rotations by axis or with two shortcuts
    for 2D rotations (scalar rotation angle) or 3D rotations (Euler angles).
    
    By default, angles are specified in radians.  They can also be specified
    in degrees if you specify.
    
    Rotation is a subclass of Linear.
    
    Examples
    --------
    
        a = t.Rotation(43,'deg')       # 2-D rotation by 43 degrees CCW
        a = t.Rotation(math.pi/4)      # 2-D rotation by 45 degrees CCW
        a = t.Rotation([0,1,27],'deg') # 2-D rotation by 27 degrees CCW
        
        ### Two ways to express Euler angles in 3D
        a = t.Rotation( euler=np.array( [10, 20, 30] ), u='deg') # axial vector
        a = t.Rotation( [[0,1,30],[2,0,20],[1,2,10]], u='deg' )  # explicit axes
        
    Parameters
    ----------
    
    rot : numpy.ndarray or list or scalar or None
        This is either a scalar or a list or tuple of 1 or more 3-arrays. If 
        it's a scalar, it is implicitly a rotation from axis 0 toward axis 1.  
        If it's a collection of 3-arrays, the 0 and 1 elements are the "from"
        and "toward" axes of each rotation, and the 2 element is the amount
        of rotation.  The unit defaults to radians but can be set to degrees
        via the "u" parameter.  The dimension of the Transform is the largest
        axis referenced, or the dimension of the pre- or post- offset vectors,
        if they are larger.
        
        If you feed in a list or tuple, then the arrays are processed in 
        *reverse* order (by analogy to function composition) -- check the
        ordering of the euler-angle vs explicit-rotation demo above.
    
    /u : string (optional; keyword only; default 'rad')
        This is the angular unit.  Only the first character is checked: If 
        it is 'd' then angle is considered to be in degrees.  If it is 'r', then
        angle is considered to be in radians.
        
    /euler:  numpy.ndarray or None
        If this is specified, rot must be None; and axial must contain a 
        3-vector.  The 3 elements are Euler angles, in order -- they form
        an axial vector.  They are applied in dimension order: X (which 
        rotates axis 1->2), Y (which rotates axis 2->0), Z (which rotates axis)
        0->1).
    
    /pre : numpy.ndarray (optional; default = 0 )
        Optional; if present must be a vector.  Offset vector to be added to
        the input data before hitting with the rotation matrix.
    
    /post : numpy.ndarray ( optional; default = 0) 
        Optional; if present must be a vector.  Offset vector to be added to
        the return data after hitting with the rotation matrix.
    '''
    def __init__(self,                      
                 rot=None,                  
                 *, post=None, pre=None,    
                 euler=None,unit='rad',       
                 iunit=None,ounit=None,     
                 itype=None,otype=None,     
                 ):
        
        d_offs = self._parse_prepost( None,   pre, 'Rotation', 'Pre-offset',  'd')
        d_offs = self._parse_prepost( d_offs, post,'Rotation', 'Post-offset', 'd')
        
        if( rot is None ):
            if(euler is None):
                raise ValueError("Rotation: either rot or euler angles must be specified")
            if(len(euler) != 3):
                raise ValueError("Rotation: euler angles must have 3 components (axial vector)")
            rot = np.array( [ [0,1,euler[2]], [2,0,euler[1]], [1,2,euler[0]] ] );
        else:
            assert euler is None, "Rotation: must specify only one of rot and euler angles"
            if( not isinstance(rot, np.ndarray) ):
                rot = np.array(rot)

        if( len(rot.shape)>2 ):
            raise ValueError("Rotation: rot parameter must be a collection of 3-vectors")
        if( rot.size == 1 ):
            # simple scalar rotation - replace with a 3-list
            rot = np.array( [ [0,1,rot] ] )
                             
        if( len(rot.shape) == 1 ):
            rot = np.expand_dims(rot,0)
        
        fr_axes = rot[:,0].astype(int)
        to_axes = rot[:,1].astype(int)
        angs    = rot[:,2] 
        
        if any(fr_axes==to_axes):
            raise ValueError('Rotation: invalid axis-to-self rotation is not allowed')
        
        # Generate the angular coefficient using the astropy Quantity infrastructure
        if(isinstance(unit,units.quantity.Quantity)):
            ang_quantity = unit
        elif(isinstance(unit,units.core.Unit)):
            ang_quantity= 1.0 * unit
        else:
            try:
                unit = getattr(units,unit)
            except:
                raise ValueError(f"Radial: couldn't convert '{unit}' to an astropy unit object")
            ang_quantity = 1.0 * unit   
        try:
            ang_quantity = ang_quantity.to(units.radian)
        except:
            raise ValueError(f"Radial: '{unit}' doesn't appear to be an angular unit or quantity")
        
        angs = angs * ang_quantity.value
                    
        d_fr = np.maximum( np.amax(fr_axes), np.amax(to_axes) ) + 1
        if(d_offs is None):
            d = d_fr
        elif(d_offs < d_fr):
            raise ValueError('Rotation: offset vectors must have at least the dims of the rotation')
        else:
            d = d_offs
        
        # Assemble identity matrix
        d = int(d)
        identity = np.zeros( [d,d] )
        for i in range(d):
            identity[i,i] = 1
        out = identity.copy()
        
        # Now loop over all rotations in order and assemble the total matrix
        for i in range(fr_axes.shape[0])[::-1]: 
            m = identity.copy()
            c = np.cos(angs[i])
            s = np.sin(angs[i])
            m[fr_axes[i],fr_axes[i]] = c
            m[to_axes[i],to_axes[i]] = c
            m[to_axes[i],fr_axes[i]] = s
            m[fr_axes[i],to_axes[i]] = -s
            out = np.matmul(m,out)
        
        # Finally -- build the object
        self.idim = d
        self.odim = d
        self.no_forward = False
        self.no_reverse = False # rotations are always invertible
        self.iunit = iunit
        self.ounit = ounit
        self.itype = itype
        self.otype = otype
        self.params = {                 
            'pre'    : pre,             
            'post'   : post,            
            'matrix' : out,             
            'matinv' : out.transpose(), 
            }
    
    def __str__(self):
        self._strtmp = "Linear/Rotation"
        return super().__str__()

class Offset(Linear):
    '''
    transform.Offset - linear transform that just displaces vectors
   
    Offset transforms implement operations of the form:
       
        data_out =  data + offset
    
    where data_out, data, and offset are column vectors. 
    
    Offset is a subclass of transform.Linear.
    
    Parameters
    ----------
    
    offset - np.ndarray (vector)
        The amount to offset the data vectors
    '''
    def __init__(self, offset):
        if( not isinstance(offset, np.ndarray)):
            offset = offset + np.array(0)
            
        if( len(np.shape(offset))>1 ):
            raise ValueError("Offset: input must be a vector")
        d = np.shape(offset)[0]
        
        self.idim = d
        self.odim = d
        self.no_forward = False
        self.no_reverse = False
        self.iunit = None
        self.ounit = None
        self.itype = None
        self.otype = None
        self.params = {
            'pre': offset,
            'post': None,
            'matrix' : None,
            'matinv' : None,
        }
    
    def __str__ (self):
        self._strtmp = "Linear/Offset"
        return super().__str__()

class Radial(Transform):
    '''
    transform.Radial - Convert Cartesian to radial/cylindrical coordinates.
    
    Convert Cartesian to radial/cylindrical coordinates.  
    (2-D/3-D; with inverse)

    Converts 2-D Cartesian to radial (theta,r) coordinates.  You can choose
    direct or conformal conversion.  Direct conversion preserves radial
    distance from the origin; conformal conversion preserves local angles,
    so that each small-enough part of the image only appears to be scaled
    and rotated, not stretched.  Conformal conversion puts the radius on a
    logarithmic scale, so that scaling of the original image plane is
    equivalent to a simple offset of the transformed image plane.

    If you use three or more dimensions, the higher dimensions are ignored,
    yielding a conversion from Cartesian to cylindrical coordinates.  If you 
    use higher dimensionality than 2, you must manually specify the origin or 
    you will get dimension mismatch errors when you apply the transform.
    
    This Transform works (by default) *opposite* to standard polar/radial
    coordinates: angle counts *clockwise* rather than *widdershins*. This 
    preserves the chirality of image features when images are resampled to 
    polar/radial coordinates.  To get the standard, set the "ccw=True" 
    flag.
    
    
    Parameters
    ----------
    
    r0: float y (optional; default = 0). If defined, this floating-point value 
    causes t_radial to generate (theta, ln(r/r0)) coordinates out.  Theta is in 
    radians, and the radial coordinate varies by 1 for each e-folding of the 
    r0-scaled distance from the input origin.  The logarithmic scaling is 
    useful for viewing both large and small things at the same time, and for 
    keeping shapes of small things preserved in the image.  
        
    origin: This is the origin of the expansion. Pass in a np.array. Default 
    is set to np.array([0,0])

    unit: Unit [default 'rad'] This is the angular unit to be used for the 
    azimuth.
    
    ccw: (default False) if set, this flag makes angle increase widdershins, 
    reversing planar chirality of small features but also conforming to the 
    usual mathematical standard. 
    
    pos_only: (default True) if set, this flag ensures angles are always in 
    the interval [0,2pi); if false, they are in (-pi,pi].
                                            
    '''

    def __init__(self, *,                      
                 r0 = None,
                 iunit = None, 
                 ounit  = None,
                 itype = None, 
                 otype = None,
                 idim  = 2, 
                 odim = 2,
                 origin = np.zeros(2),
                 unit = 'radian',
                 ccw = False,
                 pos_only = True
                ):
        
    
        if( not( isinstance( origin, np.ndarray ) ) ) :
            origin = np.array([origin])

        if r0 is not None:
            otype = ["Azimuth", "Ln radius"]
            r0_sq = r0 * r0
        else:
            otype = ["Azimuth", "Radius"]
            r0_sq = None

        # Generate the angular coefficient using the astropy Quantity infrastructure
        if(isinstance(unit,units.quantity.Quantity)):
            ang_quantity = unit
        elif(isinstance(unit,units.core.Unit)):
            ang_quantity= 1.0 * unit
        else:
            try:
                unit = getattr(units,unit)
            except:
                raise ValueError(f"Radial: couldn't convert '{unit}' to an astropy unit object")
            ang_quantity = 1.0 * unit   
        try:
            ang_quantity = ang_quantity.to(units.radian)
        except:
            raise ValueError(f"Radial: '{unit}' doesn't appear to be an angular unit or quantity")
        
        if ounit is None:
            ounit = [f"{ang_quantity.unit}", None]
        elif ounit[0] is None:
            ounit[0] = f"{ang_quantity.unit}"
            
        ###Generate the object        
        self.idim = idim
        self.odim = odim
        self.no_forward = False
        self.no_reverse = False
        self.iunit = iunit
        self.ounit = ounit
        self.itype = itype
        self.otype = otype
        self.params = {
            'origin'   : origin,
            'r0'       : r0,
            'r0_sq'    : r0_sq,
            'ang_coeff': ang_quantity.value,
            'ccw'      : ccw,
            'pos'      : pos_only,
        }


    def _forward( self, data: np.ndarray ):

        out = np.ndarray(data.shape)

        origin = self.params['origin'][0:2]
        if not np.all((origin == 0)):
            data = data - origin
        
        if( self.params['ccw'] ):
            out[..., 0] = (np.arctan2( data[..., 1], data[..., 0])) / self.params['ang_coeff']            
        else:
            out[..., 0] = (np.arctan2(-data[..., 1], data[..., 0])) / self.params['ang_coeff']
        
        if(self.params['pos']):
            out[...,0] %= 2*np.pi

        if self.params['r0']:
            out[..., 1] = 0.5 * np.log( np.sum(data*data, axis=-1) / self.params['r0_sq'] )
        else:
            out[..., 1] = np.sqrt( np.sum(data*data, axis=-1) )
        return out

    def _reverse( self, data: np.ndarray ):
        
        d0 = data[..., 0] * self.params['ang_coeff']
        
        out = np.ndarray(data.shape)
        out[...,0] = np.cos(d0)
        if( self.params['ccw'] ):
            out[...,1] = np.sin(d0)
        else:
            out[...,1] = -np.sin(d0)            

        if self.params['r0'] is not None:
            out *= self.params['r0'] * np.exp(data[...,1,np.newaxis])
        else:
            out *= data[...,1,np.newaxis]
        
        origin = self.params['origin'][0:2]
        if not np.all((origin == 0)):
            out = out + origin
        return out

    def __str__(self):
        if(not hasattr(self,'_strtmp')):
            self._strtmp = 'Radial'
        return super().__str__()        
    


class Spherical(Transform):
    '''
    transform.Spherical - Convert Cartesian to spherical coordinates / radians.  
    (3-D; with inverse)

    Convert 3-D Cartesian to spherical (theta, phi, r) coordinates.  Theta
    is longitude, centered on 0, and phi is latitude, also centered on 0. 
    The default is for theta and phi to be in radians; you can select degrees 
    if you want them.

    Just as the transform.Radial 2-D transform acts like a 3-D
    cylindrical transform by ignoring third and higher dimensions,
    Spherical acts like a hypercylindrical transform in four (or higher)
    dimensions.  Also as with transform.Radial, you must manually specify
    the origin if you want to use more dimensions than 3.

    Parameters
    ----------
        
    origin: This is the origin of the expansion. Pass in a np.array. Default 
    is set to np.array([0,0,0])

    unit: Unit [default 'rad'] This is the angular unit to be used for the 
    longitude and latitude angles. Acceptable values are "degrees","radians". 

    '''

    def __init__(self, *,
                 iunit = None, 
                 ounit  = None, 
                 itype = None, 
                 otype  = None,
                 idim  = 3, 
                 odim = 3,
                 origin = np.zeros(3),
                 unit = 'radian'
                ):

        if( not( isinstance( origin, np.ndarray ) ) ) :
            origin = np.array([origin])

        otype = ["Longitude", "Latitude", "Radius"]

        # Generate the angular coefficient using the astropy Quantity infrastructure
        if(isinstance(unit,units.quantity.Quantity)):
            ang_quantity = unit
        elif(isinstance(unit,units.core.Unit)):
            ang_quantity= 1.0 * unit
        else:
            try:
                unit = getattr(units,unit)
            except:
                raise ValueError(f"Spherical: couldn't convert '{unit}' to an astropy unit object")
            ang_quantity = 1.0 * unit   
        try:
            ang_quantity = ang_quantity.to(units.radian)
        except:
            raise ValueError(f"Spherical: '{unit}' doesn't appear to be an angular unit or quantity")

        if ounit is None:
            ounit = [f"{ang_quantity.unit}",f"{ang_quantity.unit}", None]
        elif ounit[0] is None:
            ounit[0] = f"{ang_quantity.unit}"
        elif ounit[1] is None:
            ounit[1] = f"{ang_quantity.unit}"

        ###Generate the object        
        self.idim = idim
        self.odim = odim
        self.no_forward = False
        self.no_reverse = False
        self.iunit = iunit
        self.ounit = ounit
        self.itype = itype
        self.otype = otype
        self.params = {
            'origin'  : origin,
            'ang_coeff': ang_quantity.value,
            }


    def _forward( self, data: np.ndarray ):

        out = data.copy()

        origin = self.params['origin'][0:3]

        if not np.all((origin == 0)):
            data = data - origin

        d0 = data[..., 0].copy()
        d1 = data[..., 1].copy()
        d2 = data[..., 2].copy()

        out[..., 0] = (np.arctan2(d1, d0))
        out[..., 2] = (np.sqrt(d0*d0 + d1*d1 + d2*d2))
        out[..., 1] = (np.arcsin(d2 / out[..., 2]))
                       
        out[..., 0:2] = out[..., 0:2] * self.params['ang_coeff'] 

        return out

    def _reverse( self, data: np.ndarray ):

        theta = data[..., 0].copy() * self.params['ang_coeff']
        phi = data[..., 1].copy() * self.params['ang_coeff']
        r = data[..., 2].copy()

        out = np.ndarray(data.shape)
        
        out[..., 2] = r * np.sin(phi) #z
        out[..., 0] = r * np.cos(phi) #x
        out[..., 1] = out[..., 0] * np.sin(theta) #y
        out[..., 0] = out[..., 0] * np.cos(theta) #x
        
        origin = self.params['origin'][0:3]
        if not np.all((origin == 0)):
            out = out + origin
        return out

    def __str__(self):
        if(not hasattr(self,'_strtmp')):
            self._strtmp = 'Spherical'
        return super().__str__()    


class Quadpin(Transform):
    '''
    transform.Quadpin - Quadratic pincushion scaling (n-d; with inverse)
    
    Quadratic scaling emulates pincushion in a cylindrical optical system:
    A separate quadratic scaling is applied to each axis.  You can apply
    separate distortion along any of the principal axes.  If you want
    different axes, compose with Linear to rotate them to the correct angle.  
    The scaling options may be scalars or vectors; if they are scalars then 
    the expansion is isotropic.

    The formula for the expansion is:

        f(a) = ( <a'> + <strength> * <a'>^2/<length> ) / (abs(<strength>) + 1)

    where <a'>=(<a>-origin), <strength> is a scaling coefficient and <length> is 
    a fundamental length scale. Negative values of <strength> result in a 
    pincushion type contraction.

    Note that, because quadratic scaling does not have a strict inverse for
    coordinate systems that cross the origin, x * np.abs(x) is used rather than 
    x**2, which adequately produces pincushion and barrel distortion.

     but means 
    that t_quadratic does not behave exactly like t_cubic with a null cubic strength coefficient.

    Parameters
    ----------
        
    origin :  numpy.ndarray (optional; default = np.array([0,0]))
        This is the origin of the pincushion. Pass in a np.array. 

    length : float (optional; default = 1)
        The fundamental scale of the transformation -- the radius that remains
        unchanged.

    strength : float (optional; default = 0.1)
        The relative strength of the pincushion.

    dim : int (optional; positional or keyword; default = size of input)
        This is an optional dimension specifier in case you pass in a scalar
        and no other dimensionality hints.



    Notes
    -----

    due to the use of x * np.abs(x) instead of x**2, t_quadratic does not behave 
    exactly like t_cubic with a null cubic strength coefficient.

    '''

    def __init__(self, *,
                 iunit = None, 
                 ounit  = None, 
                 itype = None, 
                 otype  = None,
                 idim  = 0, 
                 odim = 0,
                 origin = 0,
                 length = 1,
                 strength = 0.1,
                 dim = None
                 ):

        if ( dim != None ):
            idim = dim
            odim = dim

        self.idim       = idim
        self.odim       = odim
        self.no_forward = False
        self.no_reverse = False
        self.iunit      = iunit
        self.ounit      = ounit
        self.itype      = itype
        self.otype      = otype
        self.params = {
            'origin'  : origin,
            'length' : length,
            'strength': strength,
            'dim'     : dim
            }


    def _forward( self, data: np.ndarray ):

        data = data.copy()

        if((self.idim==0) and (self.odim==0)):
            self.idim, self.odim = data.shape

        origin = self.params['origin']
        strength = self.params['strength']
        length = self.params['length']
        dim = self.params['dim']

        if( dim != None ):
            out = data[..., :dim].copy()
        else:
            out = data.copy()

        if not np.all((origin == 0)):
            try:
                origin = np.array([origin])
                out = out - origin
            except:
                raise ValueError("Quadpin : origin shape %s is not broadcastable with data shape %s" % (out.shape, origin.shape))

        out = out + strength * (out * np.abs(out)) / length
        out = out / ( np.abs(strength)+1 )

        if not np.all((origin == 0)):
            out = out + origin
        
        if( dim != None ):
            data[..., :dim] = out
        else:
            data = out

        return data


    def _reverse( self, data: np.ndarray ):

        origin = self.params['origin']
        strength = self.params['strength']
        length = self.params['length']
        dim = self.params['dim']

        data = data.copy()

        if((self.idim==0) and (self.odim==0)):
            self.idim, self.odim = data.shape

        if( dim != None ):
            out = data[..., :dim].copy()
        else:
            out = data.copy()

        Origindata = out.copy()

        if not np.all((origin == 0)):
            try:
                origin = np.array([origin])
                Origindata = out - origin
            except:
                raise ValueError("Quadpin : origin shape %s is not broadcastable with data shape %s" % (out.shape, origin.shape))

        out = -1.0 + np.sqrt( 1.0 + 4.0 * strength / length * np.abs(Origindata) * (1.0 + np.abs(strength)))
        out = out * length / ( 2.0 * strength ) 
        out = out * (1.0 - 2.0 * (data < origin) );
            
        if not np.all((origin == 0)):
            try:
                out = out + origin
            except:
                raise ValueError("Quadpin : origin shape %s is not broadcastable with data shape %s" % (out.shape, origin.shape))

        if( dim != None ):
            data[..., :dim] = out
        else:
            data = out

        return data

    def __str__(self):
        if(not hasattr(self,'_strtmp')):
            self._strtmp = 'Quadpin'
        return super().__str__()



class Cubicpin(Transform):
    '''
    transform.Cubicpin scaling - cubic pincushion scaling (n-d; with inverse)

    Cubic scaling is a generalization of Transform.Quadpin to a purely cubic 
    expansion.  You can apply separate distortion along any of the principal 
    axes.  If you want different axes, compose with Linear to rotate them to 
    the correct angle.  The scaling options may be scalars or vectors; if they 
    are scalars then the expansion is isotropic.

    The formula for the expansion is:

        f(a) = ( <a'> + <strength> * <a>^3/<length>^2 ) / (1 + abs(<strength>) + origin

    where <a'>=(<a>-origin), <strength> is a scaling coefficient and <length> is 
    a fundamental length scale. Negative values of <strength> result in a 
    pincushion type contraction.

    That is a simple pincushion expansion/contraction that is fixed at a 
    distance of length from the origin.

    Because there is no quadratic term the result is always invertible with
    one real root, removing the need for complex numbers or multivalued 
    solutions.


    Parameters
    ----------
        
    origin :  numpy.ndarray (optional; default = np.array([0,0]))
        This is the origin of the pincushion. Pass in a np.array. 

    length : float (optional; default = 1)
        The fundamental scale of the transformation -- the radius that remains
        unchanged.  (default=1)

    strength : float (optional; default = 0.0)
        The relative strength of the pincushion.

    dim : int (optional; positional or keyword; default = size of input)
        This is an optional dimension specifier in case you pass in a scalar
        and no other dimensionality hints.


    '''

    def __init__(self, *,
                 iunit = None, 
                 ounit  = None, 
                 itype = None, 
                 otype  = None,
                 idim  = 0, 
                 odim = 0,
                 origin = 0,
                 length = 1,
                 strength = 0.0,
                 dim = None
                 ):

        if ( dim != None ):
            idim = dim
            odim = dim

        ###Generate the object        
        self.idim = idim
        self.odim = odim
        self.no_forward = False
        self.no_reverse = False
        self.iunit = iunit
        self.ounit = ounit
        self.itype = itype
        self.otype = otype
        self.params = {
            'origin'   : origin,
            'length'   : length,
            'strength' : strength,
            'dim'      : dim
            }


    def _forward( self, data: np.ndarray ):

        data = data.copy()

        if((self.idim == 0) and (self.odim ==0)):
            self.idim, self.odim = data.shape

        origin = self.params['origin']
        strength = self.params['strength']
        length = self.params['length']
        dim = self.params['dim']

        if( dim != None ):
            out = data[..., :dim].copy()
        else:
            out = data.copy()

        if not np.all((origin == 0)):
            try:
                origin = np.array([origin])
                out = out - origin
            except:
                raise ValueError("Cubicpin : origin shape %s is not broadcastable with data shape %s" % (out.shape, origin.shape))

        dl0 = out / length
        out = out + strength * out * dl0 * dl0
        out = out / ( (strength*strength)+1 )

        if not np.all((origin == 0)):
            out = out + origin

        if( dim != None ):
            data[..., :dim] = out
        else:
            data = out
        return data

    def _reverse( self, data: np.ndarray ):

        origin = self.params['origin']
        strength = self.params['strength']
        length = self.params['length']
        dim = self.params['dim']

        data = data.copy()

        if((self.idim==0) and (self.odim==0)):
            self.idim, self.odim = data.shape

        if( dim != None ):
            out = data[..., :dim].copy()
        else:
            out = data.copy()

        if not np.all((origin == 0)):
            try:
                origin = np.array([origin])
                out = out - origin
            except:
                raise ValueError("Cubicpin : origin shape %s is not broadcastable with data shape %s" % (data.shape, origin.shape))
        else:
            out = out.copy()

        out = out * (strength+1)
        Avar = strength/length/length
        Cvar = 1
        Dvar = -out
        Alphavar = 27 * Avar * Avar * Dvar;
        Betavar = 3 * Avar * Cvar        
        inner_root = np.sqrt( Alphavar * Alphavar + 4.0 * Betavar * Betavar * Betavar )
            
        try:
            aVar2 = 0.5 * ( Alphavar - inner_root )
            cuberootNegative = 1 - 2 * (aVar2<0);
            cuberootNegative = cuberootNegative * (  abs(aVar2) ** (1/3) )
            cuberoot = cuberootPositive
        except: 
            aVar2 = 0.5 * ( Alphavar + inner_root )
            cuberootPositive = 1 - 2 * (aVar2<0);
            cuberootPositive = cuberootPositive * (  abs(aVar2) ** (1/3) )
            cuberoot = cuberootPositive
        
        cuberoot = cuberootPositive + cuberootNegative
            
        try:
            out = (-1 / (3 * Avar)) * cuberoot
        except ZeroDivisionError:
            out = (-1 / (3 * np.nan)) * cuberoot
            
        if not np.all((origin == 0)):
            try:
                out = out + origin
            except:
                raise ValueError("Cubicpin : origin shape %s is not broadcastable with data shape %s" % (data.shape, origin.shape))

        if( dim != None ):
            data[..., :dim] = out
        else:
            data = out

        return data

    def __str__(self):
        if(not hasattr(self,'_strtmp')):
            self._strtmp = 'Cubicpin'
        return super().__str__()


class Poly(Transform):

    '''
    transform.Poly scaling - Generic forward polynomial scaling using 
    a pincusion formalism (n-d; with no inverse).

    The Polynomial scaling is a generalization using both even and odd powers, 
    and therefore inverse transforms are not possible. A generic Quadratic 
    pincusion transform, with inverse, can be generated with the 
    transform.QuadPin tranformation. A generic Cubic pincusion transform, with 
    inverse, can be generated with the transform.CubicPin tranformation.

    You can apply a separate distortion along any of the principal axes. If
    you want different axes, use Transform.Linear to rotate them to the 
    correct angle.  The scaling options may be scalars or vectors; if they 
    are scalars then the expansion is isotropic.

    Two polynomial expansions are included, and can be chosen with the 
    denominator keyword, (0|1). The difference between the expansions are
    the denominator strengths; <den>=0 uses a non varying (1 + <strength>) 
    <den>=1 uses (1 + <strength>^(<degree>-1)).

    The formula for expansion 0 is:

    f(a) = ( <a'> 
             + <a'>^2 * <strength>/<length> 
             + <a'>^3 * <strength>/<length>^2 
                     .... 
             + <a'>^<degree> * <strength>/<length>^(<degree>-1) ) 
             / (1 + <strength>)
             + origin

    The formula for expansion 1 is:

    f(a) = ( <a'> 
             + <a'>^2 * <strength>/<length> 
             + <a'>^3 * <strength>/<length>^2 
                     .... 
             + <a'>^<degree> * <strength>/<length>^(<degree>-1) ) 
             / (1 + <strength>^(<degree>-1))
             + origin

    where <a'>=(<a>-origin), <strength> is a scaling coefficient and <length> is 
    a fundamental length scale. Negative values of <strength> result in a 
    pincushion type contraction.


    Parameters
    ----------
        
    origin :  numpy.ndarray (optional; default = np.array([0,0]))
        This is the origin of the pincushion. Pass in a np.array. 

    length : float (optional; default=1)
        The fundamental scale of the transformation -- the radius that remains
        unchanged.  (default=1)

    strength : float (optional; default=0.1)
        The relative strength of the pincushion.

    dim : int (optional; positional or keyword; default = size of input)
        This is an optional dimension specifier in case you pass in a scalar
        and no other dimensionality hints.
    
    degree :  int (optional; default=2)
        TThe largest degree of the polynomial expansion.

    den : int (optional; (0|1); default=1)
        The den keyword decides which deminator to use in the polynomial 
        expansion. If any other value is selected the method reverts to the
        default. 

    '''

    def __init__(self, *,
                 iunit = None, 
                 ounit  = None, 
                 itype = None, 
                 otype  = None,
                 idim  = 0, 
                 odim = 0,
                 origin = 0,
                 length = 1,
                 strength = 0.0,
                 degree = 2,
                 dim = None,
                 den = 1
                 ):

        if ( dim != None ):
            idim = dim
            odim = dim

        self.idim       = idim
        self.odim       = odim
        self.no_forward = False
        self.no_reverse = True
        self.iunit      = iunit
        self.ounit      = ounit
        self.itype      = itype
        self.otype      = otype
        self.params = {
            'origin'  : origin,
            'length'  : length,
            'strength': strength,
            'degree'  : degree,
            'dim'     : dim,
            'den'     : den
            }
            
    
    def _forward( self, data: np.ndarray ):

        data = data.copy()

        if((self.idim == 0) and (self.odim ==0)):
            self.idim, self.odim = data.shape

        origin = self.params['origin']
        strength = self.params['strength']
        length = self.params['length']
        degree = self.params['degree']
        den = self.params['den']
        dim = self.params['dim']        

        if( dim != None ):
            out = data[..., :dim].copy()
        else:
            out = data.copy()

        if not np.all((origin == 0)):
            try:
                origin = np.array([origin])
                out = out - origin
            except:
                raise ValueError("Poly : origin shape %s is not broadcastable with data shape %s" % (out.shape, origin.shape))

        aConst = out
        if (den!=1):
            StrengthConst = (1.0 / (np.abs(strength) +1 ))
        else:
            StrengthConst = (1.0 / (np.abs(strength)**(degree-1) +1 ))

        for degreeStep in range(2, degree+1):
            aConst = aConst + strength * (out ** degreeStep) / (length**(degreeStep-1))

        #invoke below with degree = 3 to produce the forward cubic transform
        #aConst = aConst - strength * (out**(degree-1)) / (length**(degree-2))

        out = StrengthConst * aConst

        if not np.all((origin == 0)):
            out = out + origin

        if( dim != None ):
            data[..., :dim] = out
        else:
            data = out

        return data

    def __str__(self):
        if(not hasattr(self,'_strtmp')):
            self._strtmp = 'Poly'
        return super().__str__()



class Quarticpin(Poly):
    '''
    Transform.Quartic scaling - pincushion (n-d; with inverse)

    Transform.Quartic is a subclass of the Poly(nomial) class with polynomial
    of <degree>=4 and denominator <den>=1 preselected to emulate a Quartic
    scaling.

    You can apply separate distortion along any of the principal axes.  If
    you want different axes, use Transform.Linear to rotate them to the 
    correct angle. The scaling options may be scalars or vectors; if they 
    are scalars then the expansion is isotropic.

    The formula for the expansion is:

    f(a) = ( <a'> 
             + <a'>^2 * <strength>/<length> 
             + <a'>^3 * <strength>/<length>^2 
                     .... 
             + <a'>^<degree> * <strength>/<length>^(<degree>-1) ) 
             / (1 + <strength>^(<degree>-1))
             + origin

    where <a'>=(<a>-origin), <strength> is a scaling coefficient and <length> is 
    a fundamental length scale. Negative values of <strength> result in a 
    pincushion type contraction.


    Parameters
    ----------
        
    origin :  numpy.ndarray (optional; default = np.array([0,0]))
        This is the origin of the pincushion. Pass in a np.array. 

    length : float (optional; default = 1)
        The fundamental scale of the transformation -- the radius that remains
        unchanged.  (default=1)

    strength : float (optional; default = 0.1)
        The relative strength of the pincushion.

    dim : int (optional; positional or keyword; default = size of input)
        This is an optional dimension specifier in case you pass in a scalar
        and no other dimensionality hints.

    '''
    def __init__(self, *,
                 iunit = None, 
                 ounit  = None, 
                 itype = None, 
                 otype = None,
                 idim  = 0, 
                 odim = 0,
                 origin = 0,
                 length = 1,
                 strength = 0.1,
                 degree = 4,
                 dim = None,
                 den=1
                 ):

        ### Finally - if dim, this becomes idim and odim
        if ( dim != None ):
            idim = dim
            odim = dim

        self.idim       = idim
        self.odim       = odim
        self.no_forward = False
        self.no_reverse = True
        self.iunit      = iunit
        self.ounit      = ounit
        self.itype      = itype
        self.otype      = otype
        self.params = {
            'origin'   : origin,
            'length'   : length,
            'strength' : strength,
            'degree'   : degree,
            'dim'      : dim,
            'den'      : den            
            }
        
    def __str__(self):
        if(not hasattr(self,'_strtmp')):
            self._strtmp = 'Poly/Quarticpin'
        return super().__str__()



class Projective(Transform):

    '''
    Projective transformation

    Projective transforms are simple quadratic, quasi-linear
    transformations.  They are the simplest transformation that
    can continuously warp an image plane so that four arbitrarily chosen
    points exactly map to four other arbitrarily chosen points.  They
    have the property that straight lines remain straight after transformation.

    You can specify your projective transformation directly in homogeneous
    coordinates, or (in 2 dimensions only) as a set of four unique points that
    are mapped one to the other by the transformation.

    Projective transforms are quasi-linear because they are most easily
    described as a linear transformation in homogeneous coordinates
    (e.g. (x',y',w) where w is a normalization factor: x = x'/w, etc.).
    In those coordinates, an N-D projective transformation is represented
    as simple multiplication of an N+1-vector by an N+1 x N+1 matrix,
    whose lower-right corner value is 1.

    If the bottom row of the matrix consists of all zeroes, then the
    transformation reduces to a linear affine transformation (as in
    L<transform.Linear.apply|/transform.Linear.invert>).

    If the bottom row of the matrix contains nonzero elements, then the
    transformed x,y,z,etc. coordinates are related to the original coordinates
    by a quadratic polynomial, because the normalization factor 'w' allows
    a second factor of x,y, and/or z to enter the equations.

    Notes
    -----
    

    Parameters
    ----------

    matrix : numpy.ndarray (optional; default = identity)
        Optional; if specified must be a 2-D array and the homogeneous-
        coordinate matrix to use. As always in numpy, the
        matrix is addressed in (row,column) format -- so if you use 
        nested-brackets notation for np.array(), then the innermost vectors
        are rows of the matrix. If specified, this is the homogeneous-coordinate 
        matrix to use.  It must be N+1 x N+1, for an N-dimensional transformation.

    points : numpy.ndarray (optional)
    If specified, this is the set of four points that should be mapped 
        one to the other. The homogeneous-coordinate matrix is calculated from 
        them.  You should feed in a 2x2x4 numpy.ndarray, where the 0 dimension 
        runs over coordinate, the 1 dimension runs between input and output, and
        the 2 dimension runs over point.  For example, specifying:

        input = numpy.ndarray([ [[0,1],[0,1]], [[5,9],[5,8]], 
                                [[9,4],[9,3]], [[0,0],[0,0]] ])

        maps the origin and the point (0,1) to themselves, 
                            the point (5,9) to (5,8), and
                            the point (9,4) to (9,3).
    ''' 
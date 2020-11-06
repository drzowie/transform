#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transform subclasses for basic coordinate transforms
"""
import numpy as np
import math as math
import astropy.io.fits as fits
from astropy.io.fits import CompImageHDU, HDUList, Header, ImageHDU, PrimaryHDU
from .core import Transform


class Linear(Transform):
    '''
    transform.Linear - linear transforms
    
    Linear tranforms consist of an offset and a matrix operation.  it 
    implements the transform:
        
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
        # Hammer data into NumPy form
        if( not isinstance(data, np.ndarray) ):
            data = np.array(data)
            
        # check the dimesnions to see if the dat < input dimensions
        if( data.shape[-1] < self.idim ):
            raise ValueError('This Linear needs {self.idim}-vecs; source has {data.shape[0]}-vecs')
 
#TO GO        
#        ## Chop ending vector elements off if necessary
        data0 = data[...,0:self.idim]
        
        ## Handle pre-offset
        if( self.params['pre'] is not None ):
            data0 = data0 + self.params['pre']
        
        ## Handle matrix
        m = self.params['matrix']
        if( m is not None ):
            data0 = np.expand_dims(data0,-1)  # convert vectors to Mx1
            data0 = np.matmul( m, data0 ) 
            data0 = data0[...,:,0]               # convert back to vectors
        
        ## Handle post-offset
        if( self.params['post'] is not None ):
            data0 = data0 + self.params['post']
        
        ## Handle re-attaching longer vector elements if necessary, and return
        if( data.shape[-1] > self.idim ):        
            return( np.append( data0, data[...,self.idim:], axis=-1 ) )
        else:
            return( data0 )

    def _reverse( self, data: np.ndarray ):
        if( not isinstance(data, np.ndarray) ):
            data = np.array(data)
            
        if( data.shape[-1] < self.odim ):
            raise ValueError('This reverse-Linear needs {self.odim}-vecs; source has {data.shape[0]}-vecs')

#TO GO
#        ## Chop ending vector elements off if necessary
        data0 = data[...,0:self.odim]
        
        ## Handle reversing the post-offset
        if( self.params['post'] is not None ):
            data0 = data0 - self.params['post']
        
        ## Handle inverse-matrix
        m = self.params['matinv']
        if( m is not None ):
            data0 = np.expand_dims(data0,-1)
            data0 = np.matmul( m, data0 )
            data0 = data0[...,:,0]
            
        ## Handle reversing the pre-offset
        if( self.params['pre'] is not None ):
            data0 = data0 - self.params['pre']
       
        ## Reattach the longer vector elements if necessary, and return
        if( data.shape[-1] > self.odim ):
            return( np.append( data0, data[...,self.odim:], axis=-1 ) )
        else:
            return( data0 )
        


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
        if( dimspec is not None   and  not isinstance( dimspec, np.ndarray) ):
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
    simple scale, specify d=1 in the construction parameters.
    
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
            }
    
    def __str__(self):
        self.strtmp = "Linear/Scale"
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
                 euler=None, u='rad',       
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
        
        if( u[0] == 'r' ):
            pass
        elif( u[0] == 'd' ):
            angs = angs * math.pi/180
                    
        d_fr = np.maximum( np.amax(fr_axes), np.amax(to_axes) ) + 1
        if(d_offs is None):
            d = d_fr
        elif(d_offs < d_fr):
            raise ValueError('Rotation: offset vectors must have at least the dims of the rotation')
        else:
            d = d_offs
        
        # Assemble identity matrix
        d = d.astype(int)
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
    
class FITS(Linear):
    '''
    transform.FITS - linear World Coordinate System translation
    
    FITS Transforms implement the World Coordinate System (WCS) that is used in 
    the FITS image standard that's popular among scientists.  (WCS: Greisen & 
    Calabretta 2002; "http://arxiv.org/abs/astro-ph/0207407") WCS includes 
    both linear and nonlinear components; at present FITS Transforms only 
    represent the linear component.
    
    FITS Transforms convert vectors from standard (X,Y) image pixel 
    coordinates (in which (0,0) is the center of the pixel at lower left of 
    the image, X runs right, and Y runs up), to world coordinates using the
    WCS information embedded in a FITS header.
    
    Scientific variable types and unit names from the FITS header are imported
    into the Transform object, for future use although they are at present 
    advisory only.
    
    You can also set the number of dimensions to represent in the header.  If
    you do that, only the first few dimensions of the FITS header are imported
    into the object, and matrix elements that mix in higher dimensions (etc.)
    are ignored.  That is useful, e.g., for handling RGB images that are 
    organized as (X,Y) image planes of each color (X,Y,C).  
    
    Note that the pixel coordinates are *not the same* as default NumPy array 
    indices, since X is generally the smallest-stride index in the pixel 
    coordinate system and NumPy arrays place the smallest stride at the *end*
    of index vectors (so that normal indexing in NumPy treats pixel 
    coordinates as (Y,X)).
    
    Because only the linear portion of the WCS standard is supported 
    (at present), FITS is currently a subclass of Linear.  
    
    Parameters
    ----------
    
    Header: An astropy.fits.ImageHDU or astropy.fits.Header or dictionary
    
    /dim: an optional limiting dimension
    '''
    def __init__(self, header, /, dim=None):
        pass
    
    def __str__(self):
        self.strtmp = "FITS"
        return super().__str__()
    
    def parse_data(data, hdu=None):
        '''
        Parse data to return a Numpy array and WCS object.
        If the dimensions 
        '''

        ### reshape if needed
        orig_shape = data.shape
        if data.ndim == 1:
            data = data.reshape((1, 1, data.size))
        else:
            data = data.reshape((data.shape[0], 1, data[0].size))


        if isinstance(data, str):
            return parse_data(fits.open(data), hdu=hdu)

        elif isinstance(data, HDUList):
            if hdu is None:
                if len(data) > 1:
                    raise ValueError("More than one Header Data Unit (HDU) is present,"
                                     "please specify HDU to use with hdu=option")
                else:
                    hdu=0
            return parse_data(data[hdu])

        ### Add WCS structure
        
        elif isinstance(data, (PrimaryHDU, ImageHDU, CompImageHDU)):
            return data.data, WCS(data.header)
    
        elif isinstance(data, tuple) and isinstance(data[0], np.ndarray):
            if isinstance(data[1], Header):
                return data[0], WCS(data[1])
            else:
                return data

        elif isinstance(data, astropy.nddata.NDDataBase):
            return data.data, data.wcs
        else:
            raise TypeError("data should either be an HDU object or a tuple "
                            "of (array, WCS) or (array, Header)")


        ### Do stuff with the data


        ### Update the header



        ### reshape the data to the original dimensions
        out = out.reshape(og_shape)

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
            'pre'    : None,             
            'post'   : None,            
            'matrix' : out,             
            'matinv' : out.transpose(), 
            }

    
    def _forward( self, data: np.ndarray ):    
        pass
    
    
    
    
    
    
# -*- coding: utf-8 -*-
"""
Helper routines for the main Transform

These are mainly to do with interpolation: several interpolators exist
in numpy but they don't have clean treatments of boundary conditions, 
or an orthogonal interface for the various common interpolation methods.

"""
import numpy as np
import copy
from itertools import repeat

def apply_boundary(vec, size, /, bound='f', rint=True):
    '''
    apply_boundary - apply boundary conditions to a vector or collection
    
    apply_boundary accepts input vector(s) and applied boundary conditions
    on a suitably-dimensioned rectangular volume.  It can process generic
    floating-point vectors but by default it converts its input to int,
    for use in indexing arrays.

    Parameters
    ----------
    vec : array
        Collection of vectors (in the final axis) to have boundaries applied
        
    size : list or array
        List containing the allowable size of each dimension in the index.  The
        allowable dimension is on the interval [0,size).
        
    bound : string or list of strings (default 'f')
        The boundary type to apply. Only the first character is checked.
        Each string must start with one of:
                        
            'f' - forbid boundary violations
            
            't' - truncate the array at the boundary.  Values outside are
                replaced with -1.
                
            'e' - extend the array at the boundary.  Values outside are 
                replaced with the nearest value in the source array.
                
            'p' - periodic boundary conditions apply. Indices are modded with
                the size of the corresponding axis in the source array.
                
            'm' - mirror boundary conditions apply.  Indices reflect off each
                boundary of the data, counting backward to the opposite 
                boundary (where they reflect again).
                
    rint : Boolean (default True)
        Causes the vectors to be rounded and reduced to ints, for use in 
        indexing regular gridded data.  

    Returns
    -------
    The index, with boundary conditions applied 
    '''
    try:
        shape = vec.shape
    except:
        raise ValueError("apply_boundary requires a np.ndarray")
        
    # I'm sure there's a better way to do this - explicitly broadcast
    # size if it's a scalar or a 1-element list
    if( not isinstance(size,(tuple,list,np.ndarray) )):
        size = list( map( lambda i:size, range(shape[-1])))
    elif(len(size)==1):
        size = list( map ( lambda i:size[0], range(shape[-1]))) 
    # Check that size matches the vec
    if(len(size) != shape[-1]):
        raise ValueError("apply_boundary: size vector must match vec dims")
    
    # Now do the same thing with the boundary string(s)
    if( not isinstance(bound, list) ):
        bound = list( map( lambda i:bound, range(shape[-1])))
    elif(len(bound)==1):
        bound = list( map( lambda i:bound[0], range(shape[-1])))
    # Check that boundary size matches the vec
    if(len(bound) != shape[-1]):
        raise ValueError("apply_boundary: boundary list must match vec dims")
        
    # rint if clled for
    if rint and not issubclass(vec.dtype.type, np.integer):
        vec = np.floor(vec+0.5).astype('int')
    else:
        vec = copy.copy(vec)
        
    
    # Apply boundary conditions one dim at a time
    for ii in range(shape[-1]):
        b = bound[ii][0]
        s = size[ii]
    
        ## forbid
        if    b=='f':
            if not  (np.all((vec[...,ii] >= 0) * (vec[...,ii] < s))):
                raise ValueError("apply_boundary: boundary violation with 'forbid' condition")
        ## truncate      
        elif  b=='t': 
            # Replace values outside the boundary with -1
            np.place(vec[...,ii], (vec[...,ii]<0)+(vec[...,ii]>=s), -1)
        ## extend 
        elif  b=='e':
            # Replace values outside the boundary with the nearest boundary
            np.place(vec[...,ii], (vec[...,ii]<0), 0)
            np.place(vec[...,ii], (vec[...,ii]>= s), s-1)
        ## periodic
        elif  b=='p':
            # modulo
            vec[...,ii] = vec[...,ii] % s
        ## mirror
        elif  b=='m': 
            # modulo at twice the size
            vec[...,ii] = vec[...,ii] % (2*s)
            # enlarged modulo runs backwards
            np.putmask(vec[...,ii], vec[...,ii]>=s, (2*s-1)-vec[...,ii])
        else:
            raise ValueError("apply_boundary: boundaries are 'f', 't', 'e', 'p', or 'm'.")
        
    return vec
            
            



def sampleND(source, /, index=None, chunk=None, bound='f', fillvalue=0, strict=False):
    '''
    sampleND - better N-dimensional lookup, with switchable boundaries.
    
    sampleND looks up single values or neighborhoods in a source array.  You
    supply source data in reversed dimension order (...,Y,X) and index data
    as a collection of vectors pointing into the source.  The index is 
    collapsed one dimension by interpolation into source:  the contents along
    the final dimension axis are a vector indexing as [X,Y,...] a location
    in the source;  other prior axes in the index are broadcast.
    
    Each vector in index is replaced with data from source, using the nearest-
    neighbor method (integer rounding of the vector).  In the default case,
    a single value is returned.  Optionally you can use the "chunk" keyword
    to return a chunk of values from the source array, at each index location.
    If present, "chunk" must be either a scalar or an array matching the 
    size of the index parameter.  Each element gives the size of a subarray
    of values to be extracted from each indexed location in source.  Zero 
    values cause the corresponding dimension to be omitted.
    
    The current implementation uses NumPy's explicit array indexing scheme.  
    It assembles a direct, boundary-conditioned index into the source array
    for every point to be output (including chunks if called for), then 
    carries out the indexing operation and subsequently makes another pass to
    fix up truncation boundary conditions if necessary.  It is undoubtedly 
    quite slow compared to a Cython approach, but allows sampling and ranging 
    just like its Perl/PDL predecessor, range().
    
    As with most general-purpose code, it seems to spend most of its lines 
    parsing and conditioning the input.  The actual meat of the algorithm 
    is quite simple and short.
    
    Parameters
    ----------
    
    source : numpy.ndarray
        The source data from which to interpolate.
        
    index : numpy.ndarray
        The index data to collapse by interpolation from source.  The final 
        axis runs across dimensionality of the index vector into source. If
        the 'strict' flag is set (the default case) then the size of the
        final axis must match the number of dimensions in source.  If index
        has additional dimensions they are broadcast.
        
    chunk : list, array, or None
        Chunk specifies the size of additional axes to be appended to the end 
        of the axis list of the output.  If it is present, it must agree 
        dimensionally with the last axis of index.  This produces a small
        neighborhood in the vicinity of each sampled point.  The chunk
        extends upward in index from the sample point, along each chunked
        axis. If one of the chunk dimensions is 0, the corresponding axis is
        omitted from the output.
        
    bound : string or list (default 'forbid')
        This is either a string describing the boundary conditions to apply to 
        every axis in the input data, or a list of strings containing the
        boundary conditions on each axis.  The boundary conditions are listed
        in vector order, not index order -- so an image array containing
        values in (Y,X) order should be indexed in (X,Y) order in the index
        array, and boundary conditions are [boundX, boundY].  Only the first
        character of each string is checked.  The strings must be one of the
        strings accepted by apply_boundary():
            
            'f' - forbid boundary violations
            
            't' - truncate the array at the boundary.  Values outside are
                replaced with the fillvalue.
                
            'e' - extend the array at the boundary.  Values outside are 
                replaced with the nearest value in the source array.
                
            'p' - periodic boundary conditions apply. Indices are modded with
                the size of the corresponding axis in the source array.
                
            'm' - mirror boundary conditions apply.  Indices reflect off each
                boundary of the data, counting backward to the opposite 
                boundary (where they reflect again).
        
    strict : Boolean (default False)
        The 'strict' parameter forces strict matching of index vector dimension
        to the number of axes in the source array.  If it is False, then the
        indexing vectors may be smaller than the number of indices in Source, 
        in which case the interpolation is broadcast over the additional axes.
  

    Returns
    -------
    The indexed data extracted from the source array, as a numpy ND array
    '''
    if( not isinstance( index, np.ndarray ) ):
        index = np.array(index)
        
    if( not isinstance( source, np.ndarray ) ):
        source = np.array(source)
    
    if( strict and  len(source.shape) != index.shape[-1] ):
        raise ValueError("sampleND: source shape must match index size when strict flag is set")
            
    if(fillvalue is None):
        fillvalue = np.array([0])
    elif( not isinstance(fillvalue, np.ndarray)):
        fillvalue = np.array(fillvalue)
    
    
    # chunks are implemented by extending the index array.  The -1 axis has
    # the index vectors in it (in x,y,... order).  The strategy is to loop
    # over the passed-in chunk shape, adding new axes to the index as we go.
    # Because axis order is the opposite direction from vector component order,
    # we loop in reverse order. 
    if(chunk is not None):
        
        if( not isinstance(chunk,np.ndarray ) ):
            chunk = np.array(chunk)
            
        if(len(chunk.shape)==0):
            chunk = np.expand_dims( chunk, -1 )
        elif(len(chunk.shape) > 1):
            raise ValueError("sampleND: chunk must be a scalar or 1-D array")
        
        # Make sure the chunk size is an integer
        if not issubclass(chunk.dtype.type, np.integer):
            chunk = np.floor(chunk+0.5).astype('int')
            
        # Make sure no chunk sizes are negative (oops).  Zero is okay.
        if( any(chunk < 0) ):
            raise ValueError("sampleND: chunk size must be nonnegative")
        
        # Broadcast the chunk shape to the size of the index vector, if it's a scalar
        if(chunk.shape[0] == 1):
            chunk = chunk + np.zeros( index.shape[-1] )
            
        # Now extend the index along each requested size of the chunk, to 
        # return the requested chunk, starting at each indexed point.
        for ii in range( chunk.shape[0]-1, -1, -1 ):
            ch = chunk[ii]
            if(ch>0):
                # chunksize is greater than zero: insert the new dimension,
                # and increment the appropriate index along that dimension
                index = np.expand_dims( index, axis=-2 ) + np.zeros([int(ch), index.shape[-1]])
                index[...,ii] = index[...,ii] + np.mgrid[0:ch] 
    
    ## Convert to integer, and apply boundary conditions
    index = apply_boundary( index, source.shape[0:index.shape[-1]], bound=bound )
    
    ## Perform direct indexing.  Range() call reverses dim order, for
    ## Python standard (...,Y,X) indexing.  We have to add an ellipsis object
    ## to the start of the list after assembling it from the map.  Ick.
    dexlist = list( map ( lambda ii: index[...,ii], 
                          range(index.shape[-1]-1,-1,-1 ) 
                          ) 
                    )
    dexlist.insert(0,...)
    dextuple = tuple(dexlist)
    retval = source[dextuple]
    
    ## Truncation -- hardwire negative indices to the fill value
    ## All values should be in-range after apply_boundary, so anything negative
    ## is a truncation value.
    ## Note that dexlist has the vector index at the *start* (for the indexing
    ## operation), so the any operation happens along axis 0.
    if any( map ( lambda s:s[0]=='t', bound ) ):
        retval = np.where( np.any(dexlist[1:len(dexlist)]<np.array(0), axis=0), fillvalue, retval)
    
    return retval
    

def interpND(source, /, index=None, method='s', bound='f', fillvalue=0, strict=False):
    '''
    interpND - a better N-dimensional interpolator, with switchable boundaries.
    
    You supply source data in reversed dimension order (...,Y,X) and index
    data as a collection of vectors pointing into the source.  The index is
    collapsed one dimension by interpolation into source: the contents along
    the final dimension axis are a vector indexing as [X,Y,..] a location in 
    the source;  other prior axes in the index are broadcast.
    
    If source has more axes than the length of the index vectors, then those
    axes are broadcast: the return value of the interpolation is a collection
    of values in source rather than a single scalar value.
    
    if index has more axes than just the one axis, those additional axes are
    broadcast and end up at the beginning of the dimension list of the output.
    
    So if source has axes (Xi...X0) and index has axes (IXj...IX0,n) then
    the output has axes (IXj...IX0,Xi...X(i-n)).  
    
    There is a "strict" option, on by default, that requires the index array
    axis to have the same size as the number of source dimensions, so that
    the interpolation result is a single value for each index vector.

    Note that interpND indexes axes in reverse order: in the 2-D case of 
    image interpolation, the source array *axes* are treated as being in 
    (image-plane, Y, X) order and the 2-vector *values* in index are
    considered to be in (X,Y) order.   

    Parameters
    ----------
    
    source : numpy.ndarray
        The source data from which to interpolate.
        
    index : numpy.ndarray
        The index data to collapse by interpolation from source.  The final 
        axis runs across dimensionality of the index vector into source. If
        the 'strict' flag is set (the default case) then the size of the
        final axis must match the number of dimensions in source.  If index
        has additional dimensions they are broadcast.
        
    bound : string or list (default 'truncate')
        This is either a string describing the boundary conditions to apply to 
        every axis in the input data, or a list of strings containing the
        boundary conditions on each axis.  The boundary conditions are listed
        in vector order, not index order -- so an image array containing
        values in (Y,X) order should be indexed in (X,Y) order in the index
        array, and boundary conditions are [boundX, boundY].  Only the first
        character of each string is checked.  The strings must be one of:
            
            'f' - forbid boundary violations
            
            't' - truncate the array at the boundary.  Values outside are
                replaced with the fillvalue.
                
            'e' - extend the array at the boundary.  Values outside are 
                replaced with the nearest value in the source array.
                
            'p' - periodic boundary conditions apply. Indices are modded with
                the size of the corresponding axis in the source array.
                
            'm' - mirror boundary conditions apply.  Indices reflect off each
                boundary of the data, counting backward to the opposite 
                boundary (where they reflect again).
    
    fillvalue : scalar (default 0)
        This is the value used to fill elements that are outside the bounds of
        the source array, if the 'truncate' boundary condition is selected.
        
    method : string (default 'sample')
        Only the first character of the string is checked.  This controls the 
        interpolation method.  The character may be one of:
            
            's' - sample the nearest value of the array
            
            'l' - linearly interpolate from the hypercube surrounding each point
            
            'c' - cubic spline interpolation along each axis in order
            
            'f' - Fourier interpolation using discrete FFT coefficients
       
        
    strict : Boolean (default True)
        The 'strict' parameter forces strict matching of index vector dimension
        to the number of axes in the source array.  If it is False, then the
        indexing vectors may be smaller than the number of indices in Source, 
        in which case the interpolation is broadcast over the additional axes.

    Returns
    -------
    The indexed data extracted from the source array, as a numpy ND array

    '''
    if not isinstance(index,np.ndarray):
        index = np.array(index)
        
    if not isinstance(source, np.ndarray):
        index = np.array(index)
        
    # Sample: just grab the nearest value using sampleND
    if(method[0] == 's'):
        return sampleND(source, 
                        index=index, 
                        bound=bound, 
                        fillvalue=fillvalue)
    
    # Linear: grab the 2x...x2 hypercube containing the indexed point,
    # then assemble weighting coefficients based on closeness of the indexed
    # point to the corresponding corner of the hypercube.  (N-D version of the
    # usual alpha/beta mult-and-sum)
    elif(method[0] == 'l'):
        fldex = np.floor(index)
        # sample the ncube region around each point.  Dimensions of 
        # region are [ <index-broadcast-dims>, <N twos> ]
        # where N is the dimensionality of the index (that's what the chunk
        # parameter does)
        region = sampleND(source, 
                          index=fldex, 
                          bound=bound, 
                          fillvalue=fillvalue, 
                          strict=strict, 
                          chunk = np.array(
                              list( repeat( int(2), index.shape[-1]) )
                              )
                          )
        
        # Enumerate a hypercube with 0/1
        # Dimension is [<N twos>, index-vec-dim-size-N]
        ncube = np.mgrid[ tuple( repeat( range(2), index.shape[-1])) ].transpose()
        
        # Now assemble alpha/beta weighting coefficients
        # expand the index dims by adding ncube dimensions
        # alpha is [ <index-broadcast-dims>, <N 1's>, index-vec-dim-size-N ]
        # This lets it broadcast against ncube which has N 2's.
        # beta is the complement.
        alpha = np.expand_dims( index - fldex,
                                tuple( range( -2,
                                             -index.shape[-1]-2,
                                             -1
                                              ) 
                                      ) 
                                )
        beta = 1 - alpha
        
        # Now let the ncube corner coordinates (0 or 1) select alpha or or 
        # beta coefficients. weight gets a total weighting value for each 
        # corner of the ncube (which matches the chunk size in the output).
        # Take the product along the vector axis to get weight, which 
        # has dim [ <index-broadcast-dims>, <N twos> ] to match region.
        wvec = np.where( ncube, alpha, beta )
        weight = wvec.prod( axis=-1 )
        
        # finally... collapse by weighting-and-summing the region around each
        # index value.  The twos are gone and we have only the 
        # [ <index-broadcast-dims> ] left.
        value = (region * weight).sum( 
            axis=tuple( range( -1,
                               -index.shape[-1]-1,
                               -1
                               )
                       )
            )
        return value
    
    ## Cubic:  grab a hypercube around each indexed point and calculate
    ## a fittted cubic polynomial to it.
    
    elif(method[0]=='c'):
        fldex = np.floor(index)
        
        # sample the ncube region around each point.  Dimensions of 
        # region are [ <index-broadcast-dims>, <N fours> ]
        # where N is the dimensionality of the index (that's what the chunk
        # parameter does)
        region = sampleND(source, 
                          index=fldex - 1, 
                          bound=bound, 
                          fillvalue=fillvalue, 
                          strict=strict, 
                          chunk = np.array(
                              list( repeat( int(4), index.shape[-1]) )
                              )
                          )
        
        # Grab the subpixel offset, and expand it to be broadcastable to 
        # the new region.  At the end, alpha has dimension 
        # [<index-broadcast-dims>, <N ones>, N]
        b = np.expand_dims( index - fldex,
                                tuple( range( -2,
                                             -index.shape[-1]-2,
                                             -1
                                              ) 
                                      ) 
                                )
        
        # Now collapse by cubic polynomial interpolation, one dim at a time
        for ii in range(index.shape[-1]):
            # a0 gets just-under sample; 
            # a1 gets just-over sample;
            # a1_a0 gets slope in innermost pair
            a0 = region[...,1] 
            a1 = region[...,2] 
            a1_a0 = a1-a0

            # s0 gets average lower slope
            # a1 gets average upper slope
            s0 = (region[...,2]-region[...,0]) * 0.5
            s1 = (region[...,3]-region[...,1]) * 0.5
            
            # bb gets the correct vector component of b
            bb = b[...,0,ii]
            
            # now collapse the region by polyomial interpolation.  
            # Everything has the same dimensions as region, with 
            # one dim lopped off the end
            region = (
                        a0 +
                        bb * (
                            s0 +
                            bb * ( (3 * a1_a0 - 2*s0 - s1) +
                                   bb * (s1 + s0 - 2*a1_a0)
                                  )
                            )
                    )

            # b needs to be collapsed also, to match the collapse of 
            # region.  We just strip off one of the expanded dims, keeping
            # the vector axis at the end.
            b = b[...,0,:]
        
        # On exit from the loop, all the hypercube indices have been stripped
        # off of region, which now just has dimension [<index-broadcast>]
        return region
    
    # Fourier interpolation: find the Fourier components of the data, and 
    # explicitly evaluate them at the provided points
    elif(method[0]=='f'):

        raise AssertionError("interpND: Fourier interpolation is not yet implemented")        
    
    else: 
        
        raise ValueError(f"interpND: valid methods are 'sample', 'linear', or 'cubic' (got '{method}')")


        

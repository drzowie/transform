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
import itertools

def apply_boundary(vec, size, /, bound='f', rint=True, pixels=True):
    '''
    apply_boundary - apply boundary conditions to a vector or collection
    
    apply_boundary accepts input vector(s) and boundary conditions to apply
    on a suitably-dimensioned N-rectangular volume.  It can process generic
    floating-point vectors but by default it rounds its input to int,
    for use in indexing arrays.
    
    

    Parameters
    ----------
    vec : array
        Collection of vectors (in the final axis) to have boundaries applied
        
    size : list or array
        List containing the allowable size of each dimension in the index.  The
        allowable dimension is on the interval [0,size) (i.e. not including
        the corresponding value in size).  Note that size is in the same 
        (X,Y,...) order as the vector elements.  In this respect it is not the
        same as an array shape, which would be in the opposite order.
        
    bound : string or list of strings (default 'f')
        The boundary type to apply. Only the first character is checked.
        Each string must start with one of:
                        
            'f' - forbid boundary violations
            
            't' - truncate at the boundary.  Index values outside the range 
                are replaced with -1.
                
            'e' - extend the array at the boundary.  Index values outside the
                range are replaced with the nearest in-range value.
                
            'p' - periodic boundary conditions. Indices are modded with the 
                corresponding size.
                
            'm' - mirror boundary conditions apply.  Indices reflect off each
                boundary, counting backward to the opposite boundary (where 
                they reflect again).
                
    rint : Boolean (default True)
        Causes the vectors to be rounded and reduced to ints, for use in 
        indexing regular gridded data.  
        
    pixels: Boolean (default True)
        In the event that rint is false, the pixels flag causes the boundaries
        to be offset a half-pixel downward from their nominal position, i.e. 
        from [-0.5, size-0.5) along each axis.  This is consistent with the 
        behavior of rint, in the sense that the positions can be rinted 
        afterward to get integer pixel locations.  This is the
        intended behavior so the default is True.  Clearing the flag gives 
        boundaries for the active interval [0,size) along each axis.]

    Returns
    -------
    The index, with boundary conditions applied 
    '''
    if( (not isinstance(vec, np.ndarray)) ):
        raise ValueError("apply_boundary requires a np.ndarray")
        
    # Scalars are interpreted as a single 1-vector
    if( len(vec.shape) == 0 ):
        vec = np.expand_dims(vec,0)
        
    shape = vec.shape
    
    # Explicitly broadcast size if it's a scalar or a 1-element list
    if( not isinstance(size,(tuple,list,np.ndarray) )):
        size = [ size for i in range(shape[-1])]
    elif(len(size)==1):
        size = [ size[0] for i in range(shape[-1])]

    # Check that size matches the vec
    if(len(size) != shape[-1]):
        raise ValueError("apply_boundary: size vector must match vec dims")
    
    # Now do the same thing with the boundary string(s)
    if( not isinstance(bound, (tuple,list) ) ):
        bound = [ bound for i in range(shape[-1])]
    elif(len(bound)==1):
        bound = [ bound[0] for i in range(shape[-1])]
        
    # Check that boundary size matches the vec
    if(len(bound) != shape[-1]):
        raise ValueError("apply_boundary: boundary list must match vec dims")
        
    # rint if clled for
    if(rint):
        if issubclass(vec.dtype.type, np.integer):
            vec = copy.copy(vec)
        else: 
            vec = np.floor(vec+0.5).astype('int')
    else:
        if(pixels):
            vec = vec+0.5
        else:
            vec = copy.copy(vec)
    
    
    # Apply boundary conditions one dim at a time
    for ii in range(shape[-1]):
        b = bound[ii][0]
        s = size[ii]
    
        ## forbid conditions - throw an error on violation
        if    b=='f':
            if not  (np.all((vec[...,ii] >= 0) * (vec[...,ii] < s))):
                raise ValueError(
                  "apply_boundary: boundary violation with 'forbid' condition")
        ## truncate - set violating values to -1    
        elif  b=='t': 
            # Replace values outside the boundary with -1
            np.place(vec[...,ii], (vec[...,ii]<0)+(vec[...,ii]>=s), -1)
        ## extend - clip the vectors
        elif  b=='e':
            # Replace values outside the boundary with the nearest boundary
            # Use explicit place() because clip() doesn't do the right thing
            # at the upper boundary.
            np.place(vec[...,ii], (vec[...,ii]<0), 0)
            if(rint):
                np.place(vec[...,ii], (vec[...,ii]>= s), s-1)
            else:
                np.place(vec[...,ii], (vec[...,ii]>=s), s-1e-10)
        ## periodic - modulo works fine
        elif  b=='p':
            # modulo
            vec[...,ii] = vec[...,ii] % s
        ## mirror - modulo with reversal in every other instance
        elif  b=='m': 
            # modulo at twice the size
            vec[...,ii] = vec[...,ii] % (2*s)
            # enlarged modulo runs backwards
            np.putmask(vec[...,ii], vec[...,ii]>=s, (2*s-1)-vec[...,ii])
        else:
            raise ValueError(
                "apply_boundary: boundaries are 'f', 't', 'e', 'p', or 'm'.")
        
    if( (not rint) and pixels ):
        vec -= 0.5
        
    return vec
            
def sampleND(source, /, 
             index=None, 
             chunk=None, 
             bound='f', 
             fillvalue=0, 
             strict=False):
    '''
    sampleND - better N-dimensional lookup, with switchable boundaries.
    
    sampleND looks up single values or neighborhoods in a source array.  You
    supply source data in the usual reversed dimension order (...,Y,X) and 
    index data as a collection of vectors pointing into the source, with the
    vector components ordered as (X,Y,...).  The index is collapsed one 
    dimension by interpolation into source:  the contents along the final 
    dimension axis are a vector, indexing as [X,Y,...] a location
    in the source;  other prior axes in the index are broadcast.
    
    In the returned output array, each vector in the index is replaced with 
    data from the source array, using the nearest-neighbor method (integer 
    rounding of the vector).  In the default case, a single value is returned
    at each location (zero-dimensional).  
    
    Optionally you can use the "chunk" 
    keyword to return a chunk of values from the source array, at each index 
    location. If present, "chunk" must be either a scalar or an array matching
    the size of the index parameter.  Each element gives the size of a subarray
    of values to be extracted from each indexed location in source.  Zero 
    values cause the corresponding dimension to be omitted.
    
    The current implementation uses NumPy's explicit array indexing scheme.  
    It assembles a direct, boundary-conditioned index into the source array
    for every point to be output (including chunks if called for), then 
    carries out the indexing operation  -- and subsequently makes another pass 
    to fix up truncation boundary conditions if necessary.  
    
    Parameters
    ----------
    
    source : numpy.ndarray
        The source data from which to interpolate.
        
    index : numpy.ndarray
        The index data to collapse by interpolation from source.  The final 
        axis runs across dimensionality of the index vector into source. If
        the 'strict' flag is set (the default case) then the size of the
        final axis must match the number of dimensions in source.  If index
        has additional dimensions before the vector axis, the operation
        is broadcast across the earlier axes.  The result is that index
        is collapsed one dimension via lookup into the source.
        
    chunk : list, array, or None
        Chunk specifies the size of additional axes to be appended to the end 
        of the axis list of the output.  If it is present, it must agree 
        dimensionally with the last axis of index.  This produces a small
        neighborhood in the vicinity of each sampled point.  The chunk
        extends upward in index from the sample point, along each chunked
        axis. If one of the chunk dimensions is 0, the corresponding axis is
        omitted from the output.  For example indexing a BxA array with 
        an Nx2 "index" parameter and a "chunk" parameter of 3 yields an output
        that is Nx3x3, with the 3x3 containing an extracted chunk of the 
        "source" parameter, in (B-axis,A-axis) order, for each of the N 
        2-vectors in the "index".
        
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
                replaced with the fillvalue (default 0)
                
            'e' - extend the array at the boundary.  Values outside are 
                replaced with the nearest value in the source array.
                
            'p' - periodic boundary conditions apply. Indices are modded with
                the size of the corresponding axis in the source array.
                
            'm' - mirror boundary conditions apply.  Indices reflect off each
                boundary of the data, counting backward to the opposite 
                boundary (where they reflect again).
                
    fillvalue : float (default 0)
        This is a fill value used for points outside the source array, along 
        axes with the "truncate" boundary condition.
        
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
        raise ValueError(
            "sampleND: source shape must match index size when "
            "strict flag is set"
            )
            
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
            
        if( len(chunk.shape)==0 ):
            chunk = np.expand_dims( chunk, -1 )
        elif( len(chunk.shape) > 1 ):
            raise ValueError("sampleND: chunk must be a scalar or 1-D array")
        
        # Make sure the chunk size is an integer
        if not issubclass(chunk.dtype.type, np.integer):
            chunk = np.floor(chunk+0.5).astype('int')
            
        # Make sure no chunk sizes are negative (oops).  Zero is okay.
        if( any(chunk < 0) ):
            raise ValueError("sampleND: chunk size must be nonnegative")
        
        # Broadcast the chunk shape to the size of the index vector, if it's 
        # a scalar. There's probably a better way than adding zeroes to it --
        # but cost is microscopic.
        if(chunk.shape[0] == 1):
            chunk = chunk + np.zeros( index.shape[-1] )
            
        # Now extend the index along each requested size of the chunk, to 
        # return the requested chunk, starting at each indexed point.
        # We walk through the chunk axes in reverse order since the chunk 
        # is specified forward (X,Y,...) and shapes are backward (...,Y,X).
        for ii in range( chunk.shape[0]-1, -1, -1 ):
            ch = chunk[ii]
            if(ch>0):
                # chunksize is greater than zero: insert the new dimension,
                # and increment the appropriate index along that dimension
                # we expand at -2 since that's the last axis before the index
                # vector axis.  Then add more zeros to broadcast.
                index = ( np.expand_dims( index, axis=-2 ) + 
                          np.zeros([int(ch), index.shape[-1]])
                        )
                # Now add the offset index to the corresponding element 
                # of the index vector, along the new chunk axis.  
                index[...,ii] = index[...,ii] + np.mgrid[0:ch] 
    
    ## Convert to integer, and apply boundary conditions
    ## Size is extracted from source shape -- which is in reverse order of course
    index = apply_boundary( index, source.shape[-1:-index.shape[-1]-1:-1], bound=bound )
    
    ## Perform direct indexing.  Range() call reverses dim order, for
    ## Python standard (...,Y,X) indexing.  We have to add an ellipsis object
    ## to the start of the list after assembling it from the map.  Ick.
    dexlist = [ index[...,i] for i in range(index.shape[-1]-1,-1,-1) ]
    dexlist.insert(0,...)
    dextuple = tuple(dexlist)
    retval = source[dextuple]
    
    ## Truncation -- hardwire negative indices to the fill value
    ## All values should be in-range after apply_boundary, so anything negative
    ## is a truncation value.
    ## Note that dexlist has the vector index at the *start* (for the indexing
    ## operation), so the any operation happens along axis 0.  The range in
    ## dexlist starts at 1 instead of 0, to skip over the ellipsis object that
    ## got inserted just above.
    if any( map ( lambda s:s[0]=='t', bound ) ):
        retval = np.where( np.any(dexlist[1:len(dexlist)]<np.array(0), axis=0), fillvalue, retval)
    
    return retval
    

def interpND(source, /, 
             index=None, 
             method='n', 
             bound='t', 
             fillvalue=0,
             oblur=None,
             strict=False):
    '''
    interpND - a better N-dimensional interpolator, with switchable boundaries.
    
    You supply source data in reversed dimension order (...,Y,X) and index
    data as a collection of vectors pointing into the source.  The index is
    collapsed one dimension by interpolation into source: the contents along
    the final dimension axis are a vector that points (as [X,Y,..]) to a 
    location in the source;  other prior axes in the index are broadcast and 
    allow indexing a collection of locations simultaneously.
    
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

    Note that interpND indexes axes in reverse order: for example, in the 2-D 
    case of image interpolation, the source array *axes* are treated as being 
    in (image-plane, Y, X) order and the 2-vector *values* in index are
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
        
    bound : string or list or tuple (default 'truncate')
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
    
    fillvalue : float (default 0)
        This is the value used to fill elements that are outside the bounds of
        the source array, if the 'truncate' boundary condition is selected.
        
    method : string (default 'nearest')
        This controls the interpolation method.  The value is a string.  Each
        method has a single-letter abbreviation:
            
            'nearest'/'n' - nearest-value interpolate (sample) the array.
            
            'linear'/'l' - linearly interpolate from the hypercube surrounding 
                each point.
            
            'cubic'/'c' - cubic-spline interpolate from the 4-pixel hypercube 
                around each point.  Cubic splines reproduce the value and the 
                first and second derivatives of the data at original pixel 
                centers. 
                
            'sinc'/'s' - use Sinc-function weighting in the input plane; this
                corresponds to a sharp cutoff at the Nyquist frequency of the
                input plane.  The sinc function falls off only very slowly,
                so the input is enumerated over 6 pixels in all directions.
                Because the sinc function has zeroes at integer pixel offsets,
                this reproduces the original value at pixel centers.
            
            'lanczos'/'z' - Use Lanczos-function weighting in the input plane; 
                this is a modified sinc that rolls off smoothly over 3 pixels.  
                It is equivalent to an approximate trapezoid filter in the 
                frequency domain. Like the sinc function, the Lanczos filter 
                has zeroes at integer pixel offsets, so it reproduces the 
                source data when evaluated at pixel centers.
                
            'hann'/'h' - Use Hann window (overlapping cos^2) interpolation; the 
                kernel is enumerated for 1 full pixel in all directions.  The
                Hanning function produces smooth transitions between pixels, but
                introduces ripple for smoothly varying curves.
                
            'tukey'/'t' - use Tukey window (cos^2 rolloff) with alpha=0.5;
                yields a flat center to each pixel, with rounded transitions. 
                The result is smoother than sampling, but preserves vestiges 
                of pixel edges.  It can be useful for rendering pixelated data 
                and leaving pixel edges both visible and unobtrusive.
            
            'gaussian'/'g' - Use Gaussian weighted smoothing with 1 pixel FW; 
                the kernel is enumerated for 3 pixels in all directions. Note 
                that this method does not guarantee the value of integer-
                indexed samples will match the value in the array itself.
            
            'fourier'/'f' - fourier interpolate using discrete FFT coefficients; 
                this is useful for periodic data such as wave patterns.  Note,
                this involves taking the FFT of the entire input dataset, which
                is then discarded -- therefore this method benefits strongly
                from vectorization.  Because of the way Fourier interpolation is
                implemented (via explicit evaluation of a complex exponential)
                you can do Laplacian analytically continued "interpolation" also, 
                by feeding in complex-valued coordinates.
                
    oblur : Float or None (default None)
        If set, this parameter scales the width of the filter function on
        filter interpolation, which is most of the methods.  A value of 1.0 
        does nothing.  A value of greater than 1.0 yields blur in the output
        space.  Values smaller than 1.0 are allowed but not recommended: they
        may do strange things to the output, and very small values may crash
        the code. Values different from 1.0 break alignment properties of some 
        filters (e.g. the sinc and Lanczos filters).
        
    strict : Boolean (default True)
        The 'strict' parameter forces strict matching of index vector dimension
        to the number of axes in the source array.  If it is False, then the
        indexing vectors may be smaller than the number of indices in Source, 
        in which case the interpolation is broadcast over the additional axes.

    Returns
    -------
    The indexed data extracted from the source array, as a numpy ND array

    '''
    meth = method[0]
    if(method=='lanczos'):
        meth = 'z'
    if(method=='hamming'):
        meth = 'm'
        
    if not isinstance(index,np.ndarray):
        index = np.array(index)
        
    if not isinstance(source, np.ndarray):
        index = np.array(index)
        
    # Sample: just grab the nearest value using sampleND
    if(method[0] == 'n' or method[0]=='nearest'):
        if(oblur is not None):
            raise ValueError("interpND: oblur is not allowed with nearest-neighbor interpolation.")
            
        return np.array(sampleND(source, 
                        index=index, 
                        bound=bound, 
                        fillvalue=fillvalue))
    
    # Linear: grab the 2x...x2 hypercube containing the indexed point,
    # then assemble weighting coefficients based on closeness of the indexed
    # point to the corresponding corner of the hypercube.  (N-D version of the
    # usual alpha/beta mult-and-sum)
    # 
    # linear is implemented differently than the other filters because 
    # it is simpler to broadcast.  This makes it slightly faster, which 
    # is a win since people seem to reach for it alla time.
    elif(meth == 'l' and (oblur is None)):
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
        
        # Enumerate an N-cube with 0/1 on all corners
        # Dimension is [<N twos>, index-vec-dim-size-N]
        ncube = np.mgrid[ tuple( repeat( range(2), index.shape[-1])) ].T
        
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
        
        # Now let the ncube corner coordinates (0 or 1) select alpha or 
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
        
    ############################
    ## Fourier interpolation: find the Fourier components of the data, and 
    ## explicitly evaluate them at the provided points.
    ## 
    ## Collapsing this type of interpolation one dimension at a time would be 
    ## inefficient, it is broken out into its own thang.
    elif(meth=='f'):

        if(oblur is not None):
            raise ValueError("interpND: oblur is not allowed with Fourier interpolation.")
            
        # FFT the source along all (and only) the axes used by the index.
        sourcefft = np.fft.fftn(source,
                                axes=range( -1, -1 - index.shape[-1], -1 )
                                )
        
        # Make a vector of frequency along each axis.  Index mgrid with a 
        # collection of ranges running over each source axis.  Shape is:
        #   [ <useful-index-dims>, <index-N> ]
        freq = np.mgrid[ tuple(
                               map( lambda i:range(0,source.shape[i]), 
                                 range( -1, -1 - index.shape[-1], -1 )
                               )
                            )
            ].T.astype(float)
        # convert from 0:n to 0:2PI along each axis
        for i in range(index.shape[-1]):
            freq[...,i] = freq[...,i] * np.pi * 2 / freq.shape[-2-i]
            
        # Now freq has shape [<useful-source-dims>,index-N], and contains the
        # angular rate (in radians/pixel-value) for each location in the 
        # sourcefft.  We have to get it to broadcast with index, which
        # has shape [<index-bc-dims>,index-N]. To do this, pad index into bcdex 
        # with shape:
        #   [<index-bc-dims>,<1s-for-useful-source-dims>,<index-N>]
        bcdex = np.expand_dims( index, tuple( range( -2,-2-index.shape[-1],-1 )))
        
        # Now generate the overall phase and the Fourier basis values,
        # with size [<index-bc-dims>,<useful-source-dims>].  
        # The overall Fourier phase is just the sum of the phases introduced
        # along each axis of the source -- so we do the sum explicitly.  The 
        # Fourier basis elements are just complex exponentials of the phase.
        phase = (bcdex * freq).sum(axis=-1)
        basis = np.exp( 1j * phase )
        
        # Now it's all over but the shouting.  We want to sum coefficients
        # in the sourcefft, but it may have additional axes "along for the
        # ride" -- so we have to pad the source with additional axes between
        # those full-broadcast axes and the index broadcast dims.  bcsourcefft
        # gets the shape:
        #  [ <source-bc>, <1s-for-index-bc-dims>, <useful-source-dims> ]
        bcsourcefft = np.expand_dims( sourcefft, 
                        tuple( range(-1-index.shape[-1],
                                     -1-index.shape[-1]-(len(index.shape)-1),
                                     -1
                                    )
                             )
                        )
        
        # Now collapse the useful-source-dims out of result by summing over
        # all Fourier coefficients multiplied times the calculated value
        # of the corresponding basis element.  This carries out the Fourier 
        # sum at each location.  We collapse by mean and not by sum, because
        # this an *inverse* Fourier transform to get back to the spatial 
        # domain.  The final shape of result is just:
        #   [ <source-bc>, <index-bc-dims> ]
        # which is what we want to return.
        result = (bcsourcefft * basis).mean( 
                            axis = tuple(
                                range( -1, 
                                       -1 - index.shape[-1],
                                       -1
                                       )
                                )
                            )
        # Now check for complex indices and/or source.  If both are real 
        # then return the real part only (which loses some information but 
        # best approximates what the user probably wants).
        complexes = (np.dtype('complex64'), np.dtype('complex128'))
        if( not ((source.dtype in complexes) or (index.dtype in complexes) )):
            result = result.real
                
        return result
        
    #####################
    ## Collapsible filter interpolation
    ## This implements cubic-spline, sinc, Lanczos, Gaussian, Hann, and rounded
    ## interpolation, which differ in the size of region they sample
    ## and also in the actual formula.
                                                  
    elif(meth in ('l','c','s','z','g','h','t')):
        fldex = np.floor(index)
        
        # Different methods have different region sizes that get sampled 
        # around each point.
        size = {'l':2, 'c':4, 's':16, 'z':6, 'g':8, 'h':2, 't':2}[method]
        
        if(oblur is not None):
            size = (2 * np.ceil(size/2 * oblur)).astype(int)
            
        offset = int(size/2)-1
        
        # sample the ncube region around each point.  Dimensions of 
        # region are [ <index-broadcast-dims>, <N size's> ]
        # where N is the dimensionality of the index (that's what the chunk
        # parameter does)
        region = sampleND(source, 
                          index=fldex - offset, 
                          bound=bound, 
                          fillvalue=fillvalue, 
                          strict=strict, 
                          chunk = np.array(
                              list( repeat( int(size), index.shape[-1]) )
                              )
                          )
        
        # Grab the subpixel offset, and expand it to be broadcastable to 
        # the new region.  At the end, b has dimension 
        # [<index-broadcast-dims>, <N ones>, N]
        b = np.expand_dims( index - fldex,
                                tuple( range( -2,
                                             -index.shape[-1]-2,
                                             -1
                                              ) 
                                      ) 
                                )
        
        ### Now collapse one dim at a time according to method
        
        #######
        ## Cubic spline interpolation
        ## This uses a formula that is more complex than a simple linear-
        ## response filter function, so it gets broken out from the filters
        ## (below).
        if(meth=='c'):
            if(oblur is not None):
                raise ValueError("interpND: oblur is not allowed with cubic-spline interpolation.")
            
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
    
        # The other methods (linear, sinc, lanczos, gaussian, hann, hamming, 
        # and tukey) each use a tailored filter function averaged over the ROI 
        # defined up above.
        elif(meth in ('l','s','z','g','h','t')):
  
            if(oblur is None):
                oblur = 1.0
            
            # of gets the offset of each pixel in the sampled subregion, 
            # relative to the requested location.  This requires indexing 
            # mgrid with an asssembled tuple of ranges.  
            of = ( (1 - b) - (offset + 1) +
                  np.mgrid[ tuple( map( lambda i:range(size), range(index.shape[-1]) ))].transpose()
            ) / oblur
            
            # Now loop over axes, collapsing them in turn until we get to the 
            # size we need
            for ii in range(index.shape[-1]):
                bb = of[...,ii]

                if(meth=='s'):     ## sinc
                    k = np.sinc(bb)

                elif(meth=='z'):   ## lanczos
                    bb = np.clip(bb,-3,3)
                    k = 3 * np.sinc(bb) * np.sinc(bb/3)

                elif(meth=='g'):   ## Gaussian
                    k = np.exp( - bb * bb / 0.5 / 0.5 )

                elif(meth=='h'):   ## Hann (no need to divide by 2 since we normalize by k)
                    k = (1 + np.cos( np.pi * np.clip(bb,-1,1) ) )
                    
                elif(meth=='t'):   ## Tukey (cosine rollof over half a pixel)
                    k = (1 + np.cos( np.pi * np.clip( np.abs(bb) * 2-0.5, 0, 1)))
                    
                elif(meth=='l'):  ## Generalized linear (tent function)
                    k = 1 - np.clip(np.abs(bb),0,1)
                        
                else:
                    raise AssertionError("This can't happen")
                
                region = ( k * region ).sum(axis=-1) / k.sum(axis=-1)
                
                of = of[...,0,:]
                
            return region

        raise AssertionError("This shouldn't ever happen")
 
        
    else: 
        
        raise ValueError(
            "interpND: valid methods are ('n','l','c','f','s','z','g','h','t')"
            )

def simple_jacobian(index):
    '''
    simple_jacobian - given an N-D grid of N-vectors, find the numeric Jacobian 
    of the implied transformation (delta(grid-vec) / delta(pixel-loc)).  The 
    simple jacobian is just the enumerated difference between adjacent vectors,
    in each possible direction.  The size of the grid is shrunk by one in each
    direction, since the simple Jacobian exists halfway between gridpoints.  
    The derivatives along each axis are averaged over the 2**(N-1) grid edges 
    parallel to that axis (4 values for a 3D Jacobian; 2 values for a 2D 
    Jacobian).
    
    The calculation is is not broadcastable (i.e. the index variable must be
    N+1-D and its last axis must have size N), although the algorithm could
    easily be adapted to support broadcasting.
    
    Parameters
    ----------
    index : NumPy array, with N+1 axes, the last of which has size N.
        The parameter is treated as an N-D array of N-vectors. The axes are
        in reverse order (...,Y,X) and the vector is in forward order 
        (X,Y,...).

    Returns
    -------
    the Jacobian matrix at each pixel boundary in the array, as an 
    array with N+2 axes, the last two of which have size N.  The first
    N axes are shrunk by one element compared to the input.  Indexing
    the output with [...,i,j] yields the ith component of the derivative
    vector with respect to the jth direction.
    '''
    if(np.any(np.array(index.shape[0:-1]) < 2)):
        raise ValueError("SimpleJacobian: must have at least two vectors along each grid axis")
        
    ndim = index.shape[-1]
    
    if(ndim != len(index.shape)-1 ):
        raise ValueError("SimpleJacobian: vector dimension must match broadcast axes")

    # Allocate the Jacobian.  Don't bother initializing since all the values
    # are computed and inserted later.
    Jdims = [index.shape[i]-1 for i in range(ndim)] + [ndim,ndim]
    J = np.ndarray( Jdims )
 
    #########################################################
    # 1-D case: trivial
    if(ndim==1):
        J[...,:,0] = index[1:index.shape[0]]-index[0:-1]
    
    #########################################################
    # 2-D case: not quite so trivial but directly enumerable
    elif(ndim==2):
        J[...,:,0] = ( + index[1:index.shape[0],1:index.shape[1],:]
                       - index[1:index.shape[0],0:-1,            :]
                       + index[0:-1,            1:index.shape[1],:]
                       - index[0:-1,            0:-1,            :]
                       )/2
        J[...,:,1] = ( + index[1:index.shape[0],1:index.shape[1],:]
                       + index[1:index.shape[0],0:-1,            :]
                       - index[0:-1,            1:index.shape[1],:]
                       - index[0:-1,            0:-1,            :]
                       )/2   
    
    #########################################################          
    # N-D case: we assemble one column at a time.  We reduce the offset 
    # operation to a multiply-and-sum that produces the mean along each axis 
    # except the one we care about for this loop iteration.
    else:
        
        # ncube: N-cube of vector offsets with 0/1 on all corners.
        # ncube_enum: flattened collection of vectors
        ncube = np.mgrid[ tuple( repeat( range(2), ndim))].T
        ncube_enum = np.reshape(ncube,(2**ndim,ndim))


        # Set all elements of J to 0, so we can accumulate sums in it.
        J[...,:] = 0
    
        # Loop over column of the Jacobian 
        for col in range(ndim):
        
            # Generate an N-cube similar to the ncube above, but 
            # of scalar coefficients.  The coefficient should be 
            #  1 in the upper and -1 on the lower side of the 
            # axis we care about for this column.  The axes count
            # backward compared to column, because of the (...,Y,X)
            # indexing in numpy.  The 0 index after the ellipsis
            # selects the lower index.
            factors = np.ones( tuple( repeat( 2, ndim ) ) )
            factors[ tuple( [..., 0 ] + 
                            [ np.s_[:] for i in range(col)]
                            )
                    ] = -1

            # Dividing by 2**(n-1) yields the appropriate mean with 
            # a multiply-and-sum
            factors /= 2**(ndim-1)
            factors_1d = np.reshape(factors,2**ndim)
        
            for corner in range(2**ndim):
                ncv = ncube_enum[corner]            
                # Soooo cumbersome.  Grab each trimmed-by-one-in-all-directions
                # slice of index, and multiply by corresponding factor to get the
                # term in the mean-of-differences sum.  (PDL does this more 
                # elegantly with its range() -- we could do that here with a 
                # sampleND call -- but it would be very very slow)
                rangedex = tuple( 
                      [ np.s_[ ncv[i] : ncv[i]+Jdims[i]] for i in range(ndim-1,-1,-1) ]
                    + [ range(ndim) ]
                    )
                corner_term = index[ rangedex ] * factors_1d[corner]
                J[...,col] += corner_term
    
    return J

def jump_detect(Js, 
                jump_thresh = 10
                ):
    '''
    jump_detect - find discontinuities in a Jacobian grid.
    
    You supply a grid of Jacobians that represent a coordinate transform. 
    
    The jump detector uses the typical magnitude (distance from determinant) 
    of the Jacobian offset vector:
        Jmag2 = sum_j sum_i (J_ij**2).
    The value of M2 is calculated around the entire 3x...x3 neighborhood
    of each simple_jacobian value, and the 33 percentile value is kept.
    Anywhere that
        Jmag2 > Jmag2_33pct * jump_detect  
    is marked as a jump.  The  Jacobian's components there are 
    replaced with the mean of those from nearby non-marked locations.
        
    The average-magnitude-of-Jacobian method works okay because the 
    typical use case is for pixel to pixel mapping (reampling data 
    sets), where the overall singular value ratio is not likely to be 
    large (less than, say, 30).
    
    The method currently considers only the non-boundary case and single
    boundaries:  i.e. in a 2-D grid the general case and edges are handled.
    Corners are ignored.  In a 3-D grid the general case and faces are
    handled; edges and corners are ignored.
    
    The return value is a scalar grid that contains a flag: 0 at most
    locations; 1 at locations with jumps.

    Parameters
    ----------
    Js : Numpy array 
        Js is a an N-D grid of Jacobians.  It must have dimension N+2, and
        each of the last axes must have size N.
        
    jump_thresh : float, optional
        This is the size of step (normalized to the neighborhood median in the 
        Jacobian) that is considered a jump. The default value of 10 means that 
        a lone increase in offset magnitude by a factor of 10 (compared to the 
        5-step neighborhood mdian in all directions)

    Returns
    -------
        An N-D array containing 1 where the Jacobian is okay, and 0 where it
        contains a jump.
    '''
    
    ndim = len(Js.shape)-2
    if(Js.shape[-1] != ndim or Js.shape[-2] != ndim):
        raise ValueError("jump_detect: Jacobian-grid input has inconsistent dims")
    
    jumpflag = np.zeros(Js.shape[0:-2],dtype=int) + 1
    

    # Calculate the "typical magnitude" of the offset vector in the Jacobian
    Jmag2 = (Js*Js).sum(axis=(-2,-1))
    jump_thresh2 = jump_thresh * jump_thresh
    
    # 1-dimensional case: trivial.  Boundaries are ignored.
    if(ndim==1):
        # Find lags along the lone axis, to characterize each neighborhood
        # (in the final axis)
        Jmag2lag = np.stack( 
            [ Jmag2[i:i+Js.shape[0]-2] for i in range(3) ], 
            axis=-1) 
        Jmag2lag.sort(axis=-1)
        Jmag2_33pct = Jmag2lag[...,1]
        jumpflag[1:-1] = (Jmag2[1:-1] < Jmag2_33pct * jump_thresh2)
        
        jumpflag[0] = (Jmag2[0]   <= Jmag2[1] * jump_thresh2 )
        jumpflag[-1] = (Jmag2[-1]  <= Jmag2[-2] * jump_thresh2 )
        
       
    # 2-dimensional case: nearly trivial.  One general case and 
    # four edges.
    elif(ndim==2):
        # Find lags along both axes to characterize each neighborhood.
        # Then treat the boundaries independently.  Dual-boundaries
        # (corners) are unimportant; ignore them.
        Jmag2lag = np.stack(
               [ Jmag2[
                   i:i+Js.shape[0]-2,
                   j:j+Js.shape[1]-2
                   ]
                   for i in range(3) for j in range(3)
                   ],
               axis=-1)
        Jmag2lag.sort(axis=-1)
        Jmag2_33pct = Jmag2lag[...,3]
        jumpflag[1:-1,1:-1] = (Jmag2[1:-1,1:-1] <= Jmag2_33pct * jump_thresh)
            
        # Low-Y edge:
        Jmag2lag = np.stack(
            [ Jmag2[
                i:i+1,
                j:j+Js.shape[1]-2
                ]
                for i in range(2) for j in range(3) 
                ],
            axis=-1)
        Jmag2lag.sort(axis=-1)
        Jmag2_33pct = Jmag2lag[...,2]
        jumpflag[0:1,1:-1] = (Jmag2[0:1,1:-1] <= Jmag2_33pct * jump_thresh)
            
        # High-Y edge:
        Jmag2lag = np.stack(
            [ Jmag2[
                i+Js.shape[0]-2:i+Js.shape[0]-1,
                j:j+Js.shape[1]-2
                ]
                for i in range(2) for j in range(3) 
                ],
            axis=-1)
        Jmag2lag.sort(axis=-1)
        Jmag2_33pct = Jmag2lag[...,2]
        jumpflag[-1:, 1:-1] = ( Jmag2[-1:, 1:-1]  <= Jmag2_33pct * jump_thresh)
            
        # Low-X edge:
        Jmag2lag = np.stack(
            [ Jmag2[
                j:j+Js.shape[0]-2,
                i:i+1,
                ]
                for i in range(2) for j in range(3) 
                ],
            axis=-1)
        Jmag2lag.sort(axis=-1)
        Jmag2_33pct = Jmag2lag[...,2]
        jumpflag[1:-1,0:1] = (Jmag2[1:-1,0:1] <= Jmag2_33pct * jump_thresh)
            
        # High-Y edge:
        Jmag2lag = np.stack(
            [ Jmag2[
                j:j+Js.shape[0]-2,
                i+Js.shape[1]-2:i+Js.shape[1]-1,
                ]
                for i in range(2) for j in range(3) 
                ],
            axis=-1)
        Jmag2lag.sort(axis=-1)
        Jmag2_33pct = Jmag2lag[...,2]
        jumpflag[1:-1, -1: ] = ( Jmag2[1:-1, -1:] <= Jmag2_33pct * jump_thresh)
        
    else:
        # Generate a 3x3x...x3 N-cube to enumerate the lags
        ncube = np.mgrid[ tuple( repeat( range(3), ndim))].T
        ncube_enum = np.reshape(ncube, (3**ndim, ndim))
        
        # Generate the lags for the general case
        Jmag2lags = np.stack(
            [ Jmag2[
                tuple(
                    [ np.s_[ ncube_enum[i,j] : ncube_enum[i,j] + Js.shape[j]-2 ] 
                     for j in range(ndim)    
                     ]
                    )
                ] 
                for i in range( ncube_enum.shape[0] )
            ],
            axis=-1
            )
        Jmag2lags.sort(axis=-1)
        Jmag2_33pct = Jmag2lags[...,int(ncube_enum.shape[0]/3+0.5)]
        
        inset_by_one = tuple(
            [ np.s_[1:-1] for i in range(ndim) ]
            )
        
        jumpflag[ inset_by_one ] = (
            Jmag2[ inset_by_one ] <= Jmag2_33pct * jump_thresh 
            )
        
        # Now handle boundary case.  For N dimensions there are 2**N 
        # major boundaries (e.g. faces, in 3D).  Neglect the minor
        # boundaries (boundary intersections).

        # Generate a 2x3x3x...x3 (N)-rectangle to average over 
        ncube = np.mgrid[ tuple( [range(2)] + [range(3) for i in range(1,ndim)])].T
        ncube_enum = np.reshape(ncube, (2 * (3**(ndim-1)), ndim))
        
        # Loop over the axes, getting lower and upper bounds on each.
        # Move the target axis to 0 position, then enumerate the hypercube on
        # it.
        for axis in range(ndim):
            # Move appropriate axis to start, in jf_tmp (view on jumpflag)
            jf_tmp = np.moveaxis(jumpflag,axis,0)
            Jmag2_tmp = np.moveaxis(Jmag2,axis,0)
            
            # Same offset construction as for major case -- except that on 
            # the boundary axis we take just a 1-pixel slice at value=0
            Jmag2lags = np.stack( [
                Jmag2_tmp[
                    tuple(
                        [ np.s_[ ncube_enum[i,0] : ncube_enum[i,0]+1 ] 
                         ] +
                        [ np.s_[ ncube_enum[i,j] : ncube_enum[i,j] + jf_tmp.shape[j]-2 ]
                           for j in range(1,ndim) 
                        ]    
                        )
                    ]
                    for i in range(ncube_enum.shape[0])
                ],
                axis=-1
                )
            Jmag2lags.sort(axis=-1)
            Jmag2_33pct = Jmag2lags[...,int(ncube_enum.shape[0]/3+0.5)]
            #jf_tmp is a view on jumpflag - flows back
            jf_tmp[ tuple(
                [ np.s_[ 0:1 ] ] +
                [ np.s_[ 1:-1 ] for j in range(1,ndim) ]
                )
                ] = Jmag2_tmp[ tuple(
                    [ np.s_[ 0:1 ] ] +
                    [ np.s_[ 1:-1 ] for j in range(1,ndim) ]
                    )
                    ] <= Jmag2_33pct * jump_thresh
            
            # Same construction as above -- but offset to max value instead of 0.
            Jmag2lags = np.stack( [
                Jmag2_tmp[
                    tuple(
                        [ np.s_[ ncube_enum[i,0] + jf_tmp.shape[0]-2 : ncube_enum[i,0] + jf_tmp.shape[0] - 1 ] 
                         ] +
                        [ np.s_[ ncube_enum[i,j] : ncube_enum[i,j] + jf_tmp.shape[j]-2 ]
                           for j in range(1,ndim)
                        ]
                        )
                    ]
                    for i in range(ncube_enum.shape[0])
                ],
                axis=-1
                )
            Jmag2lags.sort(axis=-1)
            Jmag2_33pct = Jmag2lags[...,int(ncube_enum.shape[0]/3+0.5)]
            #jf_tmp is a view on jumpflag - flows back
            jf_tmp[ tuple(
                [ np.s_[ jf_tmp.shape[-1]-1: jf_tmp.shape[-1] ] ]+
                [ np.s_[ 1:-1 ] for j in range(1,ndim) ]
                )
                ] = Jmag2_tmp[ tuple(
                    [ np.s_[ jf_tmp.shape[-1]-1: jf_tmp.shape[-1] ]] +
                    [ np.s_[ 1:-1 ] for j in range(1,ndim) ]
                     )
                    ] <= Jmag2_33pct * jump_thresh
        
            
            
        
        
                                
        
        
        

                
        
    return jumpflag

        
        
def jacobian(index, 
             jump_detect=True,
             jump_thresh=10.
             ):
    '''
    jacobian - given an N-D grid of N-vectors, find the conditioned numeric 
    Jacobian of the implied transform (delta(grid-vec) / delta(pixel-loc)).  
    The Jacobian is assembled from the enumerated difference between adjacent 
    vectors, in each possible direction.  The Jacobian is estimated at 
    gridpoints, using the simple_jacobian as a starting point.  If the 
    jump_detect flag is set, then jumps are detected and eliminated, yielding
    a value that is not strictly a numeric Jacobian but an estimate of the 
    underlying Jacobian (ignoring jumps). The jump detection happens *before*
    the averaging over each neighborhood to grid-center the Jacobian.
    
    This routine is a great candidate for being dropped into C: the memory
    accesses break cache in a bad way.

    Parameters
    ----------
    index : NumPy array, with N+1 axes, the last of which has size N.
        The parameter is treated as an N-D array of N-vectors. The axes are
        in reverse order (...,Y,X) and the vector is in forward order 
        (X,Y,...).
    
    jump_detect : Boolean, optional
        If this flag is set, then jump detection is included, to try to 
        identify discontinuities in the underlying transformation and sidestep
        them by extrapolation from valid points nearby.
        
        If set, this uses the jump_detector subroutine to do its dirty work.
    
        The default is True.
        
    jump_thresh : Float, optional
        If jump_detect is set, this is the size of step (normalized to the 
        neighborhood median in the Jacobian) that is considered a jump.
        The default value of 10 means that a lone increase in offset magnitude
        by a factor of 10 (compared to the 5-step neighborhood mdian in all
        directions)

    Returns
    -------
    the Jacobian matrix at each pixel boundary in the array, as an 
    array with N+2 axes, the last two of which have size N.  The first
    N axes are have the same size as the corresponding axes on input.  Indexing
    the output with [...,i,j] yields the ith component of the estimated 
    derivative vector with respect to the jth direction.
    '''
    Js = simple_jacobian(index)
    
    ndim = index.shape[-1]
    if(ndim != len(index.shape)-1 ):
        raise ValueError("jacobian: input must be N+1-dimensional, with last axis size N")
    
    J = np.ndarray( tuple(list(index.shape) + [index.shape[-1]]) )

    # For jump detection we produce a validity flag (1/0) in jumpflag.
    # Also set the corresponding Js values to 0.  This allows "simple" 
    # averaging of neighborhoods in most cases.
    if(jump_detect):
        jf=jump_detect(Js)
        Js = np.where(jf.T, Js.T, np.array([0])).T
   
    # 1-D: stoopid
    if(ndim==1):
        # Direct 1-D averaging (no jump detection)            
        J[1:-1,...] = Js[ 0:Js.shape[0]-1,...] + Js[ 1:Js.shape[0],... ]
            
        if(jump_detect):
            wgt =         jf[ 0:Js.shape[0]-1]     + jf[ 1:Js.shape[0]:-2  ]
            J[1:-1,...].T /= wgt.clip(1)
        else:
            J /= 2
                
        # copy over boundaries
        J[ 0, ...] = J[ 1, ...]
        J[-1, ...] = J[-2, ...]

    # 2-D: slightly less stoopid but manageable            
    elif(ndim==2):
        # Direct 2-D averaging, plus four boundaries
        J[1:-1,1:-1,...] = ( Js[ 0:Js.shape[0]-1, 0:Js.shape[1]-1,...] +
                             Js[ 1:Js.shape[0],   0:Js.shape[1]-1,...] +
                             Js[ 0:Js.shape[0]-1, 1:Js.shape[1],  ...] +
                             Js[ 1:Js.shape[0],   1:Js.shape[1],  ...]
                             )
        
        if(jump_detect):
            wgt          = ( jf[ 0:Js.shape[0]-1, 0:Js.shape[1]-1] +
                             jf[ 1:Js.shape[0],   0:Js.shape[1]-1] +
                             jf[ 0:Js.shape[0]-1, 1:Js.shape[1]  ] +
                             jf[ 1:Js.shape[0],   1:Js.shape[1]  ]
                            )
            J[1:-1,1:-1,...].T /= wgt.clip(1)
        else:
            J[1:-1,1:-1,...] /= 4

        # Copy over boundaries
        J[ 0, ... ] = J[ 1, ... ] 
        J[-1, ... ] = J[-2, ... ]
        J[ :,  0, ...] = J[ :,  1, ... ]
        J[ :, -1, ...] = J[ :, -2, ... ]
        
    # All other cases: use a general algo
    else:
        ndim = index.shape[-1]
        
        # Generate an N-cube of size 2 to use in assembling lags
        ncube = np.mgrid[ tuple( repeat( range(2), ndim ) )].T
        ncube_enum = np.reshape(ncube,(2**ndim, ndim))

        # Use stupid N-pass algorithm to avoid large memory suck for the 
        # sum.  In C or Cython we'd use a local algorithm.          
        J[...,:,:] = 0
            
        if(jump_detect):
            wgt = np.zeros(Js.size[0:ndim]-1)
        
        # Assemble mean where possible
        croptuple = tuple( [ np.s_[1:-1] for i in range(ndim)] )
        
        for corner in range(2**ndim):
            ncv = ncube_enum[corner]
            rangedex = tuple(
                [ np.s_[ ncv[i]: ncv[i]+Js.size[i]-1 ] 
                for i in range(ndim-1,-1,-1)
                ] )
            J[ croptuple ] += Js[rangedex]
            
            if(jump_detect):
                wgt += jf[rangedex]
            
        if(jump_detect):
            J[ croptuple ] /= wgt.clip(1)
        else:
            J[ croptuple ] /= 2**ndim
            
        # Sweep out to the boundaries.
        for axis in range(ndim):
            J[ tuple(     list(repeat(np.s_[:], axis)) + [0] ) ] = (
                J[ tuple( list(repeat(np.s_[:], axis)) + [1] ) ]
                )
            J[ tuple(     list(repeat(np.s_[:], axis)) + [J.size[axis]-1] ) ] = (
                J[ tuple( list(repeat(np.s_[:], axis)) + [J.size[axis]-2] ) ]
                )
                

        

def interpND_grid(source, /, 
             index=None, 
             method='l', 
             bound='t', 
             antialias=True, aa=True,
             fillvalue=0, 
             oblur=1.0,
             iblur=1.0,
             pad_pow=8,
             sv_limit=0.25,
             jump_detect=4.0,
             strict=False):
    '''
    interpND_grid - a better N-D interpolator for grids (with anti-aliasing)
    
    You supply source data in reversed dimension order (...,Y,X) and index
    data as a collection of vectors pointing into the source.  The index is 
    collapsed one dimension by interpolation into source: the contents along
    the final dimension axis are a vector that points (as (X,Y,...)) to a 
    location in the source; other prior axes in the index are broadcast. 
    The index vectors are taken to represent a regular grid (implicit in their
    broadcast axis coordinates).  This means that the index data themselves
    can be used to infer how best to sample the source data at each point.
    
    interpND_grid differs from interpND in that interpND does not use the 
    relationship between sample points, while interpND_grid can.  In other
    respects, interpND_grid is identical to interpND.  In fact, if the 
    "antialias" parameter or its, er, alias "aa" is set to False (default is 
    True), then interpND is used to generate the output.
    
    If "antialias" and "aa" are True, then interpND_grid linearizes the 
    implicit coordinate transformation at each gridpoint, using the discrete
    Jacobian derivative of the grid vector values to calculate a tailored
    resampling function at each sample point.  This eliminates aliasing, at 
    the cost of band-limiting the input data.
    
    Local filter calculation uses a variant of the singular-value padding 
    described by DeForest (2004, Sol. Phys. 219, 3):  singular values of the 
    Jacobian matrix are used to find the "footprint" of the grid in the 
    input plane.  These singular values are padded to unity (using either 
    the min function or quadrature addition), and used to modify the filter
    function used for weighted interpolation of nearby points.                                                        
    
    Parameters
    ----------
    
    source : numpy.ndarray
        The source data from which to interpolate.
        
    index : numpy.ndarray
        The index data to collapse by interpolation from source.  The final
        axis runs across dimensionality of the index vector into source.  If
        the 'strict' flag is set (the default case) then the size of the 
        final axis must match the number of dimensions in source.  If index 
        has additional dimensions they are broadcast.
        
    bound : string or list or tuple (default 'truncate')
        This is either a string describing the boundary conditions to apply to
        every axis in the input data, or a list of strings containing the 
        boundary conditions on each axis.  The boundary conditions are listed
        in vector order, not index order -- so an image array containing
        values in (Y,X) order should be indexed in (X,Y) order in the index 
        array, and boundary conditions are [boundX, boundY].  Only the first
        character of each string is checked.   The strings must be one of:
            
            'f' - forbid boundary violations
            
            't' - truncate the array at the boundary.  Values outside are 
                replaced with the fillvalue.
                
            'e' - extend the array at the boundary.  Values outside are 
                replaced with the nearest value in the source array.
                
            'p' - periodic boundary conditions apply.  Indices are modded with
                the size of the corresponding axis in the source array.
                
            'm' - mirror boundary conditions apply.  Indices reflect off each
                boundary of the data, counting backward to the opposite 
                boundary (where they reflect again).
                
    fillvalue : float (default 0)
        This is the value used to fill elements that are outside the bounds of
        the source array, if the 'truncate' boundary condition is selected.
        
    method : string (default 'nearest')
        This controls the interpolation method.  The value is a string.  Each 
        method has a single-letter abbreviation.  If the aa or antialias flag
        is set to False, then the interpND methods are accepted. If it is set
        to True (the default) then only the following methods are accepted.  
        
            'linear'/'l' - use linear (tent function) interpolation; this is 
                only true linear interpolation for grid offset operations, as
                the tent filter includes more than just two points along each
                axis in the general case.
        
            'lanczos'/'z' - use Lanczos-function weighting in the input plane; 
                this is a modified sinc that rolls off smoothly over three 
                samples and approximates a trapezoid in frequency space.  In 
                the context of anti-aliasing, the Lanczos filter loses the 
                property of reproducing the input value exactly at pixel centers.
        
            'hann'/'h' - use Hann window (overlapping sin^2) interpolation.  
                The Hann window smoothly transitions between pixels in the 
                up-sampling case and preserves correct weighting in the 
                down-sampling case with affine (uniform-Jacobian) resampling.
            
            'tukey'/'t' - The Tukey window rolls off with a cosine taper like
                the Hann window, but over only half the width of each pixel. 
                This imposes slightly broader sidelobes on the data but also
                (gently) renders pixel boundaries for upsampled data.      
    
            'gaussian'/'g' - use Gaussian weighted smoothing.  The Gaussian 
                has a full-width of 1 pixel, and is enumerated for a radius
                of three half-widths in all directions.  Gaussian filtering 
                is an analytic balance between frequency response and 
                spatial footprint. 
        
        antialias : Boolean (default True)
            Setting this to False falls through to InterpND, providing a 
            uniform API for the two methods.
            
        aa : Boolean (default True)
            Abbreviation for "antialias".
            
        oblur: Float (default 1.0)
            This is a coefficient applied to the width parameter of the filter
            function, in the output space after padding.  1.0 uses the nominal 
            width (typically 1 adjusted pixel); larger numbers cause uniform
            blur in the output space.  Numbers smaller than 1.0 are allowed
            but not recommended.
        
        iblur: Float (default 1.0)
            This is the value to which the singular values of the Jacobian are
            padded, in the input plane.  The default ensures sampling of a 
            one-pixel neighborhood even for upsampling.  Larger values cause
            some blurring on upsample/enlargement, but do not affect resampling 
            to decimate/reduce data.
        
        pad_pow: Float (default 8)
            This controls how the singular values of the Jacobian are padded to
            ensure that a reasonable neighborhood in the input plane is sampled.
            Singular values padded using the formula
                (S = (1 + s^pad_pow)^(1/pad_pow)).  
            High values of pad_pow approximate direct padding (as in DeForest 
            2004).  The default value of 8 has the effect of broadening the 
            filter function by 9% if the input and output grids have exactly
            the same spacing.
            
        sv_limit: Float (default 0.25)
            This controls what fraction of the input array may be averaged over
            by a single output pixel, to prevent runaway in case of singular or
            pathological sampling.  The default value allows up to a quarter of
            each dimension to be averaged over in a single output pixel.
            
        jump_detect: Float or Boolean (default 4.0)
            This flag indicates whether the linearizing logic should detect 
            jumps in the Jacobian values from pixel to pixel, and guess at a 
            reasonable smooth Jacobian value at those jumps.  Some transforms 
            (for example radial coordinate transforms) include jumps between
            branches of the analytic solution; and others involve wrapping
            around a source array with periodic boundary conditions.  Either 
            condition causes local jumps in the value of the Jacobian, from 
            reasonable values to very large values. Jump detection works by 
            finding jumps in the Jacobian values. Spikes that are more than 
            <jump_detect> times the value of the running local median for that 
            value cause that particular value to be replaced with the median.  
            Setting jump_detect to False or 0 turns off the feature.
    
        strict : Boolean (default True)
        The 'strict' parameter forces strict matching of index vector dimension
        to the number of axes in the source array.  If it is False, then the
        indexing vectors may be smaller than the number of indices in Source, 
        in which case the interpolation is broadcast over the additional axes.
        (Strict must be True at present)
            
    Returns
    -------
    The indexed data resampled from the source array, as a numpy ND array
    '''
    if not(antialias and aa):
        return interpND(source,
                        index=index,
                        method=method,
                        bound=bound,
                        fillvalue=fillvalue,
                        strict=strict)
    if not isinstance(index, np.ndarray):
        index = np.array(index)
    
    if not isinstance(source, np.ndarray):
        index = np.array(index)
        
    if not strict:
        raise ValueError("interpND_grid: non-strict interpolation is not implemented")
        
    if strict:
        if index.shape[-1] != len(source.shape):
            raise ValueError("interpND_grid: source shape must match index vector length")
        if len(source.shape) != len(index.shape)-1:
            raise ValueError("interpND_grid: index broadcast dims must match source dims")

    # Apply boundary conditions *before* calculating the Jacobian, to catch jumps 
    # from either the intrinsic transformation or the applied boundaries.
    bindex = apply_boundary(index, source.shape, bound=bound, rint=False)
    dims = len(index.shape[-1])
    
    J = jacobian(bindex, jump_detect=jump_detect)


    
    
            
        


        
    
    
    
    
            
            
        
            

            
        
            
            
    
                
    
    
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transform subclasses for basic coordinate transforms
"""
import numpy as np
from .core import Transform

class Linear(Transform):
    '''
    Linear - linear transforms
    
    Linear tranforms consist of an offset and a matrix operation.
    For convenience, two offsets are allowed: a pre-offset and a post-offset.
    the pre-offset, post-offset, and matrix, if present, must all be 
    ndarrays.  They must agree dimensionally: the pre must be a vector with 
    the same size as the matrix's 0 dim, and the post must be a vector with
    the same size as the matrix's 1 dim.
    
    Parameters
    ----------
    pre : numpy.ndarray
        Optional; if present must be a vector.  Offset vector to be added
        to the input data before hitting with the matrix
        
    post : numpy.ndarray
        Optional; if present must be a vector.  Offset vector to be added
        to the return data after hitting with the matrix
        
    matrix : numpy.ndarray
        Optional; if present must be a 2-D array. 

    '''
    
    sub __init__(self,
                 pre    = None, post   = None,\
                 matrix = None,\
                 iunit  = None, ounit  = None,\
                 itype  = None, otype  = None\
                ):
        
        idim = None
        odim = None
        
        ### Check if we got a matrix.  if we did, make sure it's 2-D and use
        ### it to set the idim and odim.
        if( matrix != None ):
            if( isinstance( matrix, np.ndarray ) ):
                if( len( matrix.shape ) == 2 ):
                    idim,odim = matrix.shape
                else:
                    raise ValueError("Linear: matrix must be 2D")
            else:
                raise ValueError("Linear: matrix must be a 2D numpy.ndarray or None")
        
        ### Now validate the pre and post, if present
        if( pre != None ):
            if( isinstance( pre, np.ndarray ) ):
                if( len( pre.shape ) == 1 ):
                    if( idim != None ):
                        if( idim != pre.shape[0] ):
                            raise ValueError("Linear: pre-offset and matrix have different sizes")
                    else:
                        idim = pre.shape[0]
                else:
                    raise ValueError("Linear: pre-offset must be a 1-D vector")
            else:
                raise ValueError("Linear: pre-offset must be a 1-D numpy.ndarray or None")
        
        if( post != None ):
            if( isinstance( post, np.ndarray ) ):
                if( len( post.shape ) == 1 ) :
                    if( odim != None ):
                        if( odim != post.shape[0] ) :
                            raise ValueError("Linear: post-offset and matrix have different sizes")
                    else:
                        odim = post.shape[0]
                else:
                    raise ValueError("Linear: post-offset must be a 1-D vector")
            else:
                raise ValueError("Linear: post-offset must be a 1-D numpy.ndarray or None")
                
        if( matrix==None ):
            if( odim == None ):
                odim = idim
            elsif( idim == None ):
                idim = odim
            elsif( idim != odim ):
                raise ValueError("Linear: idim and odim must match if no matrix is supplied")
      
        ### Finally - if no idim and odim, default to 2
        if( idim==None ):
            idim = 2
        if( odim==None ):
            odim = 2

   
        ### Now check for inverses.
        ### linalg.inv will throw an exception if the matrix isn't square or
        ### can't be inverted.
        matinv = None
        if( matrix != None ):
            try:
                matinv = np.linalg.inv(matrix)
            except:
                pass
            
        self.idim       = idim             
        self.odim       = odim
        self.no_forward = False
        self.no_reverse = (matinv == None  and  matrix != None)
        self.iunit      = iunit
        self.ounit      = ounit
        self.itype      = itype
        self.otype      = otype
        self.params = {\
            'pre'    :  pre,\
            'post'   :  post,\
            'matrix' :  matrix,\
            'matinv' :  matinv,\
            }
        
        def _forward(self,data):
            pass
        
        def _reverse(self,data):
            pass
        
        def __str__(self):
            pass
            
            
    
    
    
    


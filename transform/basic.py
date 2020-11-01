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
    
    def __init__(self,
                 pre    = None, post   = None,\
                 matrix = None,\
                 iunit  = None, ounit  = None,\
                 itype  = None, otype  = None\
                ):
        
        idim = None
        odim = None
        
        ### Parse arguments.  Pre, Matrix, Post.
        ### This takes a fair few lines but is straightforward:
        ### if there is a matrix, check that it's a 2-D array and 
        ### set the idim/odim values from it -- remembering that 
        ### Python lists matrix dims in mathematical (row,column) order.
        ### Then deal with the offsets and either set or check the idim/odim
        ### parameters for consistency with the matrix.
        
        ### Check if we got a matrix.  if we did, make sure it's 2-D and use
        ### it to set the idim and odim.
        if( np.all( matrix != None ) ):
            if( isinstance( matrix, np.ndarray ) ):
                if( len( matrix.shape ) == 2 ):
                    odim,idim = matrix.shape
                else:
                    raise ValueError("Linear: matrix must be 2D")
            else:
                raise ValueError("Linear: matrix must be a 2D numpy.ndarray or None")
        
        ### Now validate the pre and post, if present
        if( np.all( pre != None ) ):
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
        
        if( np.all( post != None ) ):
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
                
        if( np.all( matrix==None ) ):
            if( odim == None ):
                odim = idim
            elif( idim == None ):
                idim = odim
            elif( idim != odim ):
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
        self.params = {\
            'pre'    :  pre,\
            'post'   :  post,\
            'matrix' :  matrix,\
            'matinv' :  matinv,\
            }
            
    def __str__(self):
        self._strtmp = 'Linear'
        return (super().__str__())
    
    def _forward( self, data: np.ndarray ):
        if( data.shape[0] < self.idim ):
            raise ValueError('This Linear needs {self.idim}-vecs; source has {data.shape[0]}-vecs')
        
        data0 = data[0:self.idim]
        
        if( isinstance(self.params['pre'], np.ndarray)):
            data0 = data0 + self.params['pre']
        
        m = self.params['matrix']
        if( isinstance(m, np.ndarray) ):
            data0 = np.expand_dims(data0,1)  # convert vectors to Mx1
            data0 = np.matmul( m, data0 ) 
            data0 = data0[:,0]               # convert back to vectors
        
        if( isinstance( self.params['post'], np.ndarray ) ):
            data0 = data0 + self.params['post']
        
        if( data.shape[0] > self.idim ):
            return( np.append( data0, data[self.idim:], axis=0 ) )
        else:
            return( data0 )

    def _reverse( self, data: np.ndarray ):
        
        if( data.shape[0] < self.odim ):
            raise ValueError('This reverse-Linear needs {self.odim}-vecs; source has {data.shape[0]}-vecs')
        
        data0 = data[0:self.odim]
        
        if( isinstance( self.params['post'], np.ndarray ) ):
            data0 = data0 - self.params['post']
        
        m = self.params['matinv']
        if( isinstance( m, np.ndarray ) ):
            data0 = np_expand_dims(data0,1)
            data0 = np.matmul( m, data0 )
            data0 = data0[:,0]
            
        if( isinstance( self.params['pre'], np.ndarray ) ):
            data0 = data0 - self.params['pre']
        
        if( data.shape[0] > self.odim ):
            return( np.append( data0, data[self.odim:], axis=0 ) )
        else:
            return( data0 )
        
        
        
    
       
            
    
    
    
    


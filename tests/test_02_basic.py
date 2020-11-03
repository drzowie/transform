#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 22:48:57 2020

@author: zowie
"""

import transform as t
import numpy as np

def test_001_Linear_pre_post():
    ### Test basic pre-offset handling: add [1,2] offset to data: [0,0]
    a = t.Linear(pre=np.array([1,2]))
    assert isinstance(a,t.Linear), 'Constructor made a pre-offset Linear' 
    d1 = np.array([0,0])
    d2 = a.apply( d1 )
    assert isinstance(d2,np.ndarray), "apply returned an NDarray" 
    assert np.all(d2.shape == np.array([2])), "return value was correct size" 
    assert np.all( d2 == np.array([1,2]) ), "return value was correct value "
    d3 = a.invert( d2 )
    assert np.all( d1 == d3 ),  "reverse transform worked" 
    b = a.inverse()
    d4 = b.apply( d2 )
    assert np.all( d1==d4 ), "inverse transform worked" 
    
    ### Test pre-offset 1-D broadcast
    d1 = np.array([[0,1],[1,2],[3,4],[5,6]])
    d2 = a.apply( d1 )
    assert np.all( d1.shape == d2.shape ), "1-D broadcast: shapes work"
    assert np.all( d2 == d1 + np.array([1,2])), "1-D broadcast: value is right"
    
    ### Test pre-offset multi-D broadcast
    d1 = np.array([[[0,1],[2,3],[4,5]],[[6,7],[8,9],[10,11]]])
    d2 = a.apply( d1 )
    assert np.all( d1.shape == d2.shape ), "2-D broadcast: shapes work"
    assert np.all( d2 == d1 + np.array([1,2])), "2-D broadcast: value is right"       
    
    ### Test post-offset multi-D 
    a = t.Linear(post=np.array([1,2]))
    d1 = np.array([[0,0],[10,20],[30,40]])
    d2 = a.apply( d1 )
    assert isinstance(d2, np.ndarray) 
    assert np.all(d2.shape == d1.shape)
    print(f"d2 is {d2}")
    print(f"d1 is {d1}")
    print(f"d1 + [1,2] is {d1+np.array([1,2])}.")
    assert np.all( d2 == (d1 + np.array([1,2]))) 
    d3 = a.inverse().apply( d2 )
    assert np.all( d3==d1 )
    
    ### Test post-offset truncation (2-D transform; 3-D data)
    ### Should ignore last dim
    a = t.Linear( post=np.array([1,2]) )
    d1 = np.array( [[1,2,3],[4,5,6],[7,8,9]] )  # 3-vectors
    d2 = a.apply( d1 )
    assert isinstance( d2, np.ndarray )
    assert np.all( d2.shape == d1.shape )
    assert np.all( d2 == d1 + np.array([1,2,0]) )
    
    # Test apply with a list
    d2 = a.apply( [0,0] ) 
    d2 = a.invert( [0,0] )
    
def test_001_Linear_matrix():
    ### Test identity matrix - basic functionality
    a = t.Linear( matrix = np.array( [[1,0],[0,1]] ) )
    d1 = np.array([[1,2],[3,4],[5,6]])
    d2 = a.apply(d1)
    assert np.all( d1.shape == d2.shape )
    assert np.all( d1 == d2 )
    
    ### Test sense of matrix - (row,column) indexing, so rows are innermost
    a = t.Linear( matrix = np.array( [[1,1],[2,2]] ) )
    d1 = np.array( [[1,2],[3,4],[5,6]] )
    d2 = a.apply(d1)
    assert np.all( d1.shape == d2.shape )
    assert np.all( d2 == np.array( [[3,6],[7,14],[11,22]] ) )
    
    ### Test inversion of matrix 
    a = t.Linear( matrix = np.array( [[1,1],[1,2]] ) )
    d1 = np.array( [[1,2],[3,4],[5,6]] )
    d2 = a.apply(d1)
    d3 = a.inverse().apply(d2)
    assert np.all( d1 == d3 )
    
    ### Test failure to invert a non-invertible matrix
    a = t.Linear( matrix = np.array( [[1,1],[2,2]] ) ) # zero determinant
    b = a.inverse()
    d1 = np.array( [[1,2],[3,4],[5,6]] )
    try:
        d2 = b.apply(d1)
    except:
        pass
    else:
        assert False,"Should have chucked an invalid-direction exception"
    d2 = a.apply(d1)
    d3 = b.invert(d1)
    assert np.all(d2==d3)
    
def test_002_Scale():
    a = t.Scale( np.array([1,2]) )
    assert a.idim == 2  and a.odim == 2 
    d1 = a.apply( np.array([10,20]) )
    assert all( d1 == np.array([10,40]) )
    
    a = t.Scale( 3 )
    assert a.idim == 2 and a.odim == 2 
    d1 = a.apply( np.array( [5,50] ) )
    assert all( d1 == np.array([15,150]) )
    
    a = t.Scale( 3, dim=3 )
    assert a.idim==3 and a.odim==3
    assert np.all(np.isclose(a.params['matrix'],
                             np.array([[3,0,0],[0,3,0],[0,0,3]]),
                             atol=1e-15
                             )
                  )
                                      

def test_003_Rotation():
    # Test implicit 2D rotation
    a = t.Rotation( 0 )
    assert a.idim == 2 and a.odim == 2
    assert np.all(a.params['matrix']== np.array([[1,0],[0,1]]))
    
    # Test direction of implicit 2D rotation: matrix
    a = t.Rotation( 90, u='deg' )
    assert np.all( np.isclose( a.params['matrix'],          
                               np.array( [[0,-1], [1,0]] ), 
                               atol=1e-15,
                               ) )
    # Test direction of implicit 2D rotation: actual vector application
    d0 = np.array( [[1,0],[0,1]] )
    d1 = a.apply( d0 )
    assert np.all( np.isclose( d1,                          
                               np.array( [[0,1],[-1,0]] ),  
                               atol=1e-15,
                               ) )
    # Test inverse
    d2 = a.invert( d1 )
    assert np.all( np.isclose( d2, d0, atol=1e-15 ) )
    
                
    # Make sure default rotation is radians not degrees
    a = t.Rotation( 90 )
    assert not np.any( np.isclose( a.params['matrix'],           
                                   np.array( [[ 0,-1], [1,0]] ), 
                                   atol=1e-15
                                  ) )
    
    # Test rotation *from* Y *to* X (reverse sense from standard)  
    a = t.Rotation([1,0,90],u='deg')
    assert np.all( np.isclose( a.params['matrix'],           
                               np.array( [[0,1],[-1,0]] ),   
                               atol=1e-15,
                                   ) )
    
    # Test rotation order
    a = t.Rotation([[1,2,90],[0,1,90]],u='deg')
    assert a.idim==3
    d0 = np.array( [[1,0,0],[0,1,0],[0,0,1]] )
    d1 = a.apply(d0)
    d1a = np.array( [[0,0,1],[-1,0,0],[0,-1,0]] )
    assert np.all( np.isclose( d1, d1a, atol=1e-15 ) )
    
    # Test Euler angles
    a = t.Rotation([[0,1,90],[1,2,90]],u='deg')
    b = t.Rotation(euler= np.array( [90,0,90] ), u='deg')
    d1a = a.apply(d0)
    d1b = b.apply(d0)
    d1c = np.array([[0,1,0],[0,0,1],[1,0,0]])
    assert np.all( np.isclose( d1a, d1b, atol=1e-15) )
    assert np.all( np.isclose( d1a, d1c, atol=1e-15) )
    
    assert( f"{a}" == "Transform( Linear/Rotation )")
    
    
def test_004_Offset():
    a = t.Offset([1,2])
    assert( f"{a}" == "Transform( Linear/Offset )")
    data = [0,0]
    b = a.apply(data)
    
    
    
    
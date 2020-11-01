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
        assert false,"Should have chucked an invalid-direction exception"
    d2 = a.apply(d1)
    d3 = b.invert(d1)
    assert np.all(d2==d3)


    
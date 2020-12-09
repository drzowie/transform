#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 22:48:57 2020

Tests for the basic round of transforms in basic.py

These are intended for use with pytest.

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
           
    # Check that lists of scales also work for a diagonal matrix                           
    a = t.Scale([3,2])
    assert a.idim==2 and a.odim==2
    assert np.all(np.isclose(a.params['matrix'],
                             np.array([[3,0],[0,2]]),
                             atol=1e-15
                             )
                  )

def test_003_Rotation():
    # Test implicit 2D rotation
    a = t.Rotation( 0 )
    assert a.idim == 2 and a.odim == 2
    assert np.all(a.params['matrix']== np.array([[1,0],[0,1]]))
    
    # Test direction of implicit 2D rotation: matrix
    a = t.Rotation( 90, unit='deg' )
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
    a = t.Rotation([1,0,90],unit='deg')
    assert np.all( np.isclose( a.params['matrix'],           
                               np.array( [[0,1],[-1,0]] ),   
                               atol=1e-15,
                                   ) )
    
    # Test rotation order
    a = t.Rotation([[1,2,90],[0,1,90]],unit='deg')
    assert a.idim==3
    d0 = np.array( [[1,0,0],[0,1,0],[0,0,1]] )
    d1 = a.apply(d0)
    d1a = np.array( [[0,0,1],[-1,0,0],[0,-1,0]] )
    assert np.all( np.isclose( d1, d1a, atol=1e-15 ) )
    
    # Test Euler angles
    a = t.Rotation([[0,1,90],[1,2,90]],unit='deg')
    b = t.Rotation(euler= np.array( [90,0,90] ), unit='deg')
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
    
def test_005_TestForNdims():
    a = t.Scale( np.array([1,2,3]) )
    assert a.idim == 3 and a.odim == 3 
    d1 = a.apply( np.array([10,20,30]) )
    assert all( d1 == np.array([10, 40, 90]) )

def test_007_Radial():
    
    # Start with a transform with origin at (130,130), and
    # test inversion.  First vector is (0,1):  radius 1, angle 0.
    # It should direct to (131,130).  Second vector is (2,3): 
    # radius 3, angle 2 radians (i.e. 114.6 degrees).  
    # Given that Radial works in the *clockwise* direction by default
    # (opposite of mathematical standard) this should result in
    # an offset left and down.
    x = np.arange(4.).reshape(2,2)
    b = t.Radial(origin=[130,130])
    d1 = np.array([[131.0 , 130.0], [ 128.75,  127.27]])
    d2 = b.invert(x)  
    assert np.all( np.isclose(d1 , d2, atol=1e-2))
    d3 = np.array([ [2.3600555, 183.14202], [2.360116, 180.31362]])
    d4 = b.apply(x) 
    assert np.all( np.isclose(d3 , d4, atol=1e-2))
    cd = t.Radial(origin=[130,130], r0=5)
    bd1 = np.array([[2.3600555, 3.600824], [2.360116, 3.5852597]])
    bd2 = cd.apply(x)
    bd3 = np.array([ [143.59141, 130], [88.207337, 38.681365]])
    bd4 = cd.invert(x)
    assert(  np.allclose(bd1,bd2, atol=1e-2))
    assert(  np.allclose(bd3,bd4, atol=1e-2))
    
    # Test handling of multi-dim broadcasting
    # make a 7x7 grid, 0 centered, and test the corners (good enough for
    # basics and polarities)
    # For the angular test of the corners, remember that array indices
    # are in Python (Y,X) order -- so b[0,-1] corresponds to the 
    # lower, rightmost corner of the square -- which has coordinates (-3,3)
    # and should be PI/4 radians around the (clockwise) circle.
    a = np.mgrid[0:7,0:7].T - 3
    trans = t.Radial()
    b = trans.apply(a)
    assert( all([np.isclose(b[0,0,1],  3*np.sqrt(2), atol=1e-5),
                 np.isclose(b[-1,-1,1],3*np.sqrt(2), atol=1e-5),
                 np.isclose(b[0,-1,1], 3*np.sqrt(2), atol=1e-5),
                 np.isclose(b[-1,0,1], 3*np.sqrt(2), atol=1e-5)
                 ]
                )
           )
    assert( all([np.isclose(b[0,-1,0], (0.25)*np.pi, atol=1e-5),
                 np.isclose(b[0,0,0],  (0.75)*np.pi, atol=1e-5),
                 np.isclose(b[-1,0,0], (1.25)*np.pi, atol=1e-5),
                 np.isclose(b[-1,-1,0],(1.75)*np.pi, atol=1e-5)
                 ]
                )
           )
    assert(b.shape==a.shape)
    c = trans.invert(b)
    assert(np.all(np.isclose(a,c,atol=1e-10)))
    
    # Test CCW (conventional) transform with the multi-dim case
    # Uses the same a as above.
    trans2 = t.Radial(ccw=True)
    bb = trans2.apply(a)
    # all radii should be the same
    assert( np.all( np.isclose(bb[...,1], b[...,1],atol=1e-5)))
    # angles should be reversed

    assert( np.all( np.isclose( (5*np.pi + bb[...,0]) % (2*np.pi), (5*np.pi + b[range(6,-1,-1), ...,0]) % (2*np.pi) )))
    cc = trans2.invert(bb)
    assert(np.all(np.isclose(a,cc,atol=1e-10)))
    
    # Test allowing negative-going angles
    trans = t.Radial(pos_only=False)
    b = trans.apply(a)
    assert( all([np.isclose(b[0,0,1],  3*np.sqrt(2), atol=1e-5),
                 np.isclose(b[-1,-1,1],3*np.sqrt(2), atol=1e-5),
                 np.isclose(b[0,-1,1], 3*np.sqrt(2), atol=1e-5),
                 np.isclose(b[-1,0,1], 3*np.sqrt(2), atol=1e-5)
                 ]
                )
           )
    assert( all([np.isclose(b[0,-1,0], (  0.25)*np.pi, atol=1e-5),
                 np.isclose(b[0,0,0],  (  0.75)*np.pi, atol=1e-5),
                 np.isclose(b[-1,0,0], ( -0.75)*np.pi, atol=1e-5),
                 np.isclose(b[-1,-1,0],( -0.25)*np.pi, atol=1e-5)
                 ]
                )
           )
    c = trans.invert(b)
    assert(np.all(np.isclose(a,c,atol=1e-10)))
   
    
    
    
    

def test_008_Spherical():
    x = np.arange(27.).reshape(3,3,3)
    a = t.Spherical(origin=[130,130,130])
    d1 = a.apply(x)  
    d2 = np.array( [[[ -2.36005547,  -0.60999482, 223.43902972],
                     [ -2.36014704,  -0.60986414, 218.24298385],
                     [ -2.36024305,  -0.60972709, 213.04694318]],
                    [[ -2.36034384,  -0.60958318, 207.8509081 ],
                     [ -2.36044978,  -0.60943189, 202.65487904],
                     [ -2.36056127,  -0.60927262, 197.45885647]],
                    [[ -2.36067877,  -0.60910475, 192.26284092],
                     [ -2.36080275,  -0.60892754, 187.06683298],
                     [ -2.36093379,  -0.60874019, 181.87083329]]] )
    assert(  np.allclose(d1,d2, atol=1e-2))
    d3 = a.invert(d1) 
    d4 = x
    assert(  np.allclose(d3,d4, atol=1e-2))
    
def test_009_Quadratic():
    x = np.arange(9.).reshape(3,3)
    a = t.Quadratic()
    d1 = a.apply(x)
    d2 = np.array( [[         0,          1,  2.1818182],
                    [ 3.5454545,  5.0909091,  6.8181818],
                    [ 8.7272727,  10.818182,  13.090909]])
    assert(  np.allclose(d1,d2, atol=1e-2))
    d3 = a.invert(d1) 
    d4 = x
    assert(  np.allclose(d3,d4, atol=1e-2))
    a = t.Quadratic(length=5.0)
    d5 = a.apply(x)
    d6 = np.array( [[         0, 0.92727273,  1.8909091],
                    [ 2.8909091,  3.9272727,          5],
                    [ 6.1090909,  7.2545455,  8.4363636]])
    assert(  np.allclose(d5,d6, atol=1e-2))
    d7 = a.invert(d6) 
    assert(  np.allclose(d7,d4, atol=1e-2))
    a = t.Quadratic(origin=5.0)
    d8 = a.apply(x)
    d9 = np.array( [[  -1.8181818, -0.090909091,    1.4545455],
                    [   2.8181818,            4,            5],
                    [           6,    7.1818182,    8.5454545]])
    assert(  np.allclose(d8,d9, atol=1e-2))
    d10 = a.invert(d8) 
    assert(  np.allclose(d10,d4, atol=1e-2))

    a = t.Quadratic(strength=6.0)
    d11 = a.apply(x)
    d12 = np.array( [[         0,          1,  3.7142857],
                    [ 8.1428571,  14.285714,  22.142857],
                    [ 31.714286,         43,         56]])
    assert(  np.allclose(d11,d12, atol=1e-2))
    d13 = a.invert(d12) 
    assert(  np.allclose(d13,d4, atol=1e-2))

    a = t.Quadratic(idim=2)
    d14 = a.apply(x)
    d15 = np.array( [[         0,          1,          2],
                     [ 3.5454545,  5.0909091,          5],
                     [ 8.7272727,  10.818182,          8]])
    assert(  np.allclose(d14,d15, atol=1e-2))
    d16 = a.invert(d14)
    assert(  np.allclose(d16,d4, atol=1e-2))
    
    
def test_010_Cubic():
    x = np.arange(9.).reshape(3,3)
    a = t.Cubic(strength=2)
    d1 = a.apply(x)
    d2 = np.array( [[    0,   0.6,   3.6],
                    [ 11.4,  26.4,    51],
                    [ 87.6, 138.6, 206.4]])
    assert(  np.allclose(d1,d2, atol=1e-2))
    d3 = a.invert(d1) 
    d4 = np.array( [[         0, 0.79501675,  1.6595098],
                    [ 2.5116355,  3.3596201,  4.2058325],
                    [ 5.0511344,  5.8959087,  6.7403507]])
    assert(  np.allclose(d3,d4, atol=1e-2))
    a = t.Cubic(length=2.0)
    d5 = a.apply(x)
    d6 = x
    assert(  np.allclose(d5,d6, atol=1e-2))
    d7 = a.invert(d5)
    d8 = np.empty((3,3))
    d8[:] = np.NaN
    assert(  np.allclose(d7,d8, atol=1e-2, equal_nan=True))

    a = t.Cubic(idim=2)
    d9 = a.apply(x)
    d10 = x
    assert(  np.allclose(d9,d10, atol=1e-2))
    d11 = a.invert(d9)
    d12 = x
    d12[:,:2] = np.NaN
    
    assert(  np.allclose(d11,d12, atol=1e-2, equal_nan=True))


def test_011_Quartic():
    x = np.arange(9.).reshape(3,3)
    a = t.Quartic()
    d1 = a.apply(x)
    d2 = np.array( [[         0,          1,  2.1818182],
                    [ 3.5454545,  5.0909091,  6.8181818],
                    [ 8.7272727,  10.818182,  13.090909]])
    assert(  np.allclose(d1,d2, atol=1e-2))
    d3 = a.invert(d1) 
    d4 = x
    assert(  np.allclose(d3,d4, atol=1e-2))
    a = t.Quartic(length=5.0)
    d5 = a.apply(x)
    d6 = np.array( [[         0, 0.92727273,  1.8909091],
                    [ 2.8909091,  3.9272727,          5],
                    [ 6.1090909,  7.2545455,  8.4363636]])
    assert(  np.allclose(d5,d6, atol=1e-2))
    d7 = a.invert(d6) 
    assert(  np.allclose(d7,d4, atol=1e-2))
    a = t.Quartic(origin=5.0)
    d8 = a.apply(x)
    d9 = np.array( [[  -1.8181818, -0.090909091,    1.4545455],
                    [   2.8181818,            4,            5],
                    [           6,    7.1818182,    8.5454545]])
    assert(  np.allclose(d8,d9, atol=1e-2))
    d10 = a.invert(d8) 
    assert(  np.allclose(d10,d4, atol=1e-2))

    a = t.Quartic(strength=6.0)
    d11 = a.apply(x)
    d12 = np.array( [[         0,          1,  3.7142857],
                    [ 8.1428571,  14.285714,  22.142857],
                    [ 31.714286,         43,         56]])
    assert(  np.allclose(d11,d12, atol=1e-2))
    d13 = a.invert(d12) 
    assert(  np.allclose(d13,d4, atol=1e-2))

    a = t.Quartic(idim=2)
    d14 = a.apply(x)
    d15 = np.array( [[         0,          1,          2],
                     [ 3.5454545,  5.0909091,          5],
                     [ 8.7272727,  10.818182,          8]])
    assert(  np.allclose(d14,d15, atol=1e-2))
    d16 = a.invert(d14)
    assert(  np.allclose(d16,d4, atol=1e-2))
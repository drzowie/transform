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
    
def test_005_TestForNdims():
    a = t.Scale( np.array([1,2,3]) )
    assert a.idim == 3 and a.odim == 3 
    d1 = a.apply( np.array([10,20,30]) )
    assert all( d1 == np.array([10, 40, 90]) )
    
def test_006_WCS():
    a = t.WCS('sample.fits')
    assert(a.idim == 2)
    assert(a.odim == 2)
    assert(len(a.itype)==2 and len(a.otype)==2 and len(a.iunit)==2 and len(a.ounit)==2)
    assert(a.itype == ['X','Y'])
    assert(a.iunit==['Pixels','Pixels'])
    assert(a.ounit==['arcsec','arcsec'])
    assert(a.otype==['Solar-X','Solar-Y'])
    assert( np.all( np.isclose ( a.apply([[0,0]],0), np.array([[-386.15825,  -676.1092]]), atol=1e-4 ) ) )


def test_007_Radial():
    x = np.arange(4.).reshape(2,2)
    b = t.Radial(origin=[130,130])
    d1 = np.array([[131.0 , 130.0], [ 128.75,  127.27]])
    d2 = b.invert(x)  
    assert np.all( np.isclose(d1 , d2, atol=1e-1))
    d3 = np.array([ [2.3600555, 183.14202], [2.360116, 180.31362]])
    d4 = b.apply(x) 
    assert np.all( np.isclose(d3 , d4, atol=1e-1))
    cd = t.Radial(origin=[130,130], r0=5)
    bd1 = np.array([[2.3600555, 3.600824], [2.360116, 3.5852597]])
    bd2 = cd.apply(x)
    bd3 = np.array([ [143.59141, 130], [88.207337, 38.681365]])
    bd4 = cd.invert(x)
    assert(  np.allclose(bd1,bd2, atol=1e-2))
    assert(  np.allclose(bd3,bd4, atol=1e-2))


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
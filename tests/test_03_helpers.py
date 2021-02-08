#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 22:58:09 2020

Tests for the interpolation/helper routines in helpers.py

(for use with pytest)

"""

import numpy as np

import transform as t
from transform.helpers import apply_boundary
from transform.helpers import sampleND
from transform.helpers import interpND
import copy

def test_001_apply_boundary():
    
    vec = np.array([[0,2],[2,2],[3,2],[4,2],[-1,2],[-2,2],[-0.49,2],[-0.51,2],[-0.1,2],[-0.9,2]])
    
    try:
        v1 = apply_boundary(vec,[3,3],bound='f')
        raise AssertionError("apply_boundary should have thrown an exception")
    except:
        pass
    
    # Check no exception is thrown by 'f' when the vector is legal -- whether
    # it be a single element or multi-element.
    v1 = apply_boundary(0*vec+1,[2,2],bound='f')
    assert np.all(v1==1)
    v1 = apply_boundary(np.array([2,2]),[3,3],bound='f')
    assert np.all(v1==2)
    
    
    
    v1 = apply_boundary(vec,[3,3],bound='t')
    
    assert all(v1[:,1]==2)
    assert all(v1[:,0]==[0,2,-1,-1,-1,-1,0,-1,0,-1])
    
    v1 = apply_boundary(vec,[3,3],bound='e')
    assert all(v1[:,1]==2)
    assert all(v1[:,0]==[0,2,2,2,0,0,0,0,0,0])
    
    v1 = apply_boundary(vec,[3,3],bound='p')
    assert all(v1[:,1]==2)
    assert all(v1[:,0]==[0,2,0,1,2,1,0,2,0,2])
    
    v1 = apply_boundary(vec,[3,3],bound='m')
    assert all(v1[:,1]==2)
    assert all(v1[:,0]==[0,2,2,1,0,1,0,0,0,0])
    
    v0 = copy.copy(vec)
    v1 = apply_boundary(vec,[3,3],bound='e',rint=False,pixels=False)
    # make sure non-rint,non-pixels version doesn't change original
    assert all(v0.flat==vec.flat)
    assert all(v1[:,1]==2)
    assert all(np.isclose(v1[:,0],[0,2,3,3,0,0,0,0,0,0],atol=1e-9))
    
    v0 = copy.copy(vec)
    v1 = apply_boundary(vec,[3,3],bound='e',rint=False,pixels=True)
    # make sure non-rint,non-pixels version doesn't change original
    assert all(v0.flat==vec.flat)
    assert all(v1[:,1]==2)
    assert all( np.isclose(v1[:,0], 
                           [0,2,2.5,2.5,-0.5,-0.5,-0.49,-0.5,-0.1,-0.5],
                           atol=1e-9
                           )
               )
                           
    
    v1 = apply_boundary(vec,[3,3],bound='p',rint=False,pixels=False)
    assert all(v0.flat==vec.flat)
    assert all(np.isclose(v1[:,0],[0,2,0,1,2,1,2.51,2.49,2.9,2.1],atol=1e-8))
    
    

def test_002_sampND():
    # Make 5x5 sequence "image": 10s digits gets Y, 1s digit gets X; image is
    # in Python-standard (Y,X) order.
    datasource = (np.mgrid[0:5,0:50:10].transpose()).sum(axis=-1)
    
    # Test basic sampling
    a = sampleND(datasource, index=[3,4])
    assert(a==43)
    
    # Test sampling of a list of points
    a = sampleND(datasource, index=[[3,4],[1,2],[0.4,1.2]])
    assert len(a.shape)==1
    assert a.shape[0]==3
    assert all(a==[43,21,10])
    
    # Test boundary conditions - list of 1 vec, periodic/extend 
    # (collapses to list of 1 element)
    a = sampleND(datasource, index=[[-1,-1]],bound=['p','e'])
    assert len(a.shape)==1
    assert a==4
    
    # Test full collapse - same as before, but just a 1-vec
    # (collapses to scalar)
    a = sampleND(datasource, index=[-1,-1],bound=['p','e'])
    assert len(a.shape)==0
    assert a==4
    
    ####
    # Test extraction of a 1x1 array from each place (just adds two axes
    # of length 1 onto the end)
    datasource = (np.mgrid[0:7,0:70:10].transpose()).sum(axis=-1)
    dex = [[3,4],[1,2],[0.4,1.2]]
    a = sampleND(datasource, index=dex, chunk=[1,1])
    assert len(a.shape)==3
    assert all(a.shape==np.array([3,1,1]))
    assert np.all(a == np.array([[[43]],[[21]],[[10]]]))
    
    ###
    # Test extraction of a 2x2 array from each place
    a = sampleND(datasource, index=dex, chunk=[2,2])
    assert len(a.shape)==3
    assert all(a.shape==np.array([3,2,2]))
    assert np.all( a == [[[43,44],[53,54]],[[21,22],[31,32]],[[10,11],[20,21]]] )
    
    ###
    # Chunk parameter of 0 should omit the dimension
    a = sampleND(datasource, index = dex, chunk=[0,2])
    assert len(a.shape)==2
    assert all(a.shape==np.array([3,2]))
    assert np.all( a == [[43,53],[21,31],[10,20]])
    
    a = sampleND(datasource, index = dex, chunk=[2,0])
    assert len(a.shape)==2
    assert all(a.shape==np.array([3,2]))
    assert np.all( a == [[43,44],[21,22],[10,11]])
    
    ###
    # Check  boundary conditions in extended chunk
    datasource = (np.mgrid[0:2,0:20:10].transpose()).sum(axis=-1)
    try:
        a = sampleND(datasource, index = [-1,-1], chunk=[4,4])
        assert False
    except:
        pass
    
    a = sampleND(datasource, index = [-1,-1], chunk=[4,4], bound='p') 
    assert len(a.shape)==2
    assert all(a.shape == np.array([4,4]))
    assert np.all( a== np.array( [[11,10,11,10],[1,0,1,0],[11,10,11,10],[1,0,1,0]]) )
    
    a = sampleND(datasource, index=[-1,-1], chunk=[4,4], bound='t')
    assert len(a.shape)==2
    assert all(a.shape == np.array([4,4]))
    assert np.all( a== np.array( [[0,0,0,0],[0,0,1,0],[0,10,11,0],[0,0,0,0]]))
    
    a = sampleND(datasource, index=[-1,-1],chunk=[4,4],bound='m')
    assert len(a.shape)==2
    assert all(a.shape==np.array([4,4]))
    assert np.all( a== np.array( [[0,0,1,1],[0,0,1,1],[10,10,11,11],[10,10,11,11]] ))
    
    ####
    # Test non-strict indexing
    datasource = np.mgrid[0:10,0:100:10].transpose().sum(axis=-1)
    try:
        a = sampleND(datasource,index=[3],strict=1)
        assert False
    except:
        pass
    a = sampleND(datasource,index=[3])
    assert len(a.shape)==1
    assert a.shape[0]==10
    assert np.all(a == [3,13,23,33,43,53,63,73,83,93])
    
    # Single dimension: treat as a single index
    a = sampleND(datasource,index=[3,4])
    assert len(a.shape)==0
    assert a==43
    
    # Two dimensions, 1x2: treat as a list of one 2D index
    a = sampleND(datasource, index=[[3,4]])
    assert len(a.shape)==1
    assert a.shape[0]==1
    assert a[0]==43
    
    # Two dimensions, 2x1:  only innermost is index; they get broadcast as two rows of 10
    a = sampleND(datasource, index = [[3],[4]])
    assert len(a.shape)==2
    assert np.all(a.shape==np.array([10,2]))
    assert np.all( a == [[3,4],[13,14],[23,24],[33,34],[43,44],[53,54],[63,64],[73,74],[83,84],[93,94]])
    
    #
    # Test chunk order
    # 
    
    # First - double-check that two elements resolve correctly in an mgrid
    datasource = np.mgrid[0:6,0:60:10].transpose().sum(axis=-1)
    dexes = np.array([[1,2],[3,4]])
    a = t.sampleND(datasource, dexes)
    assert a.shape[0]==2
    assert len(a.shape)==1
    assert all(a==np.array([21,43]))
    
    # Make sure chunk samples are in the correct direction -- (...,Y,X)
    a = t.sampleND(datasource, dexes, chunk=[2,2])
    assert len(a.shape)==3
    assert a.shape[0]==2 and a.shape[1]==2 and a.shape[2]==2
    assert np.all( a == np.array([[[21,22],[31,32]],[[43,44],[53,54]]]))
    
    # Make sure that chunk sizes are handled right
    a = t.sampleND(datasource, dexes, chunk=[3,2])
    assert len(a.shape)==3
    assert a.shape[0]==2 and a.shape[1]==2 and a.shape[2]==3
    assert np.all( a[0] == np.array([[21,22,23],[31,32,33]])) 
    assert np.all( a[1] == np.array([[43,44,45],[53,54,55]]))
    
    
    
    
    
    
def test_003_interpND_nearest():
    data = np.mgrid[0:5,0:50:10].transpose().sum(axis=-1)
    
    # Test sampling. 
    # It's just a pass-through to sampleND, so
    # need for extensive testing here.
    a = interpND(data, [1.2, 2.8], method='n')
    assert a==31
    
def test_004_interpND_linear():
    # Test linear interpolation
    data = np.mgrid[0:5,0:50:10].transpose().sum(axis=-1)
    
    a = interpND(data, [1,3], method='l')
    assert np.isclose(a, 31, atol=1e-5)
    
    a = interpND(data, [1.2,3], method='l')
    assert np.isclose(a, 31.2, atol=1e-5)
    
    a = interpND(data, [1,2.8], method='l')
    assert np.isclose(a, 29, atol=1e-5)
    
    a = interpND(data, [1.2,2.8], method='l')
    assert np.isclose(a, 29.2, atol=1e-5)
    
    data = np.mgrid[0:5,0:500:100].transpose().sum(axis=-1)
    dex = np.mgrid[ 1:2.1:0.2, 2:3.1:0.2 ].transpose()
    a = interpND(data, dex, method='l')
    assert all(a.shape==np.array([6,6]))
    assert np.all(
        np.isclose(a, np.array(
            [ [201, 201.2, 201.4, 201.6, 201.8, 202 ],
              [221, 221.2, 221.4, 221.6, 221.8, 222 ],
              [241, 241.2, 241.4, 241.6, 241.8, 242 ],
              [261, 261.2, 261.4, 261.6, 261.8, 262 ],
              [281, 281.2, 281.4, 281.6, 281.8, 282 ],
              [301, 301.2, 301.4, 301.6, 301.8, 302 ]
             ]
            ), atol=1e-4 )
        )
    
def test_005_interpND_cubic():
    ## Cubic interpolation reduces to linear if the data are linear
    data = np.mgrid[0:5,0:500:100].transpose().sum(axis=-1)
    dex = np.mgrid[ 1:2.1:0.2, 2:3:.2 ].transpose()
    a = interpND(data, dex, method='c')
    np.isclose(a, np.array(
            [ [201, 201.2, 201.4, 201.6, 201.8, 202 ],
              [221, 221.2, 221.4, 221.6, 221.8, 222 ],
              [241, 241.2, 241.4, 241.6, 241.8, 242 ],
              [261, 261.2, 261.4, 261.6, 261.8, 262 ],
              [281, 281.2, 281.4, 281.6, 281.8, 282 ]
             ]
            ), 
        atol=1e-4
        )
    data = np.array([0,0,0,1,0,0,0])
    
    ## Basic test of localization and negativity for impulse response
    data = np.array([0,0,1,0,0])
    x = np.arange(0,6.1,0.1)
    # expand_dims call is necessary so that the arange is interpreted
    # as a collection of 1-vectors, rather than as a single 61-vector
    y = interpND(data, np.expand_dims(x,axis=-1), method='c', bound='e')

    #  Check that the interpolated curve passes through the points in the data
    assert all(np.isclose(y[ [0,10,20,30,40] ],
                       [0,0,1,0,0],
                       atol=1e-5
                       )
               )
    
    # Check: slight ringing just before and just after the impulse; 
    # all positive during the impulse
    assert all(y[1:10]<0)
    assert all(y[11:30]>0)
    assert all(y[31:40]<0)
    
def test_006_interpND_fft():
    a = np.array([1,0,0,1,1,0,0,0])
    b = interpND(a, np.array([[0],[1],[2],[3],[4],[5],[6],[7]]), method='f')
    assert b.shape[0]==8
    assert b.dtype in (np.dtype('double'),np.dtype('float'))
    assert all( np.isclose( b, a, atol=1e-12) )
    
    aa = a + 0j
    b = interpND(aa, np.array([[0],[1],[2],[3],[4],[5],[6],[7]]), method='f')
    assert b.shape[0]==8
    assert b.dtype in(np.dtype('complex64'),np.dtype('complex128'))
    assert all( np.isclose( b, aa, atol=1e-12))
                                              
def test_007_interpND_filtermethods():
    
    a = np.array([1,0,0,1,1,0,0,0])
    dex = np.mgrid[0:7.1:0.1,].transpose().astype(float)-0.5
    
    # Verify that pixel values are reproduced -- all but 'g'
    for m in ('s','z','h'):
        b = interpND(a, dex, bound='t',method=m)
        assert( all( np.isclose( 
            b[np.array( [5,15,25,35,45,55])], 
            [1,0,0,1,1,0],
            atol=1e-10 
            ))
            )
        
    # Hand-check a couple of Gaussian values
    b = interpND(a, dex, bound='t', method='g')
    assert( np.isclose(b[5], 0.96466,atol=1e-5))
    assert( np.isclose(b[0], 0.5, atol=1e-3) )
    assert( np.isclose(b[10],0.5, atol=1e-3) )
    assert( np.isclose(b[20],0,   atol=1e-3) )

    # Verify that the filter functions can reproduce a 2D pattern
    a = np.array([[1,0,0,0],[0,1,0,0],[1,1,1,1],[0,0,0,1]])
    dex = np.mgrid[0:4,0:4].transpose()
    b = t.interpND(a,dex,method='h')
    assert np.all(np.isclose(a,b,atol=1e-9))
    
# oblur scales the filter function - blurring by 3 should expand the filter 
# function by 3.  test this with generalized linear interpolation (additional
# code path); the code is straightforward enough that the other filters are 
# good if this one is.
def test_008_interpND_with_oblur():
    a = np.zeros((7,7))
    a[3,3] = 1
    grid = np.mgrid[0:7,0:7].T
    b = t.interpND(a,grid,method='l',oblur=3)
    assert(np.all(np.isclose(b[3],np.array([0,0.037,0.074,0.111,0.074,0.037,0]),atol=1e-3)))
    assert(np.isclose(b.sum(),1.0,atol=1e-8))
    
def test_009_SimpleJacobian():
    ### Test error cases
    
    # Nonsquare Jacobian doesn't work (for now -- broadcasting is a useful idea)
    try:
        grid = np.mgrid[0:10,0:100:10,0:1000:100].T
        g1 = grid[...,0:2]
        J = t.simple_jacobian(g1)
        assert(False)
    except:
        pass
    
    # Single element along an axis doesn't work
    try:
        grid= (np.mgrid[0:1,0:10].T)
        J = t.simple_jacobian(grid)
        assert(False)
    except:
        pass
    
    # 1-D Jacobian
    grid = np.zeros([5,1])
    grid[2,0] = 1
    J = t.simple_jacobian(grid)
    assert(np.all(J.shape == np.array([4,1,1])))
    assert(all(J[...,0,0] == [0,1,-1,0]))
    
    # 2-D Jacobian
    grid = np.zeros([4,4,2])
    # X component is 1 at X=2,Y=1
    grid[1,2,0] = 1
    # Y component is 0.5 at X=2,Y=2
    grid[2,2,1] = 0.5
    J = t.simple_jacobian(grid)
    assert(np.all(J.shape == np.array([3,3,2,2])))
    # Check X component of X derivative
    assert(np.all(  J[...,0,0] == 
           np.array( [[0,0.5,-0.5],[0,0.5,-0.5],[0,0,0]] )
           ))
    # Check X component of Y derivative 
    assert(np.all( J[...,0,1] == 
            np.array( [[0,0.5,0.5],[0,-0.5,-0.5],[0,0,0]] )
            ))
    # Check Y component of X derivative
    assert(np.all( J[...,1,0] ==
            np.array( [[0,0,0],[0,0.25,-0.25],[0,0.25,-0.25]])
            ))
    # Check Y component of Y derivative
    assert(np.all( J[...,1,1]  == 
            np.array( [[0,0,0],[0,0.25,0.25],[0,-0.25,-0.25]])
            ))
            
    # 3-D Jacobian (exercises N-D code)
    # 4x4x4 grid of 3-vectors
    grid = np.zeros([4,4,4,3])
    # X component is 1 at X=2,Y=1,Z=1
    grid[1,1,2,0] = 1
    # Y component is 2 at X=1,Y=2,Z=2
    grid[1,2,2,1] = 2
    # Z component is 3 at X=2,Y=2,Z=2
    grid[2,2,2,2] = 3
    J = t.simple_jacobian(grid)
    assert(np.all(J.shape == np.array([3,3,3,3,3])))
    # Check X component of X derivative - horizontal
    assert(np.all(  J[...,0,0] ==
        np.array( [
                    [[0.,  0.25, -0.25],
                     [0.,  0.25, -0.25],
                     [0.,  0.,    0.  ]],
                    
                    [[0.,  0.25, -0.25],
                     [0.,  0.25, -0.25],
                     [0.,  0.,    0.  ]],
                    
                    [[0.,  0.,    0.  ],
                     [0.,  0.,    0.  ],
                     [0.,  0.,    0.  ]]
                    ]
            )
        ))
    # Check X component of Y derivative - vertical
    assert(np.all( 
        np.isclose(
            J[...,0,1],
            np.array( [[[ 0.,  0.25,  0.25],
                        [ 0., -0.25, -0.25],
                        [ 0.,  0.,    0.  ]
                        ],
                       [[ 0.,  0.25,  0.25],
                        [ 0., -0.25, -0.25],
                        [ 0.,  0,     0.  ]
                        ],
                       [[ 0.,  0.,    0.  ],
                        [ 0.,  0.,    0.  ],
                        [ 0.,  0.,    0.  ]
                        ]
                       ]
                     ),
            atol=1e-10
            )
        )
        )
    # Check X component of Z derivative - cross-plane
    assert(np.all( 
        np.isclose(
            J[...,0,2],
            np.array( [[[ 0.,  0.25,  0.25],
                        [ 0.,  0.25,  0.25],
                        [ 0.,  0.,    0.  ]
                        ],
                       [[ 0., -0.25, -0.25],
                        [ 0., -0.25, -0.25],
                        [ 0.,  0,     0.  ]
                        ],
                       [[ 0.,  0.,    0.  ],
                        [ 0.,  0.,    0.  ],
                        [ 0.,  0.,    0.  ]
                        ]
                       ]
                     ),
            atol=1e-10
            )
        )
        )
    # Check Y component of X derivative - horizontal
    assert(np.all( 
        np.isclose(
            J[...,1,0],
            np.array( [[[ 0.,  0.  ,  0.  ],
                        [ 0.,  0.5,  -0.5 ],
                        [ 0.,  0.5,  -0.5 ]
                        ],
                       [[ 0.,  0.,    0.  ],
                        [ 0.,  0.5,  -0.5 ],
                        [ 0.,  0.5,  -0.5 ],
                        ],
                       [[ 0.,  0.,    0.  ],
                        [ 0.,  0.,    0.  ],
                        [ 0.,  0.,    0.  ]
                        ]
                       ]
                     ),
            atol=1e-10
            )
        )
        )
    # Check Y component of Y derivative - vertical
    assert(np.all( 
        np.isclose(
            J[...,1,1],
            np.array( [[[ 0.,  0.  ,  0.  ],
                        [ 0.,  0.5,   0.5 ],
                        [ 0., -0.5,  -0.5 ]
                        ],
                       [[ 0.,  0,     0.  ],
                        [ 0.,  0.5,   0.5 ],
                        [ 0., -0.5,  -0.5 ]
                        ],
                       [[ 0.,  0.,    0.  ],
                        [ 0.,  0.,    0.  ],
                        [ 0.,  0.,    0.  ]
                        ]
                       ]
                     ),
            atol=1e-10
            )
        )
        )
     # Check Y component of Z derivative - cross-plane
    assert(np.all( 
        np.isclose(
            J[...,1,2],
            np.array( [[[ 0.,  0.  ,  0.  ],
                        [ 0.,  0.5,   0.5 ],
                        [ 0.,  0.5,   0.5 ]
                        ],
                       [[ 0.,  0,     0.  ],
                        [ 0., -0.5,  -0.5 ],
                        [ 0., -0.5,  -0.5 ],
                        ],
                       [[ 0.,  0.,    0.  ],
                        [ 0.,  0.,    0.  ],
                        [ 0.,  0.,    0.  ]
                        ]
                       ]
                     ),
            atol=1e-10
            )
        )
        )
    # Check X component of Z derivative - horizontal
    assert(np.all( 
        np.isclose(
            J[...,2,0],
            np.array( [
                       [[ 0.,  0.,    0.  ],
                        [ 0.,  0.,    0.  ],
                        [ 0.,  0.,    0.  ]
                        ],
                       [[ 0.,  0.,    0.  ],
                        [ 0.,  0.75, -0.75],
                        [ 0.,  0.75, -0.75]
                        ],
                       [[ 0.,  0.,    0.  ],
                        [ 0.,  0.75, -0.75],
                        [ 0.,  0.75, -0.75]
                        ]
                       ]
                     ),
            atol=1e-10
            )
        )
        )
    # Check Y component of Z derivative - vertical
    assert(np.all( 
        np.isclose(
            J[...,2,1],
            np.array( [
                       [[ 0.,  0.,    0.  ],
                        [ 0.,  0.,    0.  ],
                        [ 0.,  0.,    0.  ]
                        ],
                       [[ 0.,  0.,    0.  ],
                        [ 0.,  0.75,  0.75],
                        [ 0., -0.75, -0.75]
                        ],
                       [[ 0.,  0.,    0.  ],
                        [ 0.,  0.75,  0.75],
                        [ 0., -0.75, -0.75]
                        ]
                       ]
                     ),
            atol=1e-10
            )
        )
        )
    # Check Z component of Z derivative - cross-plane
    assert(np.all( 
        np.isclose(
            J[...,2,2],
            np.array( [
                       [[ 0.,  0.,    0.  ],
                        [ 0.,  0.,    0.  ],
                        [ 0.,  0.,    0.  ]
                        ],
                       [[ 0.,  0.,    0.  ],
                        [ 0.,  0.75,  0.75],
                        [ 0.,  0.75,  0.75]
                        ],
                       [[ 0.,  0.,    0.  ],
                        [ 0., -0.75, -0.75],
                        [ 0., -0.75, -0.75]
                        ]
                       ]
                     ),
            atol=1e-10
            )
        )
        )
    
def test_010_jump_detect():
    # Just like simple_jacobian, test 1, 2, and 3 dimensiona cases.
    
    # 1-D typical case:
    J1 = np.array([[[1]],[[1]],[[30]],[[1]]])
    assert( np.all(J1.shape == np.array([4,1,1])))
    jf = t.jump_detect(J1)
    assert( np.all( jf == np.array([1,1,0,1])))
    
    # 1-D end case:
    J1 = np.array([[[30]],[[1]],[[1]],[[11]]])
    jf = t.jump_detect(J1)
    assert( np.all( jf == np.array([0,1,1,0])))
   
    J1 = np.array([[[30]],[[1]],[[1]],[[10]]])
    jf = t.jump_detect(J1)
    assert( np.all( jf == np.array([0,1,1,1])))
   
    
    # 2-D typical  case:
    grid = np.zeros([5,5,2])
    grid[1,2,0] = 1
    grid[2,2,1] = 0.5
    J2 = t.simple_jacobian(grid)
    jf = t.jump_detect(J2)
    assert( np.all( jf == 
                    np.array([
                        [1,1,1,1],
                        [1,1,1,1],
                        [1,0,0,1],
                        [1,1,1,1]
                        ])
                    )
            )
    
    # 3-D case: just dup the 2-D case along an axis.
    # Simple test exercises 3-D code and makes sure it gets 
    # the right answer in at least a trivial case.
    grid = np.zeros((5,5,5,3))
    grid[1,2,:,0] = 1
    grid[2,2,:,1] = 0.5
    J3 = t.simple_jacobian(grid)
    jf = t.jump_detect(J3)
    for i in range(4):
        assert( np.all( jf[:,:,i] ==
                       np.array([
                           [1,1,1,1],
                           [1,1,1,1],
                           [1,0,0,1],
                           [1,1,1,1]
                           ])
                       )
               )
    
    
def test_011_svd2x2_decompose():
    # Test on the identity matrix
    M = np.array(((1,0),(0,1)))
    U = np.zeros((2,2))
    V = np.zeros((2,2))
    S = np.zeros(2)
    t.svd2x2(M,U,S,V)
                       
        

    
        

    
    
     
   
    
    
    
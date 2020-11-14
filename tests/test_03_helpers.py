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
    v1 = apply_boundary(vec,[3,3],bound='e',rint=False)
    # make sure non-rint version doesn't change original
    assert all(v0.flat==vec.flat)
    assert all(v1[:,1]==2)
    assert all(v1[:,0]==[0,2,2,2,0,0,0,0,0,0])
    
    v1 = apply_boundary(vec,[3,3],bound='p',rint=False)
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
    
    
   
    
    
    
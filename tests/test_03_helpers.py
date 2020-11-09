#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 22:58:09 2020

Tests for the interpolation/helper routines in helpers.py

(for use with pytest)

"""

import numpy as np
from transform.helpers import apply_boundary
import copy

def test_001_apply_boundary():
    
    vec = np.array([[0,2],[2,2],[3,2],[4,2],[-1,2],[-2,2],[-0.49,2],[-0.51,2],[-0.1,2],[-0.9,2]])
    
    try:
        v1 = apply_boundary(vec,[3,3],bound='f')
        raise AssertionError("apply_boundary should have thrown an exception")
    except:
        pass
    
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
    pass

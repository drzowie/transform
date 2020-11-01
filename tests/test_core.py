#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pytest test suite for the core module of Transform
"""

import transform as t
import numpy as np

def test_001_transform_constructor():
    try:
        a = t.Transform()
    except AssertionError:
        return(False)
    else: 
        return(False)
    

def test_002_identity_constructor():
    try:
        a = t.Identity()
    except: 
        return(False)
    
    s = f"{a}"
    return s=="Transform( Identity )"

def test_003_identity_action():
    a = t.Identity()
    b = np.ndarray([1,2])
    c = a.apply(b)
    return(b==c)

def test_004_inverse_constructor():
    try:
        a = t.Inverse()
    except TypeError:
        a = t.Identity()
        b = t.Inverse(a)
        return (f"{a}"=="Transform( Inverse Identity )")
    else:
        return(False)
        
        
    


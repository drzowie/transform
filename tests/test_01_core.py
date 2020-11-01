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
        return
    else:
        assert( False )# should have died

def test_002_identity_constructor():
        a = t.Identity()
        assert(f"{a}" == "Transform( Identity )")

def test_003_identity_apply():
    a = t.Identity()
    b = np.ndarray([1,2])
    c = a.apply(b)
    assert( np.all(b==c) )

def test_004_inverse_constructor():
    try:
        a = t.Inverse()
    except TypeError:
        a = t.Identity()
        b = t.Inverse(a)
        assert (f"{b}"=="Transform( Inverse Identity )")
    else:
        assert (False)  # Should have thrown an error
        
def test_005_inverse_inverse():
    a = t.Identity()
    b = t.Inverse(a)
    c = t.Inverse(b)
    assert ( f"{c}" == "Transform( Inverse Inverse Identity )")

def test_006_method_inverse():
    a = t.Identity()
    b = a.inverse()
    c = b.inverse()
    assert ( f"{c}" == f"{a}"  and  f"{b}"=="Transform( Inverse Identity )")

def test_007_composition_constructor():
    a = t.Identity()
    b = a.inverse()
    c = t.Composition(a,b)
    d = c.inverse()
    assert ( f"{c}" == "Transform( ( (Identity) o (Inverse Identity) ) )"\
        and  f"{d}" == "Transform( Inverse ( (Identity) o (Inverse Identity) ) )")



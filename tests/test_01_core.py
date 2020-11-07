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
    
    a = t.Identity()
    b = t.Inverse(a)
    c = t.Inverse(b)
    assert ( f"{c}" == "Transform( Inverse Inverse Identity )")

def test_006_inverse_method():
    a = t.Identity()
    b = a.inverse()  # Identity is idempotent -- should get same transform back
    assert ( f"{b}" == f"{a}"  and  f"{b}"=="Transform( Identity )")
    a = t.PlusOne_()
    assert ( f"{a}" == "Transform( _PlusOne )")

    # PlusOne_ is non-idempotent - should get Inverse
    b = a.inverse()
    assert( f"{b}" == "Transform( Inverse _PlusOne )")
    
    # Inverting the inverse via method should unwrap the inversion
    c = b.inverse()
    assert( f"{a}" == f"{c}")
    

def test_007_composition():
    a = t.PlusOne_()
    b = a.inverse()
    try:
        c = t.Composition(a,b)
        assert False,"composition of a non-list should throw an error"
    except:
        pass
    
    # Construct an inverse
    c = t.Composition([a,b])
    d = c.inverse()
    assert ( f"{c}" == "Transform( ( (_PlusOne) o (Inverse _PlusOne) ) )"\
        and  f"{d}" == "Transform( Inverse ( (_PlusOne) o (Inverse _PlusOne) ) )")
        
    # Check that Composition flattens compositions into one list
    e = t.Composition([c,c])
    assert( f"{e}" == "Transform( ( (_PlusOne) o (Inverse _PlusOne) o (_PlusOne) o (Inverse _PlusOne) ) )")

def test_008_ArrayIndex():
    a = t.ArrayIndex()
    assert(a.idim==0)
    assert(a.odim==0)
    data = np.array([0,1,2,3,4])
    data2 = a.apply(data)
    assert( np.all(data2 == data[::-1]) ) 
    


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 22:48:57 2020

@author: zowie
"""

import transform as t
import numpy as np

def test_001_Linear_pre_post():
    a = t.Linear(pre=np.array([1,2]))
    assert( isinstance(a,t.Linear), 'Constructor made a pre-offset Linear' )
    d1 = np.array([0,0])
    d2 = a.apply( d1 )
    assert( isinstance(d2,np.ndarray), "apply returned an NDarray" )
    assert( d2.shape == [2], "return value was correct size" )
    assert( all( d2 == np.array([1,2]) ), "return value was correct value ")
    d3 = a.invert( d2 )
    assert( all( d1 == d3 ),  "reverse transform worked" )
    b = a.inverse()
    d4 = b.apply(d2)
    assert( all( d1==d4), "inverse transform worked" )
    
    a = t.Linear(post=np.array([1,2]))
    d1 = np.array([[0,0],[10,20],[30,40]]).transpose()
    d2 = a.apply( d1 )
    assert( isinstance(d2, np.ndarray) )
    assert( d2.shape == d1.shape )
    assert( all( d2 == d1 + np.ndarray([1,2])) )
    d3 = a.inverse().apply(d2)
    assert(all(d3==d1))
    
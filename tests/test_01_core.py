#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pytest test suite for the core module of Transform
"""

import transform as t
import numpy as np
import astropy.io.fits
#import sunpy
#import sunpy.map


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
    b = np.array([1,2])
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
    
    # Check that Composition tracks dimensions correctly.  In particular, idim/odim propagates through
    # transforms with 0 dim to the first nonzero one
    b = t.Identity()
    c = t.Composition([b,b])
    assert(c.idim==0 and c.odim==0)
    
    c = t.Composition([a,b])
    assert(c.idim==1 and c.odim==1)
    
    c = t.Composition([b,a])
    assert(c.idim==1 and c.odim==1)
    
    
    

def test_008_wrap():
    a = t.PlusOne_()
    b = t.Identity()
    c = t.Wrap(b,a)  
    assert( f"{c}" == "Transform( ( (Inverse _PlusOne) o (Identity) o (_PlusOne) ) )")



def test_009_ArrayIndex():
    a = t.ArrayIndex()
    assert(a.idim==0)
    assert(a.odim==0)
    data = np.array([0,1,2,3,4])
    data2 = a.apply(data)
    assert( np.all(data2 == data[::-1]) ) 
    
      
def test_010_WCS():
    a = t.WCS('sample.fits')
    assert(a.idim == 2)
    assert(a.odim == 2)
    assert(len(a.itype)==2 and len(a.otype)==2 and len(a.iunit)==2 and len(a.ounit)==2)
    assert(a.itype == ['X','Y'])
    assert(a.iunit==['Pixels','Pixels'])
    assert(a.ounit==['arcsec','arcsec'])
    assert(a.otype==['Solar-X','Solar-Y'])
    assert( np.all( np.isclose ( a.apply([[0,0]],0), np.array([[-386.15825,  -676.1092]]), atol=1e-4 ) ) )

def test_011_DataWrapper():
    # string should get read as a FITS file
    a = t.DataWrapper('sample.fits')
    assert( isinstance(a.data, np.ndarray))
    assert( isinstance(a.header, astropy.io.fits.header.Header) )
    assert(a.header['NAXIS']==2)
    assert(a.header['NAXIS1'] == a.data.shape[1])
    assert(a.header['NAXIS2'] == a.data.shape[0])
    assert(a.wcs.wcs.naxis == 2)
    assert(a.wcs.pixel_shape[0] == a.header['NAXIS1'])
    assert(a.wcs.pixel_shape[1] == a.header['NAXIS2'])
    b = a.export()
    assert(isinstance(b,t.DataWrapper)) # original was a string; result is a DataWrapper
    
    # Non-file string should fail
    try:
        a = t.DataWrapper('blargle.notafitsfile.fits')
        assert(False)
    except:
        pass
    
    # FITS object should work right  
    fits = astropy.io.fits.open('sample.fits')
    a = t.DataWrapper(fits[0])
    assert(isinstance(a.data, np.ndarray))
    assert(isinstance(a.header, astropy.io.fits.header.Header) )
    assert(isinstance(a.wcs, astropy.wcs.wcs.WCS ))
    assert(a.header['NAXIS']==2)
    a.header['TEST'] = "Testing testing"
    b = a.export()
    assert( isinstance(b, astropy.io.fits.hdu.image.PrimaryHDU))
    assert( b.header['TEST'] == "Testing testing" )
    
    # FITS file object should pull the first HDU
    a = t.DataWrapper(fits)
    assert(isinstance(a.data, np.ndarray))
    assert(isinstance(a.header, astropy.io.fits.header.Header) )
    assert(isinstance(a.wcs, astropy.wcs.wcs.WCS ))
    assert(a.header['NAXIS']==2)
    b = a.export()
    assert( isinstance(b, astropy.io.fits.hdu.image.PrimaryHDU))

    # Feeding in a dictionary should work right
    f0 = {'header':fits[0].header, 'data':fits[0].data}
    a = t.DataWrapper(f0)
    assert(isinstance(a.data, np.ndarray))
    assert(isinstance(a.header, astropy.io.fits.header.Header) )
    assert(isinstance(a.wcs, astropy.wcs.wcs.WCS ))
    assert(a.header['NAXIS']==2)
    b = a.export()
    assert( isinstance(b,dict))
    assert( isinstance(b['data'],np.ndarray))
    assert( isinstance(b['header'], astropy.io.fits.header.Header))
    assert( isinstance(b['wcs'], astropy.wcs.wcs.WCS))
    
    # Feeding in just a header should work okay
    hdr = dict(fits[0].header)
    
    # Delete COMMENT and HISTORY tag from sample to work around problem with astropy 4.1
    # (multiline COMMENT and HISTORY fieldscan't be re-imported)
    del hdr['HISTORY']
    del hdr['COMMENT']
    a = t.DataWrapper(hdr)
    assert(isinstance(a.header,dict))
    assert(isinstance(a.wcs,astropy.wcs.wcs.WCS))
    assert( a.data is None )
    b = a.export()
    assert( isinstance(b,t.DataWrapper ) )
    
    # Sunpy map tests should go here ... eventually.
    # For now it doesn't make sense since we don't export to maps yet.
    

def test_012_resample():
    # Since the main engine is in helpers, this is really just a basic
    # functionality test.
    a = np.zeros([7,5])  # 7 tall, 5 wide
    a[2,2] = 1
    trans = t.Identity()
    b = trans.resample(a,method='nearest')
    assert(b.shape==(7,5))
    assert(np.all(b==a))
    
    b = trans.resample(a,shape=[5,5],method='nearest')
    assert(b.shape==(5,5))
    assert(np.all(a[0:5,0:5]==b))
    
    
    
    
    
    
    
    

    

    
    
    
           
    
           
    

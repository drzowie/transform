# -*- coding: utf-8 -*-
"""
pytest test suite for the core module of Transform
"""

import transform as t
import numpy as np
import astropy.io.fits
import astropy.wcs
from ndcube import NDCube
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
    
def test_007b_composition_method():
    a = t.PlusOne_()
    b = a.inverse()
    c = a.composition(b)
    assert(f"{c}" == "Transform( ( (_PlusOne) o (Inverse _PlusOne) ) )")

    d = a.composition([b])
    assert(f"{d}" == "Transform( ( (_PlusOne) o (Inverse _PlusOne) ) )")
    
    e = a.composition([b,b])
    assert(f"{e}" == "Transform( ( (_PlusOne) o (Inverse _PlusOne) o (Inverse _PlusOne) ) )")
    

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
    
def test_009b__FITSHeaderFromDict():
    h = t.FITSHeaderFromDict({'A':3,
                                         'B':'foo',
                                         'HISTORY':'hey there',
                                         '':['one','two','three'],
                                         'COMMENT':"foo\nbar\nbaz"})
    assert(h['A']==3)
    assert(h['B']=='foo')
    assert(h[''][0]=='one')
    assert(h[''][1]=='two')
    assert(h[''][2]=='three')
    assert(f"{h['']}" == 'one\ntwo\nthree')
    assert(h['HISTORY']=='hey there')
    assert(f"{h['HISTORY']}"=='hey there')
    assert(h['COMMENT'][0]=='foo')
    assert(h['COMMENT'][1]=='bar')
    assert(h['COMMENT'][2]=='baz')
    assert(f"{h['COMMENT']}"=='foo\nbar\nbaz')
    
    hh = t.FITSHeaderFromDict(h)
    assert(h==hh)
    

    
def test_010_WCS():
    a = t.WCS(astropy.io.fits.open('sample.fits'))
    assert(a.idim == 2)
    assert(a.odim == 2)
    assert(len(a.itype)==2 and len(a.otype)==2 and len(a.iunit)==2 and len(a.ounit)==2)
    assert(a.itype == ['X','Y'])
    assert(a.iunit==['Pixels','Pixels'])
    assert(a.ounit==['arcsec','arcsec'])
    assert(a.otype==['Solar-X','Solar-Y'])
    assert( np.all( np.isclose ( a.apply([[0,0]],0), np.array([[-386.15825,  -676.1092]]), atol=1e-4 ) ) )

def test_011_DataWrapper():
    # string should throw an error
    try:
        a = t.DataWrapper('sample.fits')
        assert(False)
    except: 
        pass
    
    # FITS file should load okay
    a = t.DataWrapper( astropy.io.fits.open('sample.fits'))
    assert( isinstance(a.data, np.ndarray))
    assert( isinstance(a.meta, astropy.io.fits.header.Header) )
    assert(a.meta['NAXIS']==2)
    assert(a.meta['NAXIS1'] == a.data.shape[1])
    assert(a.meta['NAXIS2'] == a.data.shape[0])
    assert(a.wcs.wcs.naxis == 2)
    assert(a.wcs.pixel_shape[0] == a.meta['NAXIS1'])
    assert(a.wcs.pixel_shape[1] == a.meta['NAXIS2'])
    
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
    assert(isinstance(a.meta, astropy.io.fits.header.Header) )
    assert(isinstance(a.wcs, astropy.wcs.wcs.WCS ))
    assert(a.meta['NAXIS']==2)

    
    # FITS file object should pull the first HDU
    a = t.DataWrapper(fits)
    assert(isinstance(a.data, np.ndarray))
    assert(isinstance(a.meta, astropy.io.fits.header.Header) )
    assert(isinstance(a.wcs, astropy.wcs.wcs.WCS ))
    assert(a.meta['NAXIS']==2)

    # Feeding in a dictionary should work right
    f0 = {'header':fits[0].header, 'data':fits[0].data}
    a = t.DataWrapper(f0)
    assert(isinstance(a.data, np.ndarray))
    assert(isinstance(a.meta, astropy.io.fits.header.Header) )
    assert(isinstance(a.wcs, astropy.wcs.wcs.WCS ))
    assert(a.meta['NAXIS']==2)
    
    # Feeding in just a header should work okay
    hdr = dict(fits[0].header)
    
    #a = t.DataWrapper(hdr)
    #'DataWrapper: requires an NDCube object, or a', 'np data array.')
    try:
        a = t.DataWrapper(hdr)
        assert(False)
    except: 
        pass
    assert(isinstance(a.meta,astropy.io.fits.header.Header))
    assert(isinstance(a.wcs,astropy.wcs.wcs.WCS))
    #assert( f"{b}" == "Transform( Inverse _PlusOne )")
    #assert( a.data is None )
   
    # a tuple should work  right
    a = t.DataWrapper((fits[0].data,fits[0].header))
    assert(isinstance(a.data, np.ndarray))
    assert(isinstance(a.meta, astropy.io.fits.header.Header))
    assert(isinstance(a.wcs, astropy.wcs.wcs.WCS))
    assert(a.meta['NAXIS']==2)

    # A template should override the input
    a = t.DataWrapper((fits[0].data, fits[0].header),template=
                      {'CRPIX1':99})

    assert(a.meta['CRPIX1']==99) # this fails because template is not doing anything.
    assert(fits[0].header['CRPIX1'] != 99)
    assert(a.meta['CRPIX2'] == fits[0].header['CRPIX2'])
    
    assert(a.wcs.wcs.naxis == 2)
    assert(a.wcs.pixel_shape[0] == a.meta['NAXIS1'])
    assert(a.wcs.pixel_shape[1] == a.meta['NAXIS2'])
    assert(a.wcs.wcs.crpix[0] == a.meta['CRPIX1'])
    assert(a.wcs.wcs.crpix[1] == a.meta['CRPIX2'])


    #MT(test wcs)

    # Check that CTYPE/CUNIT are exported from the WCS to the head if need be
    a = t.DataWrapper((fits[0].data, fits[0].header))
    a.meta = None
    try:
        isinstance(a.meta, astropy.io.fits.header.Header)
        assert(False)
    except: 
        pass
    
    # make meta data with DataTemplate and add it to a
    tempTemplate = t.DataTemplate(a)
    tempTemplate.wcs2head()
    a.meta = tempTemplate.header
    #a.wcs2head()

    assert(isinstance(a.meta, astropy.io.fits.header.Header))
    assert(a.meta['NAXIS']==2)
    assert(a.meta['CTYPE1'] == fits[0].header['CTYPE1'])
    assert(a.meta['CRPIX1'] == fits[0].header['CRPIX1'])
    
    

    # Sunpy map tests should go here ... eventually.
    # For now it doesn't make sense since we don't export to maps yet.
    

def test_012_resample():
    # Since the main engine is in helpers (resmple is a wrapper), this is 
    # really just a basic functionality test that the wrapper works right.  
    # Makes use of Scale, which is in Basic -- but since this is a test, not 
    # the main code, so there's no circular dependence.
    
    # Simple 7x5 array with a single nonzero spot
    a = np.zeros([7,5])  # 7 tall, 5 wide
    a[2,2] = 1
    
    # Check that identity and shape specs work
    trans = t.Identity()
    b = trans.resample(a,method='nearest')
    assert(b.shape==(7,5))
    assert(np.all(b==a))

    b = trans.resample(a,shape=[5,5],method='nearest')
    assert(b.shape==(5,5))
    assert(np.all(a[0:5,0:5]==b))

    # Try scaling up by a factor of 2.5 -- this should make a 3x3 square
    # of ones
    trans = t.Scale(2.5,post=[2,2],pre=[-2,-2])
    b = trans.resample(a,method='neares')
    assert(b.shape==(7,5))
    checkval = np.zeros([7,5])
    checkval[1:4,1:4] = 1
    assert(np.all(b==checkval))

    # Check that anisotropy works in the correct direction
    trans = t.Scale([1,2.5],post=[2,2],pre=[-2,-2])
    b = trans.resample(a,method='nearest')
    assert(b.shape==(7,5))
    checkval = np.zeros([7,5])
    checkval[1:4,2]=1
    assert(np.all(b==checkval))
    
    
def test_013_remap():
    # remap is all about scientific coordinates, so we have to gin up some 
    # FITS headers.
    # The test article (a) is a simple asymmetric cross.
    a2 = np.zeros([7,7])
    a2[1:5,3] = 1
    a2[3,0:5] = 1
    
    a2hdr = {
        'SIMPLE':'T',       'NAXIS':2, 
        'NAXIS1':7,         'NAXIS2':7,
        'CRPIX1':4,         'CRPIX2':4,
        'CRVAL1':0,         'CRVAL2':0,
        'CDELT1':1,         'CDELT2':1,
        'CUNIT1':'pixel',   'CUNIT2':'pixel',
        'CTYPE1':'X',       'CTYPE2':'Y'
    }
    
    trans = t.Identity()
    b2 = trans.remap({'data':a2,'header':a2hdr},method='nearest')
    #assert(np.all(b['data']==a))
    assert(np.all(b2.data==a2))
    
    # This tests actual transformation and also broadcast since 
    # PlusOne_ is 1D and a is 2D
    # 
    # The autoscaling ought to completely undo the plus-one, so this should
    # be a no-op except for incrementing CRVAL1.
    trans = t.PlusOne_()
    b2 = trans.remap({'data':a2,'header':a2hdr},method='nearest')
    
    assert(np.all(a2 == b2.data))
    assert(b2.meta['CRPIX1']==4)
    assert(b2.meta['CRPIX2']==4)
    assert(b2.meta['CRVAL1']==1)
    assert(b2.meta['CRVAL2']==0)
    assert(b2.meta['CDELT1']==1)
    assert(b2.meta['CDELT2']==1)


    # This again tests autoscaling - scale(3) should expand everything, 
    # but autoscaling should put it back.
    trans = t.Scale(3,dim=2)
    b2 = trans.remap({'data':a2,'header':a2hdr},method='nearest')
    assert(np.all(a2==b2.data))
    assert(b2.meta['CRPIX1']==4)
    assert(b2.meta['CRPIX2']==4)
    assert(b2.meta['CRVAL1']==0)
    assert(b2.meta['CRVAL2']==0)
    print(f"CDELT1 is {b2.meta['CDELT1']}")
    assert(np.isclose(b2.meta['CDELT1'],3,atol=1e-10))
    assert(np.isclose(b2.meta['CDELT2'],3,atol=1e-10))


    # Test manual scaling of the output by setting the output_range
    # This *should* be a no-op
    # Note: output range is to/from the *edge* of the outermost pixel,
    # so the span is the full span of the image
    trans = t.Identity()
    b2 = trans.remap({'data':a2,'header':a2hdr},
                    method='nearest',
                    output_range=[[-3.5,3.5],[-3.5,3.5]]
                    )
    assert(np.all(a2==b2.data))
    assert(b2.meta['CRPIX1']==4)
    assert(b2.meta['CRPIX2']==4)
    assert(b2.meta['CRVAL1']==0)
    assert(b2.meta['CRVAL2']==0)
    print(f"CDELT1 is {b2.meta['CDELT1']}")
    assert(np.isclose(b2.meta['CDELT1'],1,atol=1e-10))
    assert(np.isclose(b2.meta['CDELT2'],1,atol=1e-10))

    b2 = trans.remap({'data':a2,'header':a2hdr},
                    method='nearest',
                    output_range=[[-3.5/3,3.5/3],[-3.5/3,3.5/3]]
                    )
    expando = np.array([
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0, 0]
            ])
    assert(np.all(np.isclose(expando,b2.data,atol=1e-10)))
    assert(b2.meta['CRPIX1']==4)
    assert(b2.meta['CRPIX2']==4)
    assert(b2.meta['CRVAL1']==0)
    assert(b2.meta['CRVAL2']==0)
    assert(np.isclose(b2.meta['CDELT1'],1/3.0,atol=1e-10))
    assert(np.isclose(b2.meta['CDELT2'],1/3.0,atol=1e-10))

    trans = t.Scale(3,dim=2)
    b2 = trans.remap({'data':a2,'header':a2hdr},
                    method='nearest',
                    output_range=[[-3.5,3.5],[-3.5,3.5]]
                    )
    assert(np.all(np.isclose(expando,b2.data,atol=1e-10)))
    assert(np.isclose(b2.meta['CDELT1'],1,atol=1e-10))
    assert(np.isclose(b2.meta['CDELT2'],1,atol=1e-10))

    
def test_014_ndcube():
    a = np.zeros([7,7])
    a[1:5,3] = 1
    a[3,0:5] = 1
    ahdr = {
        'SIMPLE':'T',       'NAXIS':2, 
        'NAXIS1':7,         'NAXIS2':7,
        'CRPIX1':4,         'CRPIX2':4,
        'CRVAL1':0,         'CRVAL2':0,
        'CDELT1':1,         'CDELT2':1,
        'CUNIT1':'pixel',   'CUNIT2':'pixel',
        'CTYPE1':'X',       'CTYPE2':'Y'
    }
    input_wcs = astropy.wcs.WCS(ahdr)
    testcube=NDCube(a,wcs=input_wcs)
    outcube=t.DataWrapper(testcube)

    assert(type(outcube)==type(testcube))
    
    # FITS file should load okay
    a = t.DataWrapper( astropy.io.fits.open('sample.fits'))
    
    assert( isinstance(outcube.data, np.ndarray))
    assert( isinstance(outcube.wcs, astropy.wcs.wcs.WCS) )
    assert(a.meta['NAXIS']==2)
    assert(a.meta['NAXIS1'] == a.data.shape[1])
    assert(a.meta['NAXIS2'] == a.data.shape[0])
    assert(a.wcs.wcs.naxis == 2)
    assert(a.wcs.pixel_shape[0] == a.meta['NAXIS1'])
    assert(a.wcs.pixel_shape[1] == a.meta['NAXIS2'])
    
    # Non-file string should fail
    try:
        a = t.DataWrapper('blargle.notafitsfile.fits')
        assert(False)
    except:
        pass


    # open a fits and get the inners and test output
    HDUList = astropy.io.fits.open('sample.fits')
    input_data=HDUList[0].data
    input_header=HDUList[0].header
    input_wcs = astropy.wcs.WCS(input_header)
    
    testcube2=NDCube(input_data,wcs=input_wcs,meta=input_header)
    outcube2=t.DataWrapper(testcube2)

    assert( type(outcube2)==type(testcube2))
    #assert( isinstance(outcube2, ndcube.ndcube.NDCube))
    assert( isinstance(outcube2.data, np.ndarray))
    assert( isinstance(outcube2.wcs, astropy.wcs.wcs.WCS)) 
    assert( isinstance(outcube2.meta, astropy.io.fits.header.Header)) 
    
    # tuple test
    outcube2tuple = t.DataWrapper((HDUList[0].data,HDUList[0].header))

    # HDU test
    outcubehdulist = t.DataWrapper(HDUList)

    # HEADER test - no data should fail
    try:
        outcubeHeader = t.DataWrapper(HDUList[0].header)
        assert(False)
    except:
        pass 

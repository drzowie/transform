import astropy.wcs
import numpy as np
from ndcube import NDCube
import transform as t

# create some fake data
anparray = np.zeros([7,7])
anparray[1:5,3] = 1
anparray[3,0:5] = 1

ahdr = {
    'SIMPLE':'T',       'NAXIS':2, 
    'NAXIS1':7,         'NAXIS2':7,
    'CRPIX1':4,         'CRPIX2':4,
    'CRVAL1':0,         'CRVAL2':0,
    'CDELT1':1,         'CDELT2':1,
    'CUNIT1':'pixel',   'CUNIT2':'pixel',
    'CTYPE1':'X',       'CTYPE2':'Y'
}

# convert to WCS
awcs = astropy.wcs.WCS(ahdr)

testCube=NDCube(anparray, wcs=awcs)
#print(type(testCube))
aoutcube=t.DataWrapper2(testCube)
#print(type(outcube))



# open a fits and get the inners
HDUList = astropy.io.fits.open('sample.fits')
FITSdata=HDUList[0].data
FITSheader=HDUList[0].header
FITSwcs = astropy.wcs.WCS(FITSheader)

testCube2=NDCube(FITSdata, wcs=FITSwcs, meta=FITSheader)

def test1():
    # test NDCube
    outcube=t.DataWrapper2(testCube)
    print("#######################")
    print("Type:", type(outcube))

def test2():
    # tuple test
    outcube = t.DataWrapper2((HDUList[0].data,HDUList[0].header))
    print("#######################")
    print("Type:", type(outcube))

def test3():
    # HDU test
    outcube = t.DataWrapper2(HDUList)
    print("#######################")
    print("Type:", type(outcube))

def test4():
    # data test
    outcube = t.DataWrapper2(HDUList[0].data)
    print("#######################")
    print("Type:", type(outcube))

def test5():
    # HEADER test - Fails
    outcube = t.DataWrapper2(HDUList[0].header)
    print("#######################")
    print("Type:", type(outcube))


#outarray=[]
#for ii in range(outcube.wcs.naxis):
#    outarray.append(outcube.meta[f"NAXIS{outcube.wcs.naxis-ii}"])
#newarray=np.empty(outarray)
#newarray[:] = np.NaN

def test6():
    # WCS test - Fails
    outcube = t.DataWrapper2(FITSwcs)
    print("#######################")
    print("Type:", type(outcube))

def test7():
    # np test
    outcube = t.DataWrapper2(anparray)
    print("#######################")
    print("Type:", type(outcube))

def test8():
    # dict test
    outcube = t.DataWrapper2(ahdr)
    print("#######################")
    print("Type:", type(outcube))

def test9():
    outcube = t.DataWrapper2((FITSdata, FITSheader),template={'CRPIX1':99})
    print("#######################")
    print(outcube.meta['CRPIX1'])
    

# types
# astropy.wcs.wcs.WCS
# astropy.io.fits.header.Header

#print("Full:", outcube)
#print("#######################")
#print("Data:", outcube.data)
#print("#######################")
#print("wcs:", outcube.wcs)
#print("#######################")
#print("Meta:", outcube.meta)
#print("#######################")
#print("Type:", type(outcube))


def test10():
    def outer_func(outer_number):
        print(outer_number)    
        def inner_add_func(inner_number):
            return inner_number + 1
        outer_number=inner_add_func(outer_number)
        print(outer_number)
        return outer_number



#print(outer_func(5))


#if testCube.meta:
#    print("yep")
#else:
#    print("nope:")


#test1()
#test2()
#test3()
#test4()
#test5()
#test6()
#test7()
#test8()
#test9()
#test10()


#FITSdata, wcs=FITSwcs, meta=FITSheader

print("here")

#print(FITSheader)

templatedict1={'CRPIX1':99, 'CRPIX2':95}
templatedict2={'CRPIX1':99}

def readlist(this, olditem):
    if isinstance(this,dict):
        print("in dict")
        #x = this.keys()

        for x in this:
            print(x)
            print(this[x])
            #FITSwcs.CRPIX1=2
            #FITSheader["CRPIX1"]

    if isinstance(this,list):
        print("in list")
        x = this.keys()

    #print(x)
    #print(len(x))
    #print(x(1))
    
readlist(templatedict2, FITSheader)

#print(FITSheader)

#a = t.DataWrapper2((FITSdata, FITSheader),template={'CRPIX1':99})
#b = t.DataTemplate({'CRPIX1':99})


def test_013_remap():
    # remap is all about scientific coordinates, so we have to gin up some 
    # FITS headers.
    # The test article (a) is a simple asymmetric cross.
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
    
    trans = t.Identity()
    b = trans.remap({'data':a,'header':ahdr},method='nearest')

    trans = t.PlusOne_()
    b = trans.remap({'data':a,'header':ahdr},method='nearest')

    #print("b:", b['data'])
    #print("a:", a)

#test_013_remap()

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
    
trans2 = t.Identity()
b2 = trans2.remap2({'data':a2,'header':a2hdr},method='nearest')

trans2 = t.PlusOne_()
b2 = trans2.remap2({'data':a2,'header':a2hdr},method='nearest')

print("remap2 run")
#print("b2",b2.wcs)
print("b2",b2.meta['CRVAL1'])

#print(b2.data)


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
    
trans = t.Identity()
b = trans.remap({'data':a,'header':ahdr},method='nearest')

trans = t.PlusOne_()
b = trans.remap({'data':a,'header':ahdr},method='nearest')
print("remap run")
#print("b",b.wcs)
print("b",b.header['CRVAL1'])

#print(b.data)


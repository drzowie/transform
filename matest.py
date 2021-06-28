#---how to update packages
#   conda activate base
#   python CalculateObjectStackTime.py

#---How to call a method from a local repository
#   import InstrumentCharacteristics.PixelRotationRate as PixelRotationRate
#   b = PixelRotationRate.Time_For_Object_To_Rotate_Out_Of_Pixel(10)

#---Useful Packages to Import
import numpy as np
import transform as t
import math
import timeit
#import InstrumentCharacteristics.PixelRotationRate as PixelRotationRate
#import matplotlib
#import matplotlib.pyplot as plt
#import sunpy
#from astropy.io.fits import CompImageHDU, HDUList, Header, ImageHDU, PrimaryHDU
#from astropy.wcs import WCS


#---Import data
import astropy.io.fits as fits
data_dir = '../../data/'
#image_file_aia=data_dir+'sdo/aia/l1/0171/2014/10/16/aia_lev1_171a_2014_10_16t06_10_11_34z_image_lev1.fits'
#image_file_swap=data_dir+'proba2/swap/l1/0174/2014/01/01/swap_lv1_20140101_001106.fits'
#image_file_euvi=data_dir+'stereo/euvi/l0/a/img/euvi/20131212/20131212_232530_n7euA.fts'
#image_file_whispr=data_dir+'solo⁩/whispr⁩/l3⁩/orbit03⁩/inner⁩/20190819⁩/psp_L3_wispr_20190819T040031_V1_1221.fits'
image_file_example=data_dir+'example/horsehead.fits'
#image_file_example_eit=data_dir+'example/eit_test.fits'
image_file=image_file_example

#---Open Fits
#openfits(image_file) # data_fits - fits data

#---Data Array Examples
#x332 = np.arange(18.).reshape(3,3,2)
#x33 = np.arange(9.).reshape(3,3)
#x22 = np.arange(4.).reshape(2,2)
#x12 = np.array([1,3])
#x77 = np.array([[3,4,1,1,1,1,7],[2,1,0,0,0,0,0],[1,0,0,1,0,0,0],[1,0,1,1,1,0,0],[1,0,0,1,0,0,0],[1,0,0,0,0,0,1],[6,0,0,0,0,1,5]])


def test1():
    # Singular-value decomposition
    from numpy import array
    from scipy.linalg import svd
    from transform.helpers import apply_boundary
    from transform.helpers import sampleND
    from transform.helpers import interpND
    import copy
    # define a matrix
    A = array([[1, 2], [3, 4], [5, 6]])
    #print(A)
    # SVD
    U, s, VT = svd(A)
    #print(U)
    #print(s)
    #print(VT)

    #import pyximport; pyximport.install()
    #from testCython.svd import svd2x2
    #python ./testCython/setupsvd.py build_ext --inplace
    from svd import svd2x2

    #print(svd2x2(2,3,4,5))
    #from testCython.deforest import map_coordinates
    M = np.array(((1,0),
                  (0,1))).astype(np.float64)
    U = np.zeros((3,2)).astype(np.float64)
    V = np.zeros((2,2)).astype(np.float64)
    S = np.zeros(2).astype(np.float64)
    print(V)
    svd2x2(M,U,S,V)
    print(V)

    print("numpy SVD test")

    Mtest = np.array(((1,0),
                  (0,1))).astype(np.float64)
    Utest = np.zeros((3,2)).astype(np.float64)
    Vtest = np.zeros((2,2)).astype(np.float64)
    Stest = np.zeros(2).astype(np.float64)


    Utest, Stest, Vtest = np.linalg.svd(Mtest, full_matrices=True)
    print(Vtest)





trans = t.Scale([1.5,2])
print(f"trans is a '{trans}'")
a = np.array([1,1])
b = trans.apply(a)
print(f"{a} mapped to {b}")

print(t.svd2x2(1,2,3,4))


#timeit.timeit(test2())
#timeit.timeit(test3())

#---to build the SVD code in place:
#   


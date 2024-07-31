# IMPORTS Standard:
import os
import sys
import numpy as np

# IMPORTS Astro:
from astropy.io import fits
from astropy.table import join

cwd = os.path.dirname(os.path.realpath(__file__))
pd = os.path.abspath(os.path.join(cwd,'..'))
sys.path.append(pd)

# IMPORTS Internal:
from sfft_and_animate import get_sn_info, sn_in_or_out, check_overlap, sfft

# NOTE: Use srun --mem=128GB --pty bash -i to open an interactive
# job before running this in the terminal. 

def test_get_sn_info():
    # Test that the results are what we expect. 
    oid = 20172782
    start,end = 62450.0, 62881.0
    ra,dec = 7.551093401915147, -44.80718106491529

    tra,tdec,tstart,tend = get_sn_info(oid)
    
    assert tstart == start
    assert tend == end
    assert tra == ra
    assert tdec == dec

def test_sn_in_or_out():
    # Test that the two tables do not share any rows. 
    oid = 20172782
    start,end = 62450.0, 62881.0
    band = 'R062'

    in_tab,out_tab = sn_in_or_out(oid,start,end,band)
    joined = join(in_tab,out_tab,join_type='inner')

    assert len(joined) == 0 

def test_check_overlap():
    # Test that for a cutout around an image's center, it returns True.
    # For a cutout around a coordinate in the image, but the cutout spills over the edge, it returns False.
    # For a cutout around a coordinate not in the image, it returns False. 

    path = 'testdata/Roman_TDS_simple_model_R062_6_17.fits.gz'

    with fits.open(path) as hdu:
        x_in, y_in = hdu[0].header['CRPIX1'], hdu[0].header['CRPIX2']

    x_out, y_out = (4089, 4089)

    check1 = check_overlap(x_in,y_in,path,overlap_size=4088,data_ext=1)
    check2 = check_overlap(x_in,y_in,path,overlap_size=4089,data_ext=1)
    check3 = check_overlap(x_out,y_out,path,overlap_size=4088,data_ext=1)

    assert check1
    assert not check2
    assert not check3

# def test_sfft():
#     # Use srun --mem=128GB --pty bash -i to run in an interactive kernel, or it gets OOM killed. 
#     # Test that for two images that do not overlap, it does not proceed with the procedure.
#     # For two of the same image, the result is 0 (or approx. 0)

#     path = 'testdata/Roman_TDS_simple_model_R062_6_17.fits.gz'
#     with fits.open(path) as hdu:
#         ra, dec = hdu[0].header['CRVAL1'], hdu[0].header['CRVAL2']

#     # This test fails, and it should, but it fails because it crashes and 
#     # does not fail elegantly right now. 
#     # test1a, test1b = sfft(ra,dec,'R062',6,17,6,18)
#     test2a, test2b = sfft(ra,dec,'R062',6,17,6,17)

#     with fits.open(test2a) as hdu:
#         img = hdu[0].data

#     mean = np.mean(img)
#     std = np.std(img)

#     # assert test1a is None
#     # assert test1b is None
#     assert isinstance(test2a, str)
#     assert isinstance(test2b, str)
#     assert mean-std < mean < mean+std
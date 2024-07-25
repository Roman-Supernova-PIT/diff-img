# IMPORTS Standard:
import os
import sys

# IMPORTS Astro:
from astropy.io import fits
from astropy.table import join

cwd = os.path.dirname(os.path.realpath(__file__))
pd = os.path.abspath(os.path.join(cwd,'..'))
sys.path.append(pd)

# IMPORTS Internal:
from sfft_and_animate import set_sn, sn_in_or_out, check_overlap

def test_set_sn():
    oid = 20172782
    start,end = 62450.0, 62881.0
    ra,dec = 7.551093401915147, -44.80718106491529

    tra,tdec,tstart,tend = set_sn(oid)
    
    assert tstart == start
    assert tend == end
    assert tra == ra
    assert tdec == dec

def test_sn_in_or_out():
    oid = 20172782
    start,end = 62450.0, 62881.0
    band = 'R062'

    in_tab,out_tab = sn_in_or_out(oid,start,end,band)
    joined = join(in_tab,out_tab,join_type='inner')

    assert len(joined) == 0 

def test_check_overlap():
    path1 = 'testdata/Roman_TDS_simple_model_R062_6_17.fits.gz'
    path2 = 'testdata/Roman_TDS_simple_model_R062_6_18.fits.gz'

    with fits.open(path1) as hdu:
        ra, dec = hdu[0].header['CRVAL1'], hdu[0].header['CRVAL2']

    check1 = check_overlap(ra,dec,path1,path1,overlap_size=4088,data_ext=1)
    check2 = check_overlap(ra,dec,path1,path2,overlap_size=4088,data_ext=1)

    assert check1
    assert not check2
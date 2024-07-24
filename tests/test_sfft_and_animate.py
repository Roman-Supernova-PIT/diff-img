# IMPORTS Standard:
import os
import sys

# IMPORTS Astro:
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

# def test_check_overlap():
# Get two images that do not overlap at all check that False is returned
# Check the same image against itself and check that True is returned 
import os
import argparse
import numpy as np
from phrosty.utils import get_object_instances, get_transient_radec

infodir = os.getenv('SN_INFO_DIR', None)
assert infodir is not None, 'You need to set SN_INFO_DIR as an environment variable.'

def make_object_table(oid):
    ra,dec = get_transient_radec(oid)
    objs = get_object_instances(ra=ra, dec=dec)
    savedir = os.path.join(infodir,oid)
    savepath = os.path.join(savepath,f'{oid}_instances.csv')
    if not os.path.exists(savepath):
        objs.write(savepath, format='csv', overwrite=True)
    else:
        print(f'{savepath} exists. Skipping this step.')

def parse_and_run():
    parser = argparse.ArgumentParser(
        prog='get_object_instances', 
        description='Get all images that contain the RA/Dec of the input transient OID.'
    )

    parser.add_argument(
        'oid',
        type=int,
        help='ID of transient. Used to look up information on transient.'
    )

    args = parser.parse_args()

    make_object_table(args.oid)
    print(f'Table for {args.oid} retrieved and saved!')

if __name__ == '__main__':
    parse_and_run()
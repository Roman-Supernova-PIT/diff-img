import os
import argparse
import time
import tracemalloc
import requests
import pandas as pd
from phrosty.utils import get_transient_radec

infodir = os.getenv('SN_INFO_DIR', None)
assert infodir is not None, 'You need to set SN_INFO_DIR as an environment variable.'

def make_object_table(oid):

    ###################################################################
    # Start tracemalloc. 
    tracemalloc.start()

    ###################################################################
    
    start_time = time.time()

    ra,dec = get_transient_radec(oid)
    
    server_url = 'https://roman-desc-simdex.lbl.gov'
    req = requests.Session()
    result = req.post(f'{server_url}/findromanimages/containing=({ra},{dec})')
    if result.status_code != 200:
        raise RuntimeError(f"Got status code {result.status_code}\n{result.text}")
    
    objs = pd.DataFrame(result.json())[['filter','pointing','sca']]

    oid = str(oid)
    savedir = os.path.join(infodir,oid)
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    savepath = os.path.join(savedir,f'{oid}_instances.csv')
    if not os.path.exists(savepath):
        objs.to_csv(savepath, sep=',', index=False)
    else:
        print(f'{savepath} exists. Skipping this step.')

    print(f'RUNTIMEPRINT get_object_instances.py: {time.time()-start_time}')

    ###################################################################
    # Print tracemalloc.
    current, peak = tracemalloc.get_traced_memory()
    print(f'MEMPRINT get_object_instances.py: Current memory = {current}, peak memory = {peak}')

    ###################################################################

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
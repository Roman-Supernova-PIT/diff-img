import os
import argparse
from astropy.table import Table, vstack
from phrosty.utils import make_object_table, get_templates, get_science, get_roman_bands

infodir = os.getenv('SN_INFO_DIR', None)
assert infodir is not None, 'You need to set SN_INFO_DIR as an environment variable.'

def make_object_tables(oid,n_templates):

    """
    Given an SN OID and number of template images, generate 2 text files:
    one with the templates, and one with all the science images. 
    """

    objs = make_object_table(oid)

    oid = str(oid)
    savedir = os.path.join(infodir,oid)
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    savepath = os.path.join(savedir,f'{oid}_instances.csv')
    if not os.path.exists(savepath):
        objs.to_csv(savepath, sep=',', index=False)
    else:
        print(f'{savepath} exists. Skipping this step.')

    template_tab = Table(names=('filter','pointing','sca'), dtype=(str,int,int))
    science_tab = Table(names=('filter','pointing','sca'), dtype=(str,int,int))

    for band in get_roman_bands():
        sci = get_science(oid,band,infodir,returntype='table')
        temp = get_templates(oid,band,infodir,n_templates=n_templates,returntype='table')

        science_tab = vstack([science_tab,sci])
        template_tab = vstack([template_tab,temp])

    scipath = os.path.join(savedir,f'{oid}_instances_science.csv')
    temppath = os.path.join(savedir,f'{oid}_instances_template_{n_templates}.csv')

    if not os.path.exists(scipath):
        science_tab.write(scipath,format='csv',overwrite=True)
    else:
        print(f'{scipath} exists. Skipping this step.')
    
    if not os.path.exists(temppath):
        template_tab.write(temppath,format='csv',overwrite=True)
    else:
        print(f'{temppath} exists. Skipping this step.')


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

    parser.add_argument(
        "--n-templates",
        type=int,
        help='Number of template images to retrieve.'
    )

    args = parser.parse_args()

    make_object_tables(args.oid,args.n_templates)
    print(f'Tables for {args.oid} retrieved and saved!')

if __name__ == '__main__':
    parse_and_run()
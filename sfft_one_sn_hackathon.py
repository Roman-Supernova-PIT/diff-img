# import nvtx

import preprocess
import sfftdiff
import postprocess
import mk_lc

sne = [ '20172782' ]
bands = [ 'R062' ]

for sn in sne:
    for band in bands:
        # with nvtx.annotate( "preprocess", color="red" ):
        preprocess.run( sn, band, n_templates=1, verbose=True, cpus_per_task=1 )

        # with nvtx.annotate( "sfftdiff", color="green" ):
        sfftdiff.run( sn, band, n_templates=1, verbose=True )

        # with nvtx.annotate( "postprocess", color="blue" ):
        postprocess.run( sn, band, n_templates=1, verbose=True, cpus_per_task=1 )

        # with nvtx.annotate( "mk_lc", color="violet" ):
        mk_lc.run( sn, band, n_templates=1, verbose=True, cpus_per_task=1 )

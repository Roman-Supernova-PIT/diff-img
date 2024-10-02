import nvtx

from phrosty.utils import set_logger

import preprocess
import sfftdiff
import postprocess
import mk_lc

sne = [ '20172782' ]
bands = [ 'R062' ]

ncpus = 1

logger = set_logger( "wrapper", "wrapper" )

for sn in sne:
    for band in bands:
        logger.info( f"*** Running preprocess for {sn} {band}" )
        with nvtx.annotate( "preprocess", color=0x0000ff ):
            preprocess.run( sn, band, n_templates=1, verbose=True, cpus_per_task=ncpus )

        logger.info( f"*** Running sfftdiff for {sn} {band}" )
        with nvtx.annotate( "sfftdiff", color="green" ):
            sfftdiff.run( sn, band, n_templates=1, verbose=True )

        logger.info( f"*** Running postprocess for {sn} {band} : {postprocess.__file__}" )
        with nvtx.annotate( "postprocess", color=0x0000ff ):
            postprocess.run( sn, band, n_templates=1, verbose=True, cpus_per_task=ncpus )

        # logger.info( f"*** Running mk_lc for {sn} {band}" )
        # with nvtx.annotate( "mk_lc", color="violet" ):
        #     mk_lc.run( sn, band, n_templates=1, verbose=True, cpus_per_task=ncpus )

        logger.info( f"*** Done with {sn} {band}" )

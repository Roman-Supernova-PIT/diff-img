# download input data
basn [YOUR_DIR]/SpaceSFFTPurePack-PostHack/input/download_input_files.sh

# edit configuration file was.yaml
[1] open [YOUR_DIR]/SpaceSFFTPurePack-PostHack/auxiliary/was.yaml
[2] find line 118:
    file_name: /hildafs/projects/phy220048p/leihu/AstroWork/RomanSNPIT/SpaceSFFTPurePack-OPT/auxiliary/Roman_WAS_obseq_11_1_23.fits
    >>> to your path
    file_name: [YOUR_DIR]/SpaceSFFTPurePack-OPT/auxiliary/Roman_WAS_obseq_11_1_23.fits

# edit trigger.py
[1] open [YOUR_DIR]/SpaceSFFTPurePack-PostHack/trigger.py
[2] find line 8
    GDIR = "/hildafs/projects/phy220048p/leihu/AstroWork/RomanSNPIT/SpaceSFFTPurePack-OPT"
    >>> to your path
    GDIR = "[YOUR_DIR]/SpaceSFFTPurePack-OPT"

# edit trigger.job
[1] open [YOUR_DIR]/SpaceSFFTPurePack-PostHack/trigger.job
[2] edit line 1-11 following your configuration
    edit line 24 and 27 to your Python env and nsys
    edit line 37-41 to use your script path 

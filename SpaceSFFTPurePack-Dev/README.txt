# Says the directory path is ***/SpaceSFFTPurePace-Dev

# Step1. PATH you need to adjust:
# [1] change PATH in ***/SpaceSFFTPurePace-Dev/auxiliary/was.yaml
# line 118:
#     file_name: /hildafs/projects/phy220048p/leihu/AstroWork/RomanSNPIT/SpaceSFFTPurePack-OPT/auxiliary/Roman_WAS_obseq_11_1_23.fits
# changed to 
#     file_name: ***/SpaceSFFTPurePace-Dev/auxiliary/Roman_WAS_obseq_11_1_23.fits
#
# [2] change PATH in ***/SpaceSFFTPurePace-Dev/trigger.py
# line 8:
#     GDIR = "/hildafs/projects/phy220048p/leihu/AstroWork/RomanSNPIT/SpaceSFFTPurePack-OPT"
# change to
#     GDIR = "***/SpaceSFFTPurePace-Dev"

# Step2. download input files
# $ cd ***/SpaceSFFTPurePace-Dev/input
# $ bash download_input_files.sh

# Step3. run the example code
# $ cd ***/SpaceSFFTPurePace-Dev
# EDIT trigger.job to match your enviornment
# $ sbatch trigger.job

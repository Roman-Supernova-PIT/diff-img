Requirment on Python Enviornment 

# roman_imsim for retreiving psf models
pip install https://codeload.github.com/matroxel/roman_imsim/zip/refs/heads/v2.0
python setup.py install

# sfft for image subtraction
pip install sfft==1.4.1

# AstroMatic softwares: SourceExtractor and SWarp for preprocessing
conda install -c conda-forge astromatic-source-extractor astromatic-swarp

# addtional package for quality check plot 
conda install -c conda-forge palettable 

"""
This script handles the methylation data compiled for the creation of AltumAge.
It performs the following tasks:
- Copies methylation data from an S3 bucket to a local directory.
- Reads methylation data and ages from pickled files.
- Creates an AnnData object with methylation data and ages.
- Writes the AnnData object to an HDF5 file.
"""

# Importing required libraries
import pandas as pd
import anndata
import numpy as np
import subprocess

# Copying methylation data from AWS S3 to local directory
subprocess.call('aws s3 cp s3://chromage-23149102/methylation_data data/methylation_data --recursive', shell=True)

# Reading methylation data and ages from pickled files
methylation_data = pd.read_pickle('data/methylation_data/methylation_data.pkl')
ages = pd.read_pickle('data/methylation_data/ages.pkl')

# Creating an AnnData object with methylation data and ages
adata_methylation_AltumAge = anndata.AnnData(
    X = np.array(methylation_data),
    obs = pd.DataFrame(ages),
    dtype = 'float64'
)

# Reading ages again (might be redundant, consider removing this line if ages are already loaded)
ages = pd.read_pickle('data/methylation_data/ages.pkl')

# Setting variable names in the AnnData object to match methylation data columns
adata_methylation_AltumAge.var_names = methylation_data.columns

# Writing the AnnData object to an HDF5 file for later use
adata_methylation_AltumAge.write_h5ad('data/methylation_data/adata_methylation.h5ad')

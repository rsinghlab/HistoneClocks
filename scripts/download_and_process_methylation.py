#this is methylation data compiled for the creation of AltumAge

import pandas as pd
import anndata
import numpy as np
import subprocess

subprocess.call('aws s3 cp s3://chromage-23149102/methylation_data data/methylation_data --recursive', shell=True)

methylation_data = pd.read_pickle('data/methylation_data/methylation_data.pkl')
ages = pd.read_pickle('data/methylation_data/ages.pkl')

adata_methylation_AltumAge = anndata.AnnData(
    X = np.array(methylation_data),
    obs = pd.DataFrame(ages),
    dtype = 'float64'
)

ages = pd.read_pickle('data/methylation_data/ages.pkl')

adata_methylation_AltumAge.var_names = methylation_data.columns

adata_methylation_AltumAge.write_h5ad('data/methylation_data/adata_methylation.h5ad')
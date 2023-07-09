import pandas as pd
import anndata
import pyBigWig as pbw
import os
import os.path as osp
import numpy as np
import subprocess
import scanpy as sc
from tqdm import tqdm
import subprocess
import glob

subprocess.call('aws s3 cp s3://chromage-23149102/geo_data data/geo_data --recursive', shell=True)

geo_metadata = pd.read_csv('metadata/GEO_metadata.csv')

files = glob.glob("data/geo_data/*/*/*/*.pval.signal.bigwig")
gsms = [file.split('/')[-2] for file in files]
histones = [file.split('/')[-3] for file in files]
files_df = pd.DataFrame(np.array([gsms, histones, files]).T, columns=['gsm','histone','file_path'])

chromosomes = ['1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '3', '4', '5', '6', '7', '8', '9', 'X']

genes = pd.read_csv('metadata/Ensembl-105-EnsDb-for-Homo-sapiens-genes.csv')
genes = genes[genes['chr'].apply(lambda x: x in chromosomes)]
genes.index = genes.gene_id

gaps = pd.read_csv('metadata/Ensembl-105-EnsDb-for-Homo-sapiens-gaps.csv')
gaps = gaps[gaps['chr'].apply(lambda x: x in chromosomes)]
gaps.index = gaps.apply(lambda x: "chr{}:{}-{}".format(x[0], x[1], x[2]), axis=1)

signals_genes = np.empty(shape=(0, genes.shape[0]), dtype=float)
signals_gaps = np.empty(shape=(0, gaps.shape[0]), dtype=float)
metadata = pd.DataFrame(columns=geo_metadata.columns)

for i in tqdm(range(files_df.shape[0])):
    
    search_column = files_df.histone[i] + ' GEO'
    metadata = pd.concat([metadata, geo_metadata[geo_metadata.loc[:, search_column] == files_df.gsm[i]]])
    
    bw = pbw.open(files_df.file_path[i])
            
    signal_sample = np.empty(shape=(0, 0), dtype=float)
    for i in range(genes.shape[0]):
        try: 
            signal = bw.stats('chr' + genes['chr'].iloc[i], genes['start'].iloc[i] - 1, genes['end'].iloc[i], type = 'mean', exact=True)[0]
        except:
            signal = None
        if signal is not None:
            signal_transformed = np.arcsinh(signal)
        else:
            signal_transformed = np.nan
        signal_sample = np.append(signal_sample, signal_transformed)
    signals_genes = np.vstack((signals_genes, signal_sample))
    
    signal_sample = np.empty(shape=(0, 0), dtype=float)
    for i in range(gaps.shape[0]):
        try:
            signal = bw.stats('chr' + gaps['chr'].iloc[i], gaps['start'].iloc[i] - 1, gaps['end'].iloc[i], type = 'mean', exact=True)[0]
        except:
            signal = None
        if signal is not None:
            signal_transformed = np.arcsinh(signal)
        else:
            signal_transformed = np.nan
        signal_sample = np.append(signal_sample, signal_transformed)
    signals_gaps = np.vstack((signals_gaps, signal_sample))
    
metadata['histone'] = files_df.histone.tolist()
                    
adata_genes = anndata.AnnData(
    X = signals_genes,
    obs = metadata,
    var = genes,
    dtype = 'float64'
)

adata_gaps = anndata.AnnData(
    X = signals_gaps,
    obs = metadata,
    var = gaps,
    dtype = 'float64'
)

adata_genes.obs.index.name = None
adata_gaps.obs.index.name = None

adata_genes.var.index.name = None
adata_gaps.var.index.name = None

sc.pp.calculate_qc_metrics(adata_genes, inplace=True)
sc.pp.calculate_qc_metrics(adata_gaps, inplace=True)

adata_genes.obs['n_nas'] = np.apply_along_axis(lambda x: np.sum(np.isnan(x)), 1, adata_genes.X)
adata_gaps.obs['n_nas'] = np.apply_along_axis(lambda x: np.sum(np.isnan(x)), 1, adata_gaps.X)

adata_genes.var['n_nas'] = np.apply_along_axis(lambda x: np.sum(np.isnan(x)), 0, adata_genes.X)
adata_gaps.var['n_nas'] = np.apply_along_axis(lambda x: np.sum(np.isnan(x)), 0, adata_gaps.X)

adata_genes.X = np.nan_to_num(adata_genes.X, 0)
adata_gaps.X = np.nan_to_num(adata_gaps.X, 0)

adata_genes._inplace_subset_obs(adata_genes.obs.n_nas < int(adata_genes.shape[1]/10))
adata_gaps._inplace_subset_obs(adata_gaps.obs.n_nas < int(adata_gaps.shape[1]/10))

adata_genes.write_h5ad('data/geo_data/adata_genes.h5ad')
adata_gaps.write_h5ad('data/geo_data/adata_gaps.h5ad')
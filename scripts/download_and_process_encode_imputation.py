import numpy as np
import pandas as pd
import subprocess
from tqdm import tqdm

subprocess.call("curl -s 'https://www.encodeproject.org/metadata/?type=Annotation&cart=%2Fcarts%2Fd04961b5-96d7-4347-977d-8716838e1636%2F' --output metadata/encode_metadata_imputation.tsv", shell=True)

subprocess.call("curl -s 'https://www.encodeproject.org/report.tsv?type=Annotation&cart=/carts/d04961b5-96d7-4347-977d-8716838e1636/' --output metadata/encode_report_imputation.tsv", shell=True)

metadata = pd.read_csv('metadata/encode_metadata_imputation.tsv', sep = '\t', index_col = 0, low_memory=False)
report = pd.read_csv('metadata/encode_report_imputation.tsv', sep = '\t', header = 1, index_col = 1, low_memory=False)
metadata = pd.merge(metadata, report, left_on='Dataset accession', right_index=True, how='inner')

def get_age(x):
    
    normal_gestational_week=40

    if 'newborn' in x:
        age = 0
    elif 'week' in x:
        x_list = x.split(' ')
        for i in range(len(x_list)):
            if 'week' in x_list[i]:
                days =  ''.join([y for y in x_list[i-1] if y.isdigit()])
                days = int(days)
                age = - (normal_gestational_week*7 - days)/365
                break
    elif 'year' in x:
        x_list = x.split(' ')
        for i in range(len(x_list)):
            if 'year' in x_list[i]:
                age = ''.join([y for y in x_list[i-1] if y.isdigit()])
                age = int(age)
                break 
    else:
        age = np.nan
                
    return age

def get_cancer(x):
    
    if 'oma' in x or 'cancer' in x or 'tumor' in x or 'tumour' in x:
        cancer = True
    else:
        cancer = False
                
    return cancer

metadata['file_accession'] = metadata.index.tolist()

filters = (metadata['Biosample type'] == 'tissue') 

metadata = metadata[filters]

metadata['age'] = metadata['Description'].apply(lambda x: get_age(x))

metadata['cancer'] = metadata['Description'].apply(lambda x: get_cancer(x))

metadata['url'] = metadata['File download URL']

metadata['histone'] = metadata['Target']

metadata['description'] = metadata['Description']

metadata['tissue'] = metadata['Biosample term name_x']

metadata['audit_error'] = metadata['Audit ERROR']

metadata['audit_warning'] = metadata['Audit WARNING']

metadata['file_size'] = metadata['Size']

metadata.index = metadata['file_accession']

metadata = metadata[metadata['age'].apply(lambda x: not np.isnan(x))]

metadata = metadata.sort_values('url')

import pandas as pd
import anndata
import pyBigWig as pbw
import os
import os.path as osp
import numpy as np
import subprocess
import scanpy as sc
import time

chromosomes = ['1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '3', '4', '5', '6', '7', '8', '9', 'X']

genes = pd.read_csv('metadata/Ensembl-105-EnsDb-for-Homo-sapiens-genes.csv')
genes = genes[genes['chr'].apply(lambda x: x in chromosomes)]
genes.index = genes.gene_id

gaps = pd.read_csv('metadata/Ensembl-105-EnsDb-for-Homo-sapiens-gaps.csv')
gaps = gaps[gaps['chr'].apply(lambda x: x in chromosomes)]
gaps.index = gaps.apply(lambda x: "chr{}:{}-{}".format(x[0], x[1], x[2]), axis=1)

hm27 = pd.read_table('metadata/HM27.hg38.manifest.tsv.gz')
hm27.index = hm27['probeID'].tolist()
hm450 = pd.read_table('metadata/HM450.hg38.manifest.tsv.gz')
hm450.index = hm450['probeID'].tolist()
hm27_450 = hm27.loc[np.intersect1d(hm27.index, hm450.index)]
hm27_450_cpgs = pd.read_pickle('data/methylation_data/27_450_cpgs.pkl')
hm27_450 = hm27.loc[hm27_450_cpgs]

#annot = pd.concat([genes,gaps])

mypath = 'data/encode_data/bigWigs'

signals_genes = np.empty(shape=(0, genes.shape[0]), dtype=float)
signals_gaps = np.empty(shape=(0, gaps.shape[0]), dtype=float)
signals_hm27_450_cpgs = np.empty(shape=(0, hm27_450.shape[0]), dtype=float)

for index in tqdm(metadata.index, desc='bigWig downloads'):
    
    file = str(index) + ".bigWig"
    file_path = osp.join(mypath, file)
    url = metadata['url'][index]
    
    while True:
        try:
            subprocess.call("curl -L -s " + url + " --output " + file_path, shell=True)
            bw = pbw.open(file_path)
            break
        except:
            time.sleep(60)
            continue
            
    signal_sample = np.empty(shape=(0, 0), dtype=float)
    for i in range(genes.shape[0]):
        try: 
            signal = bw.stats('chr' + genes['chr'].iloc[i], genes['start'].iloc[i] - 1, genes['end'].iloc[i], type = 'mean', exact=True)[0]
        except:
            signal = None
        if signal is not None:
            signal_transformed = np.arcsinh(signal)
        else:
            signal_transformed = 0
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
            signal_transformed = 0
        signal_sample = np.append(signal_sample, signal_transformed)
    signals_gaps = np.vstack((signals_gaps, signal_sample))
    
    signal_sample = np.empty(shape=(0, 0), dtype=float)
    for i in range(hm27_450.shape[0]):
        #hm27_450['CpG_beg'] is 0-indexed and hm27_450['CpG_end'] is 1-indexed
        try:
            signal = bw.stats(hm27_450['CpG_chrm'].iloc[i], hm27_450['CpG_beg'].iloc[i], hm27_450['CpG_end'].iloc[i], type = 'mean', exact=True)[0]
        except:
            signal = None
        if signal is not None:
            signal_transformed = np.arcsinh(signal)
        else:
            signal_transformed = np.nan
        signal_sample = np.append(signal_sample, signal_transformed)
    signals_hm27_450_cpgs = np.vstack((signals_hm27_450_cpgs, signal_sample))
        
    subprocess.call("rm " + file_path, shell=True)
        
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

adata_hm27_450_cpgs = anndata.AnnData(
    X = signals_hm27_450_cpgs,
    obs = metadata,
    var = hm27_450,
    dtype = 'float64'
)

adata_genes.obs.index.name = None
adata_gaps.obs.index.name = None
adata_hm27_450_cpgs.obs.index.name = None

adata_genes.var.index.name = None
adata_gaps.var.index.name = None
adata_hm27_450_cpgs.var.index.name = None

sc.pp.calculate_qc_metrics(adata_genes, inplace=True)
sc.pp.calculate_qc_metrics(adata_gaps, inplace=True)
sc.pp.calculate_qc_metrics(adata_hm27_450_cpgs, inplace=True)

adata_genes.obs['n_nas'] = np.apply_along_axis(lambda x: np.sum(np.isnan(x)), 1, adata_genes.X)
adata_gaps.obs['n_nas'] = np.apply_along_axis(lambda x: np.sum(np.isnan(x)), 1, adata_gaps.X)
adata_hm27_450_cpgs.obs['n_nas'] = np.apply_along_axis(lambda x: np.sum(np.isnan(x)), 1, adata_hm27_450_cpgs.X)

adata_genes.var['n_nas'] = np.apply_along_axis(lambda x: np.sum(np.isnan(x)), 0, adata_genes.X)
adata_gaps.var['n_nas'] = np.apply_along_axis(lambda x: np.sum(np.isnan(x)), 0, adata_gaps.X)
adata_hm27_450_cpgs.var['n_nas'] = np.apply_along_axis(lambda x: np.sum(np.isnan(x)), 0, adata_hm27_450_cpgs.X)

adata_genes.X = np.nan_to_num(adata_genes.X, 0)
adata_gaps.X = np.nan_to_num(adata_gaps.X, 0)
adata_hm27_450_cpgs.X = np.nan_to_num(adata_hm27_450_cpgs.X, 0)

adata_genes._inplace_subset_obs(adata_genes.obs.n_nas < int(adata_genes.shape[1]/10))
adata_gaps._inplace_subset_obs(adata_gaps.obs.n_nas < int(adata_gaps.shape[1]/10))
adata_hm27_450_cpgs._inplace_subset_obs(adata_hm27_450_cpgs.obs.n_nas < int(adata_hm27_450_cpgs.shape[1]/10))

adata_genes.write_h5ad('data/encode_data/adata_genes_imputation.h5ad')
adata_gaps.write_h5ad('data/encode_data/adata_gaps_imputation.h5ad')
adata_hm27_450_cpgs.write_h5ad('data/encode_data/adata_hm27_450_cpgs_imputation.h5ad')

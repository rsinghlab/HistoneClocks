import numpy as np
import pandas as pd
import subprocess
from tqdm import tqdm

subprocess.call("curl -s 'https://www.encodeproject.org/metadata/?type=Experiment&cart=%2Fcarts%2F7234d35d-e231-47b1-9f1b-d801b9e4c7df%2F' --output metadata/encode_metadata.tsv", shell=True)

subprocess.call("curl -s 'https://www.encodeproject.org/report.tsv?type=Experiment&cart=/carts/7234d35d-e231-47b1-9f1b-d801b9e4c7df/' --output metadata/encode_report.tsv", shell=True)

metadata = pd.read_csv('metadata/encode_metadata.tsv', sep = '\t', index_col = 0, low_memory=False)
report = pd.read_csv('metadata/encode_report.tsv', sep = '\t', header = 1, index_col = 1, low_memory=False)
metadata = pd.merge(metadata, report, left_on='Experiment accession', right_index=True, how='inner')

def get_age(x):
    normal_gestational_week = 40
    x[0] = str(x[0])
    x[1] = str(x[1])
    if x[0] == 'nan' and x[1] == 'nan':
        age = np.nan
    elif 'adult' in x[0] or 'child' in x[0]:
        age = float(x[1].split(' ')[0])
    elif 'embryonic' in x[0]:
        days = float(x[1].split(' ')[0])
        age = - (normal_gestational_week*7 - days)/365
    else:
        age = np.nan
    return age

def get_sex(x):
    x = str(x)
    if x == 'nan':
        sex = np.nan
    elif 'female' in x:
        sex = 'F'
    elif 'male' in x:
        sex = 'M'
    else:
        sex = np.nan
    return sex

def get_cancer(x):
    x = str(x)
    if x == 'nan':
        cancer = np.nan
    elif "squamous cell carcinoma" in x or "basal cell carcinoma" in x:
        cancer = True
    else:
        cancer = False
    return cancer

def get_disease(x):
    x = str(x).lower()
    if x == 'nan':
        disease = np.nan
    elif "disease" in x:
        disease = True
    elif "impairment" in x:
        disease = True
    elif "carcinoma" in x:
        disease = True
    else:
        disease = False
    return disease

def get_date_float(x):
    year, month, day = x.split('-')
    date = int(year) + (30.5*(int(month))+int(day))/365
    return date

def get_library_fragmentation_method(x):
    if x == 'none, sonication (generic)':
        library_fragmentation_method = np.nan
    else:
        library_fragmentation_method = x
    return library_fragmentation_method

#this function gets some of the values that are only given for fastq files and expands to all of the columns so that it can be retrieved later
def expand_to_whole_experiment(df, col):
    summary = df.groupby(['Experiment accession', col]).count().reset_index().astype(str)[['Experiment accession', col]].groupby('Experiment accession').agg({col: '-'.join})
    df = pd.merge(df, summary, left_on='Experiment accession', right_on='Experiment accession',how='outer')
    return df

metadata['file_accession'] = metadata.index.tolist()

metadata = expand_to_whole_experiment(metadata, 'Platform')
metadata['platform'] = metadata['Platform_y']

metadata = expand_to_whole_experiment(metadata, 'Run type')
metadata['run_end'] = metadata['Run type_y']

metadata = expand_to_whole_experiment(metadata, 'Mapped read length')
metadata['mapped_read_length'] = metadata['Mapped read length_y']

filters = (metadata['File type'] == 'bigWig') & \
    (metadata['File assembly'] == 'GRCh38') & \
    (metadata['Output type'] == 'signal p-value') & \
    (metadata['Biosample treatment'].isna()) & \
    (metadata['File analysis status'] == 'released') #REMEMBER IT IS ALREADY -LOG10(P-VALUE)

metadata = metadata[filters]

metadata['age'] = np.apply_along_axis(get_age, 1, metadata[['Life stage','Biosample age']])

metadata['sex'] = metadata['Biosample summary'].apply(lambda x: get_sex(x))

metadata['cancer'] = metadata['Biosample summary'].apply(lambda x: get_cancer(x))

metadata['disease'] = metadata['Biosample summary'].apply(lambda x: get_disease(x))

metadata['alzheimers'] = metadata['Biosample summary'].apply(lambda x: "Alzheimer's disease" in str(x))

metadata['cognitive_impairment'] = metadata['Biosample summary'].apply(lambda x: "Cognitive impairment" in str(x))

metadata['squamous_cell_carcinoma'] = metadata['Biosample summary'].apply(lambda x: "squamous cell carcinoma" in str(x))

metadata['basal_cell_carcinoma'] = metadata['Biosample summary'].apply(lambda x: "basal cell carcinoma" in str(x))

metadata['coronary_artery_disease'] = metadata['Biosample summary'].apply(lambda x: "nonobstructive coronary artery disease" in str(x))

metadata['library_fragmentation_method'] = metadata['Library fragmentation method'].apply(lambda x: get_library_fragmentation_method(x))

metadata['url'] = metadata['File download URL']

metadata['histone'] = metadata['Target of assay']

metadata['project'] = metadata['Project_y']

metadata['biological_replicates'] = metadata['Biological replicate(s)']

metadata['technical_replicates'] = metadata['Technical replicate(s)']

metadata['experiment_accession'] = metadata['Experiment accession']

metadata['cellular_component'] = metadata['Cellular component']

metadata['antibody'] = metadata['Linked antibody']

metadata['biosample_accession'] = metadata['Biosample accession']

metadata['lab'] = metadata['Lab_y']

metadata['description'] = metadata['Description']

metadata['tissue'] = metadata['Biosample term name_y']

metadata['audit_error'] = metadata['Audit ERROR']

metadata['audit_warning'] = metadata['Audit WARNING']

metadata['file_size'] = metadata['Size']

metadata['library_size_range'] = metadata['Library size range']

metadata['assay'] = metadata['Assay']

metadata['biosample_type'] = metadata['Biosample type']

metadata.index = metadata['file_accession']

obs_columns = [
    'age',
    'sex',
    'disease',
    'alzheimers',
    'cognitive_impairment',
    'squamous_cell_carcinoma',
    'basal_cell_carcinoma',
    'coronary_artery_disease',
    'cancer',
    'histone',
    'project',
    'library_fragmentation_method',
    'library_size_range',
    'biological_replicates',
    'technical_replicates',
    'experiment_accession',
    'cellular_component',
    'antibody',
    'biosample_accession',
    'lab',
    'description',
    'tissue',
    'audit_error',
    'audit_warning',
    'file_size',
    'platform',
    'run_end',
    'mapped_read_length',
    'url',
    'file_accession',
    'assay',
    'biosample_type'
]

metadata = metadata[obs_columns].copy()

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

adata_genes.write_h5ad('data/encode_data/adata_genes.h5ad')
adata_gaps.write_h5ad('data/encode_data/adata_gaps.h5ad')
adata_hm27_450_cpgs.write_h5ad('data/encode_data/adata_hm27_450_cpgs.h5ad')

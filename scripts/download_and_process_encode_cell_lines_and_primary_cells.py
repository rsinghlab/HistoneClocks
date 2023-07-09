import numpy as np
import pandas as pd
import subprocess
from tqdm import tqdm

subprocess.call("curl -s 'https://www.encodeproject.org/metadata/?type=Experiment&cart=%2Fcarts%2F0075fced-b6c4-4ee2-a44a-a15756a1518b%2F' --output metadata/encode_metadata_cell_lines_and_primary_cells.tsv", shell=True)

subprocess.call("curl -s 'https://www.encodeproject.org/report.tsv?type=Experiment&cart=/carts/0075fced-b6c4-4ee2-a44a-a15756a1518b/' --output metadata/encode_report_cell_lines_and_primary_cells.tsv", shell=True)

import numpy as np
import pandas as pd
import subprocess
from tqdm import tqdm

metadata = pd.read_csv('metadata/encode_metadata_cell_lines_and_primary_cells.tsv', sep = '\t', index_col = 0, low_memory=False)
report = pd.read_csv('metadata/encode_report_cell_lines_and_primary_cells.tsv', sep = '\t', header = 1, index_col = 1, low_memory=False)
metadata = pd.merge(metadata, report, left_on='Experiment accession', right_index=True, how='inner')

filters = (metadata['File type'] == 'bigWig') & \
    (metadata['File assembly'] == 'GRCh38') & \
    (metadata['Output type'] == 'signal p-value') & \
    (metadata['File analysis status'] == 'released') #REMEMBER IT IS ALREADY -LOG10(P-VALUE)

metadata = metadata[filters]

metadata['url'] = metadata['File download URL']

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

mypath = 'data/encode_data/bigWigs'

signals_genes = np.empty(shape=(0, genes.shape[0]), dtype=float)

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
                
    subprocess.call("rm " + file_path, shell=True)
        
adata_genes = anndata.AnnData(
    X = signals_genes,
    obs = metadata,
    var = genes,
    dtype = 'float64'
)

adata_genes.obs.index.name = None

adata_genes.var.index.name = None

sc.pp.calculate_qc_metrics(adata_genes, inplace=True)

adata_genes.obs['n_nas'] = np.apply_along_axis(lambda x: np.sum(np.isnan(x)), 1, adata_genes.X)

adata_genes.var['n_nas'] = np.apply_along_axis(lambda x: np.sum(np.isnan(x)), 0, adata_genes.X)

adata_genes.X = np.nan_to_num(adata_genes.X, 0)

adata_genes._inplace_subset_obs(adata_genes.obs.n_nas < int(adata_genes.shape[1]/10))

adata_genes.write_h5ad('data/encode_data/adata_genes_cell_lines_and_primary_cells.h5ad')

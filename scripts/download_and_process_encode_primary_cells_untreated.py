"""
This script performs various operations to download and process biological data from ENCODE project.

1. Imports: Necessary libraries are imported, including pandas, NumPy, pyBigWig, Scanpy, etc.
2. Downloading: Metadata and report files are downloaded from the ENCODE website using subprocess calls to curl.
3. Reading Files: The metadata and report files are read and merged into a DataFrame.
4. Data Extraction Functions: Functions are defined to extract and transform specific information such as age, sex, cancer status, disease status, and other experimental attributes.
5. Metadata Expansion: Certain metadata columns are expanded and new columns are added using the previously defined functions.
6. Data Filtering: Filters are applied to the metadata to obtain specific data such as the GRCh38 assembly and released status of bigWig files.
7. Adding New Columns: Additional data columns are derived and added to the metadata, including URL, biosample details, disease markers, etc.
8. Sorting and Finalizing: The metadata is sorted, chromosomes are defined, and additional metadata files like genes are read.
9. Signal Processing: A loop iterates through metadata indices to download bigWig files, extract and transform signal data for genes. This includes repeated attempts to download in case of failure.
10. Removal of Temporary Files: Downloaded bigWig files are removed after processing.
11. AnnData Objects: Placeholder to create AnnData objects for the extracted signals.

The result of this script is a processed and filtered set of data ready for subsequent biological analysis.
"""
# Importing necessary libraries
import numpy as np
import pandas as pd
import subprocess
from tqdm import tqdm
import anndata
import pyBigWig as pbw
import os
import os.path as osp
import scanpy as sc
import time

# Fetching metadata related to primary cells from the encode project
subprocess.call("curl -s 'https://www.encodeproject.org/metadata/?type=Experiment&cart=%2Fcarts%2F30169f7a-2f41-4042-a438-e447476b2dd4%2F' --output metadata/encode_metadata_primary_cells_untreated.tsv", shell=True)
# Fetching report related to primary cells from the encode project
subprocess.call("curl -s 'https://www.encodeproject.org/report.tsv?type=Experiment&cart=/carts/30169f7a-2f41-4042-a438-e447476b2dd4/' --output metadata/encode_report_primary_cells_untreated.tsv", shell=True)

# Reading the metadata and report as pandas DataFrames
metadata = pd.read_csv('metadata/encode_metadata_primary_cells_untreated.tsv', sep='\t', index_col=0, low_memory=False)
report = pd.read_csv('metadata/encode_report_primary_cells_untreated.tsv', sep='\t', header=1, index_col=1, low_memory=False)
metadata = pd.merge(metadata, report, left_on='Experiment accession', right_index=True, how='inner')

# Function to compute the age based on given parameters
def get_age(x):
    normal_gestational_week = 40
    x[0] = str(x[0])
    x[1] = str(x[1])
    if 'newborn' in x[0]:
        age = 0
    elif x[1] == 'nan' or ',' in x[1]:
        age = np.nan
    elif 'years' in x[1]:
        age = float(x[1].split(' ')[0])
    elif '2-4 days' in x[1]:
        age = 3/365
    elif 'weeks' in x[1]:
        weeks = float(x[1].split(' ')[0])
        age = - (normal_gestational_week - weeks)*7/365
    elif 'days' in x[1]:
        days = float(x[1].split(' ')[0])
        age = - (normal_gestational_week*7 - days)/365
    return float(age)

# Function to determine the sex
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

# Function to determine if cancer is present
def get_cancer(x):
    x = str(x)
    if x == 'nan':
        cancer = np.nan
    elif "squamous cell carcinoma" in x or "basal cell carcinoma" in x:
        cancer = True
    else:
        cancer = False
    return cancer

# Function to determine if any disease or impairment is present
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

# Function to convert a date string into a float value
def get_date_float(x):
    year, month, day = x.split('-')
    date = int(year) + (30.5*(int(month))+int(day))/365
    return date

# Function to determine the library fragmentation method
def get_library_fragmentation_method(x):
    if x == 'none, sonication (generic)':
        library_fragmentation_method = np.nan
    else:
        library_fragmentation_method = x
    return library_fragmentation_method

# Function to expand certain values across columns for later retrieval
def expand_to_whole_experiment(df, col):
    summary = df.groupby(['Experiment accession', col]).count().reset_index().astype(str)[['Experiment accession', col]].groupby('Experiment accession').agg({col: '-'.join})
    df = pd.merge(df, summary, left_on='Experiment accession', right_on='Experiment accession',how='outer')
    return df

# Set file_accession as the index of metadata DataFrame
metadata['file_accession'] = metadata.index.tolist()

# Expanding the 'Platform' to the whole experiment
metadata = expand_to_whole_experiment(metadata, 'Platform')
metadata['platform'] = metadata['Platform_y']

# Expanding the 'Run type' to the whole experiment
metadata = expand_to_whole_experiment(metadata, 'Run type')
metadata['run_end'] = metadata['Run type_y']

# Expanding the 'Mapped read length' to the whole experiment
metadata = expand_to_whole_experiment(metadata, 'Mapped read length')
metadata['mapped_read_length'] = metadata['Mapped read length_y']

# Filtering the metadata for specific criteria
filters = (metadata['File type'] == 'bigWig') & \
          (metadata['File assembly'] == 'GRCh38') & \
          (metadata['Output type'] == 'signal p-value') & \
          (metadata['Biosample treatment'].isna()) & \
          (metadata['File analysis status'] == 'released') # REMEMBER IT IS ALREADY -LOG10(P-VALUE)
metadata = metadata[filters]

# Applying functions to create new columns with specific information
metadata['age'] = np.apply_along_axis(get_age, 1, metadata[['Life stage', 'Biosample age']])
metadata['sex'] = metadata['Biosample summary'].apply(lambda x: get_sex(x))
metadata['cancer'] = metadata['Biosample summary'].apply(lambda x: get_cancer(x))
metadata['disease'] = metadata['Biosample summary'].apply(lambda x: get_disease(x))
metadata['alzheimers'] = metadata['Biosample summary'].apply(lambda x: "Alzheimer's disease" in str(x))
metadata['cognitive_impairment'] = metadata['Biosample summary'].apply(lambda x: "Cognitive impairment" in str(x))
metadata['squamous_cell_carcinoma'] = metadata['Biosample summary'].apply(lambda x: "squamous cell carcinoma" in str(x))
metadata['basal_cell_carcinoma'] = metadata['Biosample summary'].apply(lambda x: "basal cell carcinoma" in str(x))
metadata['coronary_artery_disease'] = metadata['Biosample summary'].apply(lambda x: "nonobstructive coronary artery disease" in str(x))
metadata['library_fragmentation_method'] = metadata['Library fragmentation method'].apply(lambda x: get_library_fragmentation_method(x))

# Mapping specific columns to new names
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

# Re-indexing the DataFrame
metadata.index = metadata['file_accession']

# Selecting only the required columns
obs_columns = [
    'age', 'sex', 'disease', 'alzheimers', 'cognitive_impairment', 'squamous_cell_carcinoma',
    'basal_cell_carcinoma', 'coronary_artery_disease', 'cancer', 'histone', 'project',
    'library_fragmentation_method', 'library_size_range', 'biological_replicates', 'technical_replicates',
    'experiment_accession', 'cellular_component', 'antibody', 'biosample_accession', 'lab', 'description',
    'tissue', 'audit_error', 'audit_warning', 'file_size', 'platform', 'run_end', 'mapped_read_length',
    'url', 'file_accession', 'assay', 'biosample_type'
]
metadata = metadata[obs_columns].copy()

# Filtering out rows with NaN in 'age' column
metadata = metadata[metadata['age'].apply(lambda x: not np.isnan(x))]

# Sorting the DataFrame by 'url'
metadata = metadata.sort_values('url')

# List of chromosomes
chromosomes = ['1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '3', '4', '5', '6', '7', '8', '9', 'X']

# Reading the genes from a CSV file and filtering by chromosomes
genes = pd.read_csv('metadata/Ensembl-105-EnsDb-for-Homo-sapiens-genes.csv')
genes = genes[genes['chr'].apply(lambda x: x in chromosomes)]
genes.index = genes.gene_id

# Path to data
mypath = 'data/encode_data/bigWigs'

# Initializing an empty array for signals_genes
signals_genes = np.empty(shape=(0, genes.shape[0]), dtype=float)

# Looping through the metadata to download bigWig files
for index in tqdm(metadata.index, desc='bigWig downloads'):

    # Creating file name and file path for the bigWig file
    file = str(index) + ".bigWig"
    file_path = osp.join(mypath, file)
    url = metadata['url'][index]

    # Downloading the bigWig file, retrying every 60 seconds if an exception occurs
    while True:
        try:
            subprocess.call("curl -L -s " + url + " --output " + file_path, shell=True)
            bw = pbw.open(file_path)
            break
        except:
            time.sleep(60)
            continue

    # Initializing an empty array for signal_sample
    signal_sample = np.empty(shape=(0, 0), dtype=float)

    # Looping through genes to extract signal statistics
    for i in range(genes.shape[0]):
        try:
            signal = bw.stats('chr' + genes['chr'].iloc[i], genes['start'].iloc[i] - 1, genes['end'].iloc[i], type='mean', exact=True)[0]
        except:
            signal = None
        if signal is not None:
            signal_transformed = np.arcsinh(signal) # Transforming the signal if not None
        else:
            signal_transformed = 0
        signal_sample = np.append(signal_sample, signal_transformed)

    # Stacking the signal_sample into the signals_genes array
    signals_genes = np.vstack((signals_genes, signal_sample))

    # Removing the downloaded bigWig file
    subprocess.call("rm " + file_path, shell=True)

# Creating an AnnData object to hold the gene signals and metadata
adata_genes = anndata.AnnData(
    X=signals_genes,
    obs=metadata,
    var=genes,
    dtype='float64'
)

# Removing the index names
adata_genes.obs.index.name = None
adata_genes.var.index.name = None

# Calculating QC metrics
sc.pp.calculate_qc_metrics(adata_genes, inplace=True)

# Calculating the number of NaNs in the observations and variables
adata_genes.obs['n_nas'] = np.apply_along_axis(lambda x: np.sum(np.isnan(x)), 1, adata_genes.X)
adata_genes.var['n_nas'] = np.apply_along_axis(lambda x: np.sum(np.isnan(x)), 0, adata_genes.X)

# Replacing NaN values with 0
adata_genes.X = np.nan_to_num(adata_genes.X, 0)

# Subsetting the observations based on a threshold for missing values
adata_genes._inplace_subset_obs(adata_genes.obs.n_nas < int(adata_genes.shape[1] / 10))

# Writing the AnnData object to an H5AD file
adata_genes.write_h5ad('data/encode_data/adata_genes_primary_cells_untreated.h5ad')

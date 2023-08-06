# ---------------------------------------------
# Description:
# This script is responsible for installing and loading necessary packages
# for querying and downloading Ensembl genome annotation files, particularly
# for "Homo sapiens". The script also extracts the genes and gaps data,
# processes them, and exports the data into CSV files.
# ---------------------------------------------

# Check if 'BiocManager' is installed; if not, install it
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager", repos = "http://cran.us.r-project.org")

# Load BiocManager and install the required packages
BiocManager::install()
BiocManager::install(c("AnnotationHub", "ensembldb", "Repitools"))

# Load required libraries
library(AnnotationHub)
library(Repitools)

# Create AnnotationHub object
ah <- AnnotationHub()

# Querying Ensembl genome annotation files
print("Querying Ensembl genome annotation files...")
print(query(ah, pattern = c("Ensembl", "Homo sapiens", "EnsDb")))

# Downloading Ensembl genome annotation files
print("Downloading Ensembl genome annotation files...")
edb <- ah[["AH98047"]]
print(edb)
genes = genes(edb)
gaps = gaps(genes(edb))

# Convert genes and gaps to data frame
genes = annoGR2DF(genes)
gaps = annoGR2DF(gaps)

# Apply as.character to all elements in genes and gaps
genes <- apply(genes, 2, as.character)
gaps <- apply(gaps, 2, as.character)

# Exporting GRanges files as csv
print("Exporting GRanges files as csv...")
write.csv(genes, file = "metadata/Ensembl-105-EnsDb-for-Homo-sapiens-genes.csv", row.names=FALSE)
write.csv(gaps, file = "metadata/Ensembl-105-EnsDb-for-Homo-sapiens-gaps.csv", row.names=FALSE)

print("Finished!")
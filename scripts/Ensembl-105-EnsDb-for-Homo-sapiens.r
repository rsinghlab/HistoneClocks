if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager", repos = "http://cran.us.r-project.org")
BiocManager::install()

BiocManager::install(c("AnnotationHub", "ensembldb", "Repitools"))

library(AnnotationHub)
library(Repitools)

ah <- AnnotationHub()

print("Querying Ensembl genome annotation files...")
print(query(ah, pattern = c("Ensembl", "Homo sapiens", "EnsDb")))

print("Downloading Ensembl genome annotation files...")
edb <- ah[["AH98047"]]
print(edb)
genes = genes(edb)
gaps = gaps(genes(edb))

genes = annoGR2DF(genes)
gaps = annoGR2DF(gaps)

genes <- apply(genes,2,as.character)
gaps <- apply(gaps,2,as.character)

print("Exporting GRanges files as csv...")

write.csv(genes, file = "metadata/Ensembl-105-EnsDb-for-Homo-sapiens-genes.csv", row.names=FALSE)
write.csv(gaps, file = "metadata/Ensembl-105-EnsDb-for-Homo-sapiens-gaps.csv", row.names=FALSE)

print("Finished!")
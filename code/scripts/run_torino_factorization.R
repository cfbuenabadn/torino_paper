# library(tidyverse)
# library(ebpmf.alpha)
.libPaths(c('/project/yangili1/cfbuenabadn/R/x86_64-pc-linux-gnu-library/4.1',
            '/software/R-4.1.0-el8-x86_64/lib64/R/library')) 

source("/project2/yangili1/cfbuenabadn/torino/scripts/torino_factorization.R")

set.seed(0)

args = commandArgs(trailingOnly=TRUE)
input_matrix = args[1]
min_counts = as.integer(args[2])
species = args[3]
out_file = args[4]
gene_name <- input_matrix %>% str_split_1('/') %>% tail(n=1) %>% str_split_1('\\.') %>% head(n=1)

counts <- input_matrix %>% read_csv() %>% column_to_rownames(var = "Sample_ID") 
row.names(counts) <- counts %>% row.names() %>% sub("\\.[^.]*$", "", .)

coords_all <- colnames(counts)

nbases_total <- dim(counts)[2] 

if ((nbases_total > 15000) & (nbases_total < 60000)){
    print('warning: matrix is too big; merging groups of 3 base pairs to reduce size.')
    n = 3
} else if ((nbases_total > 60000) & (nbases_total < 1000000)){
    print('warning: matrix is too big; merging groups of 9 base pairs to reduce size.')
    n = 9
} else if (nbases_total > 1000000) {
    print('warning: matrix is too big; merging groups of 9 base pairs to reduce size.')
    n = 30
} else {
    print('Matrix does not need grouping.')
    n = 1
}

if (species == 'Cattle'){
    select_samples <- readLines("config/Cattle.filtered_samples_strict.txt")
    counts <- counts[select_samples,]
    print('Filtered cattle samples')
    print(dim(counts))
} else if (species == 'Sheep'){
    metadata <- read_tsv('config/sheepGTEx_metadata.txt')
    counts <- counts[metadata %>% filter(LibraryLayout=='PAIRED', Tissue != 'Ovary') %>% rownames(),]
}

counts <- prepare_counts(counts, min_counts, n)

torino_K8 <- torino_factorization(counts, K=8, seed=1)
torino_K10 <- torino_factorization(counts, K=10, seed=1)
#torino_K20 <- torino_factorization(counts, K=20, seed=1)


saveRDS(list(gene=gene_name,
             torino_K8 = torino_K8,
             torino_K10 = torino_K10,
             #torino_K20 = torino_K20,
             coords_all = coords_all
            ),
        file=out_file
       )

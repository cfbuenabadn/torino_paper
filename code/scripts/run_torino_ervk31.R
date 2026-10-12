.libPaths(c('/project/yangili1/cfbuenabadn/R/x86_64-pc-linux-gnu-library/4.1/',
            '/software/R-4.1.0-el8-x86_64/lib64/R/library')) 


library(tidyverse)
library(torino)

# Sometimes I have to run this on jupyter to avoid a memory error
# dir.create(tempdir()) 

Sys.setenv("VROOM_CONNECTION_SIZE" = 131072 * 30)   # 

# Loading matrix of data
input_matrix <- '/project2/yangili1/cfbuenabadn/torino_paper/code/resources/count_matrices_ervk31/Human/ERVK3-1.csv.gz'
counts <- input_matrix %>% read_csv() %>% column_to_rownames(var = "Sample_ID") 

min_counts <- 1000


row.names(counts) <- counts %>% row.names() %>% sub("\\.[^.]*$", "", .)

coords_all <- colnames(counts)

gene_name <- 'ERVK3-1'

counts <- prepare_counts(counts, min_counts, 1)


torino_K5 <- torino_factorization(counts, K=5, seed=1)
torino_K8 <- torino_factorization(counts, K=8, seed=1)
torino_K10 <- torino_factorization(counts, K=10, seed=1)

out_file <- '/project2/yangili1/cfbuenabadn/torino_paper/code/results/ervk31/ervk31.rds'

saveRDS(list(gene=gene_name,
            torino_K5 = torino_K5,
             torino_K8 = torino_K8,
             torino_K10 = torino_K10,
             coords_all = coords_all
            ),
        file=out_file
       )



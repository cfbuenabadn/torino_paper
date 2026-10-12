#!/usr/bin/env Rscript

.libPaths(c('/project/yangili1/cfbuenabadn/R/x86_64-pc-linux-gnu-library/4.1',
            '/software/R-4.1.0-el8-x86_64/lib64/R/library')) 

source("/project2/yangili1/cfbuenabadn/torino/scripts/torino_factorization.R")

set.seed(0)

# args = commandArgs(trailingOnly=TRUE)

# Parse input arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) {
  stop("Usage: Rscript script.R <species> <filter_level>")
}

species = args[1]
filter_level = args[2]

if (filter_level == '_unfiltered'){
    filter_level = ''
}

# Load required libraries
suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
})



# Load sample list
if (species == 'Cattle') {
    samples_path <- "config/Cattle.filtered_samples_strict.txt"
    samples <- readLines(samples_path)
} else if (species %in% c('Mouse', 'Macaque', 'Rat', 'Rabbit', 'Chicken', 'Opossum')){
    samples <- read_tsv('config/cardoso_samples.tsv') %>% filter(Species == species) %>% pull(sampleID)
} else if (species == 'Human') {
    samples <- read_tsv('config/gtex_samples.tsv') %>% pull(sample_id)
}

# Load BED table and assign column names
bed_table_path <- paste0('resources/annotation/', species, '/bed/genes.protein_coding.bed.gz') #"resources/annotation/bed/Cattle.bed.gz"
bed_table <- fread(cmd = paste("zcat", bed_table_path))#, header = FALSE)
colnames(bed_table) <- c("#chrom", "start", "end", "gene_id", "gene_name", "strand")

# Update gene_name if it is "ensembl"
bed_table <- bed_table %>%
  mutate(gene_name = ifelse(gene_name == "ensembl", gene_id, gene_name))

# Generate the factorization path
factor_path <- sprintf("results/torino/factorization%s/%s/", filter_level, species)
gene_files <- list.files(factor_path, full.names = FALSE)
genes <- sub("\\.rds$", "", gene_files)

# Prepare output file path and directory
out_dir <- sprintf("results/torino/tables/%s%s/", species, filter_level)
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

out_file <- file.path(out_dir, "EL.K10.bed.gz")
# out_file <- gzfile(file.path(out_dir, "EL.K10.bed.gz"), "w")

header_written <- FALSE

# log_file 'logs/collect_EL/{species}{filter_level}/EL.log'

# dir.create("logs/collect_EL", showWarnings = FALSE)
# log_file <- sprintf("logs/collect_EL/%s%s.log", species, filter_level)

# Loop through genes
for (gene in genes) {
  rds_path <- file.path(factor_path, paste0(gene, ".rds"))
  rds <- readRDS(rds_path)
  
  if (is.null(rds$torino_K10)) next
  
  EL <- rds$torino_K10$EL
  gene_samples <- rds$torino_K10$samples
  
  if (is.null(EL) || is.null(gene_samples)) next
  
  # Add column and row names
  colnames(EL) <- paste0(gene, ".factor_", seq_len(ncol(EL)))
  rownames(EL) <- gene_samples
  
  # Subset EL to match samples, pad with NAs where missing
  full_matrix <- matrix(NA, nrow = length(samples), ncol = ncol(EL),
                        dimnames = list(samples, colnames(EL)))
  matched_rows <- intersect(samples, gene_samples)
  full_matrix[matched_rows, ] <- EL[matched_rows, , drop = FALSE]

  # Find gene annotation in BED table
  gene_info <- bed_table %>% filter(gene_name == gene)
  if (nrow(gene_info) == 0) next
  
  # Create output table
  bed_rows <- do.call(rbind, lapply(seq_len(ncol(full_matrix)), function(i) {
    data.frame(
      "#chrom" = gene_info$`#chrom`,
      start = gene_info$start,
      end = gene_info$end,
      factor_id = colnames(full_matrix)[i],
      gene_name = gene,
      strand = gene_info$strand,
      t(full_matrix[, i])
    )
  }))
    
 

  print(gene)
  # Set column names
  if (!header_written) {
    colnames(bed_rows) <- c("#Chr", "start", "end", "pid", "gid", "strand", samples)
    fwrite(bed_rows, out_file, sep = "\t", quote = FALSE, row.names = FALSE, col.names = TRUE, append = FALSE)
    header_written <- TRUE
#     cat(gene, "\n", file = log_file, append = FALSE)
  } else {
    fwrite(bed_rows, out_file, sep = "\t", quote = FALSE, row.names = FALSE, col.names = FALSE, append = TRUE)
#     cat(gene, "\n", file = log_file, append = TRUE)
  }
}

# close(out_file)

library(dplyr)
library(tidyr)
library(stringr)
library(edgeR)
library(tidyverse)

args <- commandArgs(trailingOnly=TRUE)

# FileIn <- args[1]
# FileOut <- args[2]
# Pass <- args[3]


counts_ad <- read.table(args[1], header=TRUE)
metadata  <- read.table(args[2], sep='\t', header=TRUE)

rownames(metadata) <- metadata %>% pull(specimenID)

names(counts_ad) <- names(counts_ad) %>%
  sub("^X", "", .) %>%
  sub("\\.final$", "", .) %>%
  gsub("\\.", "-", .)
rownames(counts_ad) <- counts_ad %>% pull(gene_ID)
counts_ad <- counts_ad[, colnames(counts_ad)[2:length(colnames(counts_ad))]]

selected_samples <- metadata[colnames(counts_ad),] %>%
  filter(condition %in% c('AD', 'no condition')) %>%
  select(c(msex, race, apoe_genotype, sequencingBatch, libraryBatch, condition)) %>%
  drop_na() %>%
  rownames()

dge <- DGEList(counts_ad[, selected_samples])
keep <- rowSums(cpm(dge) > 1) >= 3
dge <- dge[keep, , keep.lib.sizes=FALSE]

design <- model.matrix(
  ~ msex + race + apoe_genotype + sequencingBatch + condition,
  data = metadata[selected_samples,]
)

dge <- calcNormFactors(dge)
dge <- estimateDisp(dge, design)
fit <- glmFit(dge, design)
lrt <- glmLRT(fit, coef = "conditionno condition")

lrt$table['FDR'] <- lrt$table %>% pull(PValue) %>% p.adjust(., method = 'BH')
lrt$table['Gene'] <- rownames(lrt$table)

lrt$table[c('Gene', 'logFC', 'logCPM', 'LR', 'PValue', 'FDR')] %>%
  write_tsv(., args[3])
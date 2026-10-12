.libPaths(c('/project/yangili1/cfbuenabadn/R/x86_64-pc-linux-gnu-library/4.1',
            '/software/R-4.1.0-el8-x86_64/lib64/R/library')) 

library(bedr)
library(dplyr)
library(tidyverse)
library(qvalue)

args <- commandArgs(trailingOnly=TRUE)

PairwiseInput <- args[1]
Pi1Output <- args[2]

pi1_table <- read_tsv(PairwiseInput)
unique_combinations <- unique(pi1_table %>% select(PC1, PC2))

# # Initialize a vector to store the results
results <- vector("list", length = nrow(unique_combinations))


library(data.table)


# Read file
annot <- fread("/project2/yangili1/cfbuenabadn/torino_paper/code/results/torino/Human/isoforms/annotated.isoforms.tab.gz")

# Create gene column
annot[, gene := tstrsplit(transcript, "\\.", keep = 1)]

# Compute efficiency_factor
annot[, efficiency_factor := {

  appris_ref <- appris_ref
  appr_len <- as.integer(appris_transcript_length)

  # Detect RI:
  has_ri <- grepl("RI:", appris_ref)

  if (!has_ri) {
    FALSE
  } else {
    # Extract the RI:x entry
    ri_factor <- unlist(strsplit(appris_ref, "/"))
    ri_factor <- ri_factor[grepl("^RI:", ri_factor)]
    ri_numbers <- as.integer(sub("RI:", "", ri_factor))

    # Apply your logical conditions
    if ((ri_numbers <= 2 && appr_len >= 5) ||
        (ri_numbers <= 1 && appr_len >= 3)) {
      FALSE
    } else {
      TRUE
    }
  }
}, by = seq_len(nrow(annot))]

efficiency_factors <- annot %>% filter(efficiency_factor) %>% pull(transcript)





# # Iterate over each unique combination
for (i in 1:nrow(unique_combinations)) {
  pc1_value <- unique_combinations$PC1[i]
  pc2_value <- unique_combinations$PC2[i]

    print(c(pc1_value, pc2_value))
  
#   # Filter the data frame for the current combination
    tryCatch(
        {
            pair_pi1_1e4 <- 1-(pi1_table %>% filter(PC1 == pc1_value, PC2 == pc2_value, FDR.x <= 1e-4, trait.x.p.in.y < 1) %>%
     pull(trait.x.p.in.y) %>% pi0est())$pi0
            },
        error  = function(e){.GlobalEnv$pair_pi1_1e1 <- 1}
        )
  tryCatch(
        {
            pair_pi1_1e4_eff <- 1-(pi1_table %>% filter(PC1 == pc1_value, PC2 == pc2_value, FDR.x <= 1e-4, 
                                                    !(P1 %in% efficiency_factors), trait.x.p.in.y < 1) %>%
     pull(trait.x.p.in.y) %>% pi0est())$pi0
            },
        error  = function(e){.GlobalEnv$pair_pi1_5e2 <- 1}
        )

    tryCatch(
        {
            pair_pi1_1e10 <- 1-(pi1_table %>% filter(PC1 == pc1_value, PC2 == pc2_value, FDR.x <= 1e-10, trait.x.p.in.y < 1) %>%
     pull(trait.x.p.in.y) %>% pi0est())$pi0
            },
        error  = function(e){.GlobalEnv$pair_pi1_1e2 <- 1}
        )

    tryCatch(
        {
            print('hola')
            pair_pi1_1e10_eff <- 1-(pi1_table %>% filter(PC1 == pc1_value, PC2 == pc2_value, FDR.x <= 1e-10, 
                                                     !(P1 %in% efficiency_factors),
                                                    trait.x.p.in.y < 1) %>%
     pull(trait.x.p.in.y) %>% pi0est())$pi0
            },
        error  = function(e){
            .GlobalEnv$pair_pi1_1e4 <- 1
        }
        )
    
  
  results[[i]] <- paste0(paste(pc1_value, pc2_value, pair_pi1_1e4, pair_pi1_1e4_eff, pair_pi1_1e10, pair_pi1_1e10_eff, sep = "\t"), '\n')
}

writeLines(unlist(results), Pi1Output)
# library(tidyverse)
# library(ebpmf.alpha)
.libPaths(c('/project/yangili1/cfbuenabadn/R/x86_64-pc-linux-gnu-library/4.1',
            '/software/R-4.1.0-el8-x86_64/lib64/R/library')) 

source("/project2/yangili1/cfbuenabadn/torino/scripts/torino_factorization.R")
library(fastTopics)

set.seed(0)

args = commandArgs(trailingOnly=TRUE)
input_matrix = args[1]
min_counts = as.integer(args[2])
out_file = args[3]
gene_name <- input_matrix %>% str_split_1('/') %>% tail(n=1) %>% str_split_1('\\.') %>% head(n=1)

counts <- input_matrix %>% read_csv() %>% column_to_rownames(var = "Sample_ID") 
# row.names(counts) <- counts %>% row.names() %>% sub("\\.[^.]*$", "", .)

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


counts <- prepare_counts(counts, min_counts, n)

if (dim(counts)[1] < 2016){
    saveRDS(
      list(
        gene = gene_name,
        torino = NULL,
        coords_all = NULL
      ),
      file = out_file
    )

    } else {

rownames_split <- rownames(counts)

tissue <- sub("^.*\\.", "", rownames_split)

set.seed(1)  # for reproducibility

idx_100 <- unlist(
  tapply(seq_along(tissue), tissue, function(i) {
    sample(i, 100)
  })
)

idx_12 <- setdiff(seq_len(nrow(counts)), idx_100)


counts_100 <- counts[idx_100, , drop = FALSE]
counts_12  <- counts[idx_12,  , drop = FALSE]

library(R.utils)

run_torino_safe <- function(counts, counts_100, counts_12,
                            K,
                            torino_seeds = 1:5,
                            fixed_seed = 1,
                            timeout_sec = 3 * 60 * 60) {

  global_start <- Sys.time()

  safe_stage <- function(expr, seed = NULL) {
    start <- Sys.time()
    out <- tryCatch(
      {
        if (!is.null(seed)) set.seed(seed)
        list(
          status = "success",
          result = eval(expr),
          error = NULL
        )
      },
      error = function(e) {
        list(
          status = "failed",
          result = NULL,
          error = conditionMessage(e)
        )
      }
    )
    out$runtime_sec <- as.numeric(difftime(Sys.time(), start, units = "secs"))
    out
  }

  res <- tryCatch(
    {
      withTimeout(
        {

          ## 1) Full torino on all counts — multiple seeds
          torino_full <- lapply(torino_seeds, function(s) {
            safe_stage(
              quote(torino_factorization(counts, K = K, seed = s)),
              seed = s
            )
          })
          names(torino_full) <- paste0("seed_", torino_seeds)

          ## 2) Torino on 100 samples / tissue — fixed seed
          res_100 <- safe_stage(
            quote(torino_factorization(counts_100, K = K, seed = fixed_seed)),
            seed = fixed_seed
          )

          ## 3) Fit on remaining 12 samples — fixed seed, depends on res_100
          if (res_100$status == "success") {
            fit_12 <- safe_stage(
              quote({
                fit_init <- init_poisson_nmf(
                  as.matrix(counts_12),
                  F = as.matrix(res_100$result$EF),
                  init.method = "random"
                )

                fit_poisson_nmf(
                  as.matrix(counts_12),
                  fit0 = fit_init,
                  update.factors = NULL
                )
              }),
              seed = fixed_seed
            )
          } else {
            fit_12 <- list(
              status = "skipped",
              result = NULL,
              runtime_sec = 0,
              error = "Skipped because res_100 failed"
            )
          }

          list(
            torino = torino_full,
            res_100 = res_100,
            fit_12 = fit_12
          )
        },
        timeout = timeout_sec,
        onTimeout = "error"
      )
    },

    error = function(e) {
      list(
        torino = setNames(
          lapply(torino_seeds, function(s)
            list(status = "failed", result = NULL, runtime_sec = NA,
                 error = conditionMessage(e))),
          paste0("seed_", torino_seeds)
        ),
        res_100 = list(status = "not_run", result = NULL, runtime_sec = NA,
                       error = "Global failure"),
        fit_12 = list(status = "not_run", result = NULL, runtime_sec = NA,
                      error = "Global failure")
      )
    }
  )

  res$K <- K
  res$total_runtime_sec <- as.numeric(difftime(Sys.time(), global_start, units = "secs"))

  res
}


Ks <- c(2, 3, 5, 8, 10, 12, 15, 20, 30, 50)

torino_results <- setNames(
  lapply(Ks, function(K) {
    message("Running K = ", K)
    run_torino_safe(
      counts = counts,
      counts_100 = counts_100,
      counts_12 = counts_12,
      K = K,
      torino_seeds = 1:5,
      fixed_seed = 1
    )
  }),
  paste0("torino_K", Ks)
)



saveRDS(
  list(
    gene = gene_name,
    torino = torino_results,
    coords_all = coords_all
  ),
  file = out_file
)

}
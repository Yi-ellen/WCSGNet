library(WGCNA)
options(stringsAsFactors = FALSE)

work_dir <- "dataset/pre_data/scRNAseq_datasets_hvgs"
setwd(work_dir)

datasets <- c("Muraro", 'Baron_Mouse', 'Segerstolpe', 'Baron_Human', 'Zhang_T', 'Kang_ctrl', 'AMB', 'TM', 'Zheng68K')  
folds <- 1:5 

for (dataset in datasets) {
  cat("\n")
  cat("\n")
  cat("Processing dataset: ", dataset, "\n")

  dataset_folder <- paste0("../diff_net/", dataset)

  if (!dir.exists(dataset_folder)) {
    dir.create(dataset_folder)
  }  

  # read scRNA-seq , col: gene, row: cell
  expr_filename <- paste0(dataset, "_hvgs.csv") 
  exprData <- read.csv(expr_filename, row.names = 1)
  colnames(exprData) <- as.character(0:(ncol(exprData) - 1))
  dataExpr <- as.data.frame(exprData)


  for (fold in folds) {
    train_filename <- paste0("splits/", dataset, "_train_f", fold, ".txt") 
    train_indices <- read.table(train_filename, header = FALSE, stringsAsFactors = FALSE)
    train_indices <- as.integer(train_indices$V1)
    train_indices <- train_indices + 1 

    dataExpr_train <- dataExpr[train_indices, ]

    dim(dataExpr_train)  
    head(dataExpr_train) 


    powers <- c(1:20)  
    sft <- pickSoftThreshold(dataExpr_train, powerVector = powers, verbose = 5)

    power = sft$powerEstimate
    cat("Optimal power: ", sft$powerEstimate, "\n")
    adjacency = adjacency(dataExpr_train, power = power)

    TOM <- as.matrix(TOMsimilarity(adjacency))
    wgcna_folder <- paste0("dataset/5fold_data/", dataset, "/wgcna")

    if (!dir.exists(wgcna_folder)) {
      dir.create(wgcna_folder)
    }    

    geneNames <- colnames(dataExpr_train)

    if (is.null(dim(TOM))) {
      stop("TOM matrix is NULL or has invalid dimensions.")
    } 
    TOM[lower.tri(TOM, diag = TRUE)] <- NA  

    edgeList <- which(!is.na(TOM), arr.ind = TRUE)  
    edges <- data.frame(
      gene1 = geneNames[edgeList[, 1]],
      gene2 = geneNames[edgeList[, 2]],
      weight = TOM[edgeList]
    )
    
    cat("Number of edges before filtering:", nrow(edges), "\n")
    threshold <- 0.02
    filteredEdges <- edges[edges$weight >= threshold, ]
    cat("Number of edges after filtering:", nrow(filteredEdges), "\n")

    df_filename <- paste0(wgcna_folder, "/wgcna", "_f", fold, ".tsv")
    write.table(filteredEdges, file = df_filename, sep = "\t", row.names = FALSE, col.names = FALSE, quote = FALSE)
    cat("save successfully!")
  }

}


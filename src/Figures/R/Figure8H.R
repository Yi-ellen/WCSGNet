library(VennDiagram)
library(grid)
library(VennDiagram)
library(grid)

library(extrafont)

# Figure 8H

if(!("Times New Roman" %in% fonts())) {
  font_import(prompt = FALSE)
  loadfonts(device = "win")
}

data <- read.table("../data/AMB/AMB_top_edges/union_edges.tsv", header = TRUE, sep = "\t", fill = TRUE, na.strings ="")
set1 <- unique(na.omit(data$GABAergic)) 
set2 <- unique(na.omit(data$Glutamatergic)) 
set3 <- unique(na.omit(data$NonNeuronal))

venn_data <- list(Set1 = set1, Set2 = set2, Set3 = set3)

fill_colors <- c("#EDADC5", "#9584C1", "#6CBEC3")
line_colors <- c("#EDADC5", "#9584C1", "#6CBEC3")

svg("../../../result/Figures/AMB/AMB_threeTypes_edge100_venn.svg", 
    width = 4,   
    height = 4)


par(mar = c(3, 3, 3, 3)) 
par(xpd = TRUE)  

venn.plot <- venn.diagram(
  x = venn_data,
  category.names = c("GABAergic", "Glutamatergic", "Non-Neuronal"),
  filename = NULL,
  output = TRUE,
  fill = fill_colors,
  lwd = 2,
  col = line_colors,
  alpha = 0.5,
  
  fontfamily = "Times New Roman",
  fontface = "plain",   

  label.col = "black",
  cex = 1,
  
  cat.cex = 1,
  cat.fontfamily = "Times New Roman",
  cat.fontface = "plain",   
  cat.col = rgb(1, 1, 1, alpha = 0),

  main = "Top Weight Edges",
  main.fontfamily = "Times New Roman",
  main.fontface = "plain",  
  main.cex = 1.5,             
  main.pos = c(0.5, 0.0)   
)


plot.new()
pushViewport(viewport(width = 0.72, height = 0.72))  
grid.draw(venn.plot)
upViewport()

par(family = "Times New Roman")
legend(x = grconvertX(0.7, "npc"),  
       y = grconvertY(1.2, "npc"),
       legend = c("GABAergic", "Glutamatergic", "Non-Neuronal"),
       fill = fill_colors, 
       border = line_colors,
       bty = "n", 
       cex = 0.8,
       inset = c(0.05, 0))
par(family = "")
dev.off()




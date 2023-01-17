library(ggvenn)

data <- read.csv("D:\\FPOPP\\MoGCN\\result\\galant_sweep_14\\top20_mutation.csv")

x <- list(
  'Cluster I' = data[,'I'],
  'Cluster II' = data[,'II'],
  'Cluster III' = data[,'III']
  )

venn <- ggvenn(
  x,
  text_size  = 12,
  show_percentage = FALSE
  )
ggsave("D:\\FPOPP\\MoGCN\\result\\galant_sweep_14\\top20_mutation.png", plot = venn)
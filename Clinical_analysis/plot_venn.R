library(ggvenn)

data <- read.csv("sweeps\\top20_mutation.csv")

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
ggsave("sweeps\\top20_mutation.png", plot = venn)

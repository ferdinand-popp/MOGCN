import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import wandb
import os
import plotly.io as pio
from clustering_analysis import plot_violin, add_p_value_annotation
from clinical_selectivity import chi_test, ensg_to_gene_name

pio.templates.default = "simple_white"
# run = wandb.init(project=f"Cluster_analysis", notes='setup')
# Load the data
# df_overview = pd.read_csv(r"Z:\HiWi\Popp\TCGA_NSCLC_2022\LUNG\TCGA_LUNG_overview_table.csv", index_col=1)
#
# df_merge = df_labels.merge(df_overview, on='Sample_ID', how='left')
# print('Patient Samples: ' + str(len(df_merge.index)))

path = r'D:\FPOPP\MoGCN\result\galant_sweep_14\labels.csv'
folder = os.path.dirname(path)
df_labels = pd.read_csv(path, index_col=1)
df_labels = df_labels[['Sample_ID', 'Labels']]
if 0 in df_labels['Labels']:
    df_labels['Labels'] = df_labels['Labels'] + 1

# Load the data
df_overview = pd.read_csv(r"Z:\HiWi\Popp\TCGA_NSCLC_2022\LUNG\TCGA_LUNG_overview_table.csv", index_col=1)

df_merge = df_labels.merge(df_overview, on='Sample_ID', how='left')
print('Patient Samples: ' + str(len(df_merge.index)))

#Load cells
path = r'D:\FPOPP\MoGCN\result\galant_sweep_14\EPIC\EPIC_Clusters_cells.csv'
folder = os.path.dirname(path)
df_input = pd.read_csv(path)
df_input.set_index('Unnamed: 0', inplace=True)
df_input = df_input.T
df_input.reset_index(inplace= True)
df_input.rename({'index': 'Labels'}, axis=1, inplace=True)
df_input

df_cells = df_merge.merge(df_input, on='Labels', how='left')
df_cells.sort_values('Labels', inplace=True)
print('Patient Samples: ' + str(len(df_cells.index)))

df_cells['TILs'] = df_cells['T cell CD8+'] + df_cells['T cell CD4+']

grouped = df_cells.groupby('Labels')

for name, df in grouped:
    print(name)
    #df = df[pd.notnull(df['TILs', 'TMB', 'cyto_activity'])]
    df.dropna(subset =['TILs', 'TMB', 'cyto_activity'], inplace=True)
    print(f"TILs_vs_TMB: {df['TILs'].corr(df['TMB'])}")
    print(f"TILs_vs_cyto_activity: {df['TILs'].corr(df['cyto_activity'])}")
    print(f"cyto_activity_vs_TMB: {df['cyto_activity'].corr(df['TMB'])}")
    print(f"Mean ratio TILs_vs_TMB: {df['TILs'].mean() /df['TMB'].mean()}")
    print(f"Mean ratio TILs_vs_cyto_activity: {df['TILs'].mean() / df['cyto_activity'].mean()}")
    print(f"Mean ratio cyto_activity_vs_TMB: {df['cyto_activity'].mean() / df['TMB'].mean()}")


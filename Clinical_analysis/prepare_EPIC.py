import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import wandb
import os
import plotly.io as pio

pio.templates.default = "simple_white"


def subset_df(df, labels, df_omic):
    # Create an empty list to store the subsetted dataframes
    subsetted_dfs = {}

    # Loop through each group in labels
    for group in labels:
        # Get the Sample_IDs for the current group
        sample_ids = df[df['Labels'] == group]['Sample_ID']

        # Subset the RNAseq dataframe using the sample IDs
        subsetted_df = df_omic[df_omic['Sample_ID'].isin(sample_ids)]

        # Add the subsetted dataframe to the list
        subsetted_dfs[group] = subsetted_df

    # Return the dict of subsetted dataframes
    return subsetted_dfs


# to do get dict and use as lookup
def ensg_to_gene_name(id_list):
    df = pd.read_csv(r"Z:\HiWi\Popp\TCGA_NSCLC_2022\LUAD\RNAseq\gencode.v22.annotation.gene.probeMap", sep='\t')
    gene_values = []
    for _id in id_list:
        gene_values.append(df[df['id'] == _id]['gene'].values[0])
    return gene_values


def run_EPIC(df_labels, RNAseq):
    subsetted_dfs = subset_df(df_labels, df_labels['Labels'].unique(), RNAseq)

    for group, df in subsetted_dfs.items():
        print(group)
        df = df.set_index('Sample_ID')
        df.columns = ensg_to_gene_name(df.columns.to_list())
        df = df.T

        # drop duplicates with mean
        df.reset_index(inplace=True)
        df = df.groupby('index').mean().reset_index()
        df = df.set_index('index')
        print(df.isnull().any().any())
        df = df.mean(axis=1)
        filepath = os.path.join(folder, f'EPIC_{group}.txt')
        df.to_csv(filepath, sep='\t', index=True)


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

rna_path = r"Z:\HiWi\Popp\TCGA_NSCLC_2022\LUAD\RNAseq\LUAD_RNA_seq_36000_unscaled.csv"
RNAseq = pd.read_csv(rna_path, header=0, index_col=None).iloc[:, 1:].sort_values(by='Sample_ID')

run_EPIC(df_labels, RNAseq)

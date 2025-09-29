import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import plotly.io as pio
import plotly.graph_objects as go


pio.templates.default = "plotly_white"
# run = wandb.init(project=f"Cluster_analysis", notes='setup')
# Load the data
# df_overview = pd.read_csv(r"data\TCGA_LUNG_overview_table.csv", index_col=1)
#
# df_merge = df_labels.merge(df_overview, on='Sample_ID', how='left')
# print('Patient Samples: ' + str(len(df_merge.index)))

# load labels
path = 'sweeps/labels.csv'
folder = os.path.dirname(path)
df_labels = pd.read_csv(path, index_col=1)
df_labels = df_labels[['Sample_ID', 'Labels']]
if 0 in df_labels['Labels']:
    df_labels['Labels'] = df_labels['Labels'] + 1

# load omics
path = "gsea_results/GSEA_RESULTS_NES_QVAL.xlsx" #normal or missense

##contionue here!
df = pd.read_excel(path)

## here I'm adding a column with colors
df["Color"] = np.where(df["NES_I_vs_II"]>0, 'red', 'blue')
df["Label"] = np.where(df["QVAL_I_vs_II"]<0.05, '*', '')
df.sort_values('NES_I_vs_II', inplace=True)
# Plot
fig = go.Figure()
fig.add_trace(
    go.Bar(name='NES for I vs II',
        y=df['Hallmark'],
           x=df['NES_I_vs_II'],
           marker_color=df['Color'],
           text=df["Label"],
            orientation='h'
           ))
fig.update_layout(
    xaxis_title="Normalized Enrichment Score"
)
fig.update_xaxes(range=[-4.1, 9.1])
fig.update_traces(texttemplate = df['Label'],textposition = "outside")
fig.show()
fig.write_image(os.path.join(folder, 'GSEA', 'NES_I_vs_II.png'))

#for other comparison
## here I'm adding a column with colors
df["Color"] = np.where(df["NES_III_vs_II"]>0, 'red', 'blue')
df["Label"] = np.where(df["QVAL_III_vs_II"]<0.05, '*', '')
df.sort_values('NES_III_vs_II', inplace=True)
# Plot
fig = go.Figure()
fig.add_trace(
    go.Bar(name='NES for III vs II',
        y=df['Hallmark'],
           x=df['NES_III_vs_II'],
           marker_color=df['Color'],
           text=df["Label"],
            orientation='h'
           ))
fig.update_layout(
    xaxis_title="Normalized Enrichment Score"
)
fig.update_xaxes(range=[-4.1, 9.1])
fig.update_traces(texttemplate = df['Label'],textposition = "outside")
fig.write_image(os.path.join(folder, 'GSEA', 'NES_III_vs_II.png'))

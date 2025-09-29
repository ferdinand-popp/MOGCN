"""
MOGCN: Multi-Omics Graph Convolutional Network
==============================================

For easier usage, consider using the configuration approach:
1. Copy config_example.py to config.py
2. Edit paths in config.py to match your data
3. Run: python run_example.py

For direct command line usage, update the paths in parse_arguments() below
or use command line arguments to override the defaults.

See README.md and DATA_PREPARATION.md for detailed setup instructions.
"""

import pandas as pd
import numpy as np
import argparse
import random
import os
import torch
import torch.utils.data as Data
import torch.nn.functional as F
from torchmetrics import AUROC, F1Score, Accuracy, AveragePrecision
from torch_geometric.utils import from_networkx, train_test_split_edges
from torch_geometric.nn import GCNConv, SAGEConv, GATv2Conv, GraphConv, GAE, VGAE
import snf
import networkx as nx
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, silhouette_score, silhouette_samples
from sklearn.utils import class_weight
from sklearn.cluster import spectral_clustering, DBSCAN, AgglomerativeClustering
import wandb
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from sklearn.manifold import TSNE, MDS
import umap
# custom
from models import GCN
import autoencoder_model
from clinical_selectivity import run_clinical_selectivity
import collections


def load_data_AE_all(paths, path_overview, patient_subset, append_clinical_features=None, ):
    # read data, drop col and sort by ID ##Copynumber gene negative not working!!
    print('Loading Input Data')
    df_omics_list = []
    for path in paths:
        print(f'Loading {path}')
        df_omics_list.append(
            pd.read_csv(path, header=0, index_col=None).iloc[:, 1:].sort_values(by='Sample_ID'))

    # check quality
    for df_omics in df_omics_list:
        if True in [(df_omics.iloc[:, 1:].values < 0).any()]:
            raise Exception("No negative numbers")

    # count ids and features
    id_list = []
    in_feas = []
    # get dims and Sample_Ids for each omics
    for df_omics in df_omics_list:
        print(f'Patients in modalities: {len(df_omics["Sample_ID"])}')
        id_list.append(df_omics['Sample_ID'])
        in_feas.append(df_omics.shape[1] - 1)

    if append_clinical_features:
        df_clin_append = pd.read_csv(append_clinical_features, header=0, index_col=None).iloc[:, 1:].sort_values(
            by='Sample_ID')
        id_list.append(df_clin_append['Sample_ID'])

    if patient_subset == 'Complete':

        # subset for matching patients
        ids_overlapping = list(set.intersection(*map(set, id_list)))
        ids_overlapping.sort()
        print(f'Overlapping patients: {len(ids_overlapping)}')
        for i, df_omics in enumerate(df_omics_list):
            df_omics = df_omics.loc[df_omics['Sample_ID'].isin(ids_overlapping)].reset_index(drop=True)
            df_omics.rename(columns={df_omics.columns.tolist()[0]: 'Sample_ID'}, inplace=True)
            df_omics.sort_values(by='Sample_ID', inplace=True)
            df_omics_list[i] = df_omics.iloc[:, 1:]

        if append_clinical_features:
            df_clin_append = df_clin_append.loc[
                df_clin_append['Sample_ID'].isin(ids_overlapping)]
        # merge the multi-omics data, calculate on common samples
        merged_df = pd.concat(df_omics_list, axis=1)  # Sample + all omics
        merged_df.insert(loc=0, column='Sample_ID', value=ids_overlapping)

    # predefined Sample_IDS:
    elif patient_subset == 'Overview_LUAD':
        df_overview = pd.read_csv(path_overview, index_col=1)
        df_overview.reset_index(inplace=True)
        # check data with disease_code and for df
        merged_df = df_overview[['Sample_ID', 'disease_code']].sort_values(
            by='Sample_ID')
        merged_df = merged_df.loc[merged_df['disease_code'] == 'LUAD']
        # append every omics even when missing and fill with o left
        for i, df_omics in enumerate(df_omics_list):
            merged_df = merged_df.merge(df_omics, on='Sample_ID', how='left')
            merged_df = merged_df.fillna(0)

        # drop disease code
        merged_df.drop(['disease_code'], axis=1, inplace=True)

        if append_clinical_features:
            df_clin_append = pd.read_csv(append_clinical_features, header=0, index_col=None).iloc[:, 1:].sort_values(
                by='Sample_ID')
            df_clin_append = df_clin_append.loc[
                df_clin_append['Sample_ID'].isin(merged_df['Sample_ID'].to_list())]
    else:
        print('Choose patient subset!')

    merged_df.sort_values(by='Sample_ID', inplace=True)

    # drop_highly_correlated_features(df_omics_list, in_feas, merged_df)
    print('Number of chose patients: ' + str(len(merged_df['Sample_ID'])))

    return merged_df, in_feas, df_clin_append if append_clinical_features else None


def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def train_AE_all(data, in_feas, model='All', latent_dim=100, lr=0.001, bs=32, epochs=100, device=torch.device('cpu'),
                 modality_weights=None, mode=0,
                 topn=100):
    # name of sample
    sample_name = data['Sample_ID'].tolist()

    # change data to a Tensor
    X, Y = data.iloc[:, 1:].values.astype(float), np.zeros(data.shape[0])
    TX, TY = torch.tensor(X, dtype=torch.float, device=device), torch.tensor(Y, dtype=torch.float, device=device)
    # train a AE model
    if mode == 0 or mode == 1:
        print('Training  model...')
        Tensor_data = Data.TensorDataset(TX, TY)
        train_loader = Data.DataLoader(Tensor_data, batch_size=bs, shuffle=True)

        # initialize a model OLD ALL or SINGLES
        if model == 'All_deep':
            mmae = autoencoder_model.MMAE_ALL_DEEP(in_feas, latent_dim=latent_dim, modality_weights=modality_weights)
        elif model == 'All':
            mmae = autoencoder_model.MMAE_ALL(in_feas, latent_dim=latent_dim, modality_weights=modality_weights)
        else:
            print('Please check AE model')
        mmae.to(device)
        wandb.watch(mmae)
        mmae.train()

        mmae.train_MMAE(train_loader, learning_rate=lr, device=device, epochs=epochs, wandb=wandb)
        mmae.eval()  # before save and test, fix the variables
        torch.save(mmae, 'model/AE/MMAE_model.pkl')

    # load saved model, used for reducing dimensions
    if mode == 0 or mode == 2:
        print('Get the latent layer output...')
        mmae = torch.load('model/AE/MMAE_model.pkl')
        prev = 0
        omics_list = []
        for i, feas in enumerate(in_feas):
            omics_list.append(TX[:, prev:prev + in_feas[i]])
            prev += in_feas[i]

        latent_data, encoded_omics_tensors, decoded_omics_tensors = mmae.forward(omics_list)

        # tensor to df
        latent_df = pd.DataFrame(latent_data.detach().cpu().numpy())
        latent_df.insert(0, 'Sample_ID', sample_name)

        # tensor to df
        encoded_omics_list = []
        for item in encoded_omics_tensors:
            df = pd.DataFrame(item.detach().cpu().numpy())
            df.insert(0, 'Sample_ID', sample_name)
            encoded_omics_list.append(df)

        # tensor to df
        decoded_omics_list = []
        for item in decoded_omics_tensors:
            df = pd.DataFrame(item.detach().cpu().numpy())
            df.insert(0, 'Sample_ID', sample_name)
            decoded_omics_list.append(df)

    print('Extract features...')
    # extract_AE_features(data, in_feas, epochs, topn)
    return latent_df if 'latent_df' in locals() else None, encoded_omics_list, decoded_omics_list


def train_similarity(data, metric='cosine'):
    print('Start similarity calculation...')
    samples = data.pop('Sample_ID').to_list()

    data = data.to_numpy()

    # generate distance matrix on cosine similarity / can also do euclid
    fused_net = cdist(data, data,
                      metric=metric)

    print('Save fused adjacency matrix...')
    fused_df = pd.DataFrame(fused_net)
    fused_df.columns = samples
    fused_df.index = samples

    return fused_df


def train_SNF(feature_list, samples, metric='sqeuclidean', K=20, mu=0.5):
    print('Start similarity network fusion...')
    affinity_nets = snf.make_affinity(
        feature_list,
        metric=metric, K=K, mu=mu)

    fused_net = snf.snf(affinity_nets, K=config.K)

    print('Save fused adjacency matrix...')
    fused_df = pd.DataFrame(fused_net)
    fused_df.columns = samples
    fused_df.index = samples

    return fused_df


def train_GCN(epochs, optimizer, features, adj, labels, idx_train, regression=False, class_weights=None):
    '''
    :param epoch: training epochs
    :param optimizer: training optimizer, Adam optimizer
    :param features: the omics features
    :param adj: the laplace adjacency matrix
    :param labels: sample labels
    :param idx_train: the index of trained samples
    '''
    labels.to(device)
    loss_train = 0
    for epoch in range(epochs):
        GCN_model.train()
        # calc loss and bp
        optimizer.zero_grad()
        output = GCN_model(features, adj)
        if regression:
            loss_train = F.mse_loss(output[idx_train], labels[idx_train]).float()
            wandb.log({'Training_Loss': loss_train.data.item()})
            if (epoch + 1) % 10 == 0:
                print('Epoch: %.2f | loss train: %.4f ' % (epoch + 1, loss_train.item()))
        else:
            loss_train = F.cross_entropy(output[idx_train], labels[idx_train], weight=class_weights)
            acc_train = accuracy(output[idx_train], labels[idx_train])
            wandb.log({'Training_Loss': loss_train.data.item(), 'Training_Accuracy': acc_train.data.item()})
            if (epoch + 1) % 10 == 0:
                print('Epoch: %.2f | loss train: %.4f | acc train: %.4f' % (
                    epoch + 1, loss_train.item(), acc_train.item()))
        loss_train.backward()
        optimizer.step()

    return loss_train


def test_GCN_class(features, adj, labels, idx_test):
    '''
    :param features: the omics features
    :param adj: the laplace adjacency matrix
    :param labels: sample labels
    :param idx_test: the index of tested samples
    '''
    GCN_model.eval()
    output = GCN_model(features, adj)

    loss_test = F.cross_entropy(output[idx_test], labels[idx_test])

    # calculate the accuracy
    acc_test = accuracy(output[idx_test], labels[idx_test])

    # output is the one-hot label
    logits = output[idx_test].detach().cpu().numpy()
    # change one-hot label to digit label
    y_pred = np.argmax(logits, axis=1)
    # original label
    y_true = labels[idx_test].detach().cpu().numpy()
    print('predict label: ', y_pred)
    print('original label: ', y_true)
    wandb.log({"my_conf_mat_id": wandb.plot.confusion_matrix(
        preds=y_pred, y_true=y_true)})

    # calculate the f1 score
    f1 = f1_score(y_pred, y_true, average='weighted')

    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

    wandb.log({'Accuracy': acc_test.item(), 'F1': f1})

    val_metrics = {
        'auroc': AUROC(num_classes=nclass).to(device),
        'avg_precision': AveragePrecision(num_classes=nclass).to(device),
        'F1': F1Score(num_classes=nclass).to(device),
        'Acc:': Accuracy(top_k=1).to(device)
    }

    result = {}
    for k in val_metrics:
        result[k] = val_metrics[k](output[idx_test], labels[idx_test])
        wandb.log({k: result[k]})
        val_metrics[k].reset()

    # return accuracy and f1 score
    return acc_test.item(), f1, y_pred, y_true, logits


def predict(features, adj, sample, idx):
    '''
    :param features: the omics features
    :param adj: the laplace adjacency matrix
    :param sample: all sample names
    :param idx: the index of predict samples
    :return:
    '''
    GCN_model.eval()
    output = GCN_model(features, adj)
    predict_label = output.detach().cpu().numpy()
    predict_label = np.argmax(predict_label, axis=1).tolist()
    print(predict_label)

    res_data = pd.DataFrame({'Sample': sample, 'predict_label': predict_label})
    res_data = res_data.iloc[idx, :]
    print(res_data)

    res_data.to_csv('result/GCN_predicted_data.csv', header=True, index=False)


def load_GCN_input(adj, fea, lab=None, clinical=None):
    '''
    :param adj: the similarity matrix filename
    :param fea: the omics vector features filename
    :param lab: sample labels  filename
    :param threshold: the edge filter threshold
    '''
    print('loading data...')
    adj_df = pd.read_csv(adj, header=0, index_col=None)
    fea_df = pd.read_csv(fea, header=0, index_col=None)

    # set one val nan
    # fea_df.iloc[2,2] = np.nan #or
    # for col in fea_df.iloc[:, 1:5].columns:
    #    fea_df.loc[fea_df.sample(frac=0.1).index, col] = fea_df[col].mean()

    adj_df.rename(columns={adj_df.columns.tolist()[0]: 'Sample_ID'}, inplace=True)
    fea_df.rename(columns={fea_df.columns.tolist()[0]: 'Sample_ID'}, inplace=True)

    # align samples of different data
    adj_df.sort_values(by='Sample_ID', inplace=True)
    fea_df.sort_values(by='Sample_ID', inplace=True)

    id_check = [adj_df.Sample_ID, fea_df.Sample_ID]
    if lab:
        label_df = pd.read_csv(lab, header=0, index_col=None).iloc[:, 1:]
        label_df['labels'] = label_df.iloc[:, -1:]
        label_df.rename(columns={label_df.columns.tolist()[0]: 'Sample_ID'}, inplace=True)
        label_df = label_df[['Sample_ID', 'labels']]
        label_df.sort_values(by='Sample_ID', inplace=True)
        id_check.append(label_df.Sample_ID)
    if clinical:
        clinical_df = pd.read_csv(clinical, header=0, index_col=None).iloc[:, 1:]
        clinical_df.rename(columns={clinical_df.columns.tolist()[0]: 'Sample_ID'}, inplace=True)
        clinical_df.sort_values(by='Sample_ID', inplace=True)
        id_check.append(clinical_df.Sample_ID)

    ids_overlapping = list(set.intersection(*map(set, id_check)))
    adj_df.set_index('Sample_ID', inplace=True)
    adj_df = adj_df.loc[adj_df.index.isin(ids_overlapping), adj_df.columns.isin(ids_overlapping)]
    fea_df = fea_df.loc[fea_df['Sample_ID'].isin(ids_overlapping)].reset_index(drop=True)
    if adj_df.shape[0] != fea_df.shape[0]:
        print('Input files must have same samples.')
        exit(1)
    if lab:
        label_df = label_df.loc[label_df['Sample_ID'].isin(ids_overlapping)].reset_index(drop=True)
        if adj_df.shape[0] != label_df.shape[0]:
            print('Input files must have same samples.')
            exit(1)
    else:
        label_df = None
    if clinical:
        clinical_df = clinical_df.loc[clinical_df['Sample_ID'].isin(ids_overlapping)].reset_index(drop=True)
        if adj_df.shape[0] != clinical_df.shape[0]:
            print('Input files must have same samples.')
            exit(1)
    else:
        clinical_df = None

    # duplicate to label_plotting! remove
    # label_df is needed so take cancer type as default
    if lab is None and clinical:
        label_df = clinical_df[['Sample_ID', 'disease_code']]
        label_df = label_df.rename(columns={'disease_code': 'labels'})
    # cluster network itself by eigenverctor approach from SNF repo
    # best, second = snf.get_n_clusters(adj_df.iloc[:, 1:].values)
    # labels = spectral_clustering(adj_df.iloc[:, 1:], n_clusters=best)
    # v_measure_score(labels, label_df['label'])

    # check_same_first_column([adj_df, fea_df, label_df, clinical_df])

    return adj_df, fea_df, label_df, clinical_df


def check_same_first_column(df_list):
    # Get the first column of the first dataframe
    first_column = df_list[0].columns[0]

    # Iterate through the rest of the dataframes in the list
    for df in df_list[1:]:
        # If the first column of the current dataframe is not the same as the first column of the first dataframe, raise an error
        if df.columns[0] != first_column:
            raise ValueError("Not all dataframes have the same first column")


def accuracy(output, labels):
    pred = output.max(1)[1].type_as(labels)
    correct = pred.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def generate_graph_from_similarity(adj_df, threshold, labels=None):
    """
    A function that intakes parameter listed, filters edges by a threshold, plots graph & degreematrix and returns adjacency matrix in 0/1 format and inverse degree format
    :param adj_df: symmetric pandas df with similarities
    :param threshold: cutoffthreshold for similarity
    :param labels: for plotting coloring
    :return: returns adjacency matrix in 0/1 format and inverse degree format = adj_df , adj_m_hat
    """
    # inspect similarities
    adj_df.values[[np.arange(adj_df.shape[0])] * 2] = 0  # set diag 0
    adj_stats = adj_df.describe()
    maxs = adj_stats.loc['max']
    calculated_max_threshold = min(round(maxs, 6))

    print('Calculating the laplace adjacency matrix...')
    print('The maximum threshold is: ' + str(calculated_max_threshold))
    if calculated_max_threshold < threshold:
        print('Adjusting Threshold as it was too high to create an connected graph')
        threshold = calculated_max_threshold * 0.95
        print('New threshold: ' + str(threshold))
        wandb.config.update({'threshold': threshold}, allow_val_change=True)
    elif calculated_max_threshold > threshold / 5:
        print('Threshold may be to low')

    plt.figure(figsize=(8, 5))
    sns.distplot(maxs, hist=True, kde=True, rug=True,
                 color='darkblue',
                 kde_kws={'linewidth': 3},
                 rug_kws={'color': 'black'})
    plt.axvline(x=threshold, color='r')
    # Plot formatting
    plt.title('Density Plot of maximal similarity for each patient')
    plt.xlabel('Maximal Similarity')
    plt.ylabel('Density')
    plt.tight_layout()
    plt.savefig("result/Hist_maximal_similarity.png", format="png", dpi=300)
    wandb.log({"Hist_maximal_similarity": wandb.Image("result/Hist_maximal_similarity.png")})

    adj_m = adj_df.values
    # The SNF matrix is a completed connected graph, it is better to filter edges with a threshold
    adj_m[adj_m <= threshold] = 0  ###threshold too low? adj_m.ne(0).sum()

    # adjacency matrix after filtering
    exist = (adj_m != 0) * 1.0
    adj_df = pd.DataFrame(exist, columns=adj_df.columns, index=adj_df.index)
    np.savetxt('result/adjacency_matrix.csv', exist, delimiter=',', fmt='%d')

    # calculate the degree matrix
    factor = np.ones(adj_m.shape[1])
    res = np.dot(exist, factor)  # degree of each node
    diag_matrix = np.diag(res)  # degree matrix
    np.savetxt('result/diag.csv', diag_matrix, delimiter=',', fmt='%d')

    # calculate the laplace matrix
    d_inv = np.linalg.inv(diag_matrix)
    adj_m_hat = d_inv.dot(exist)

    # inspect graph
    plt.clf()
    G = nx.from_numpy_matrix(adj_m_hat)
    G.remove_edges_from(nx.selfloop_edges(G))
    node_color = ['red' if node == 'LUAD' else 'blue' for node in labels] if labels is not None else '#3120E0'
    nx.draw_spring(G, arrows=False, node_color=node_color, with_labels=False, node_size=15,
                   linewidths=0.2, width=0.2, label=f'Patient_graph')
    plt.savefig("result/Patient_Graph.png", format="png", dpi=300)
    wandb.log({"Patient_Graph": wandb.Image("result/Patient_Graph.png")})
    print('Edges: ' + str(len(G.edges)))
    wandb.log({'Edges': len(G.edges)})

    plt.clf()
    prep_edges = np.transpose(np.array(G.edges))
    fig = plot_in_out_degree_distributions(prep_edges, G.number_of_nodes())
    fig.savefig("result/Plot_degree_histogram.png")
    wandb.log({"Plot_degree_histogram": wandb.Image("result/Plot_degree_histogram.png")})

    return adj_df, adj_m_hat


def plot_in_out_degree_distributions(edge_index, num_of_nodes):
    """
    Cloned from https://github.com/gordicaleksa/pytorch-GAT
    """
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu().numpy()

    assert isinstance(edge_index, np.ndarray), f'Expected NumPy array got {type(edge_index)}.'

    # Store each node's input and output degree (they're the same for undirected graphs such as Cora)
    in_degrees = np.zeros(num_of_nodes, dtype=int)
    out_degrees = np.zeros(num_of_nodes, dtype=int)

    # Edge index shape = (2, E), the first row contains the source nodes, the second one target/sink nodes
    # Note on terminology: source nodes point to target/sink nodes
    num_of_edges = edge_index.shape[1]
    for cnt in range(num_of_edges):
        source_node_id = edge_index[0, cnt]
        target_node_id = edge_index[1, cnt]

        out_degrees[source_node_id] += 1  # source node points towards some other node -> increment its out degree
        in_degrees[target_node_id] += 1  # similarly here

    hist = np.zeros(np.max(out_degrees) + 1)
    for out_degree in out_degrees:
        hist[out_degree] += 1

    # to do try sns.distplot
    fig = plt.figure(figsize=(12, 8), dpi=200)  # otherwise plots are really small in Jupyter Notebook
    plt.plot(hist, color='blue')
    plt.xlabel('node degree')
    plt.ylabel('# nodes for a given out-degree')
    plt.title(f'Node out-degree distribution, Unconnected {str(np.count_nonzero(hist == 0))} from {str(num_of_nodes)}')
    plt.xticks(np.arange(0, len(hist), 10.0))

    return fig


def create_dataset(datasetname, df_adj=None, df_features=None, df_survival=None, df_clinical=None):
    """
    Creates a pytorch data object from the inputs features, adjacency and survival data.
    Returns: data object, filepath (where it was saved)
    """

    print('Creating Dataset')

    # convert matrix to G graph object
    graph = nx.from_numpy_matrix(df_adj.to_numpy())

    # remove self loops (identity)
    graph.remove_edges_from(nx.selfloop_edges(graph))

    # convert graph to Pytorch Data object ! missing features
    data = from_networkx(graph)
    data.name = datasetname

    # sort features fitting to adj matrix
    df_features.set_index('Sample_ID', inplace=True)
    df_features = df_features.reindex(df_adj.index)
    data.num_features = df_features.shape[0]

    # df to numpy array to Tensor
    x = torch.Tensor(np.array(df_features))

    # set features in data set
    data.x = x

    # save connections and survival
    data.adj_self = df_adj
    data.survival = df_survival
    data.clinical = df_clinical

    filepath = os.getcwd() + fr'\data\{data.name}\graph_dataset.pt'
    torch.save(data, filepath)

    return data, filepath


def train_GAE_adj(x, train_pos_edge_index):
    model.train()
    optimizer.zero_grad()

    z = model.encode(x, train_pos_edge_index)
    loss = model.recon_loss(z, train_pos_edge_index)
    if model_name in ['VarGCN', 'VarLinear', 'VarGAT', 'VarGraphConv']:
        loss = loss + (1 / data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)


def train_GAE_feature(x_train_cor, x_train, edge_index):
    model.train()
    optimizer.zero_grad()
    z = model.encode(x_train_cor, edge_index)
    x_rec = model.decode(z)
    loss = F.mse_loss(x_rec, x_train)
    loss.backward()
    optimizer.step()
    return float(loss)


def test_GAE(x, train_pos_edge_index, pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        # latent representation
        z = model.encode(x, train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)


def test_GAE_feature(x_test_cor, x_test, edge_index):
    model.eval()
    with torch.no_grad():
        # latent representation
        z = model.encode(x_test_cor, edge_index)
        x_rec = model.decode(z)
        loss = F.mse_loss(x_rec, x_test)
    return loss


def projection(z, dimensions, projection_type):
    """
    Dimensionality reduction via projection into 2 or 3 dimensions.
    Takes in a latent representation and outputs a dataframe with cols as dimensions.
    """
    print('Projection')
    if projection_type == 'TSNE':
        projection = TSNE(n_components=dimensions, random_state=123)
    elif projection_type == 'UMAP':
        projection = umap.UMAP(n_components=dimensions, random_state=42)  # n_neighbors=30
    elif projection_type == 'MDS':
        projection = MDS(n_components=dimensions)
    elif projection_type == 'PACMAC':
        projection = None  # Pacmac()
    else:
        print('No projection')
        pass
    result = projection.fit_transform(z)
    return pd.DataFrame(result)


def plot_silhouette_comparison(df, labels):
    """
    Calculated average silhouette score for each cluster group.
    Based on sklearn example.
    """
    df['labels'] = labels
    n_clusters = len(set(df.labels))
    X = df.iloc[:, [0, 1]].to_numpy()
    y = df.labels.to_numpy()
    fig, ax1 = plt.subplots()

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # The silhouette_score gives the average value for all the samples.
    #     # This gives a perspective into the density and separation of the formed
    #     # clusters
    silhouette_avg = silhouette_score(X, y)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, y)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[y == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.suptitle(("Silhouette analysis for clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

    plt.savefig('result/Silhoutte_avg_groups.png', dpi=300)
    wandb.log({'Silhoutte_avg_score': silhouette_avg})
    wandb.log({"Silhoutte_avg_groups": wandb.Image("result/Silhoutte_avg_groups.png")})
    return fig


def clustering_points(result_df, min_samples_per_group=30):
    """
    DBSCAN for low dimensional dataframe of latent representation.
    If settings for DBSCAN return more than 1 label for the data.
    Returns group label array.
    """
    # Clustering input result df
    try:
        clustering = DBSCAN(min_samples=min_samples_per_group).fit(result_df)  # min_samples=min_samples_per_group
        labels = clustering.labels_
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        print(f'DBSCAN: Clusters: {n_clusters_}, Excluded points:{n_noise_}')

        sil_score = 0
        # successfull clustering of groups
        if n_clusters_ > 1:
            # calculate mean Silhouette score per group
            sil_score = silhouette_score(result_df, labels)

            # # Plot silhouette figure
            # fig_silhouette = plt.figure(figsize=(8, 8))
            # # Black (ungrouped samples) removed and is used for noise instead.
            # unique_labels = set(labels)
            # colors_ = [plt.cm.Spectral(each)
            #            for each in np.linspace(0, 1, len(unique_labels))]
            # for k, col in zip(unique_labels, colors_):
            #     if k == -1:
            #         # Black used for noise.
            #         col = [0, 0, 0, 1]
            #
            #     class_member_mask = (labels == k)
            #
            #     core_samples_mask = np.zeros_like(labels, dtype=bool)
            #     core_samples_mask[clustering.core_sample_indices_] = True
            #     xy = result_df[class_member_mask & core_samples_mask]
            #     plt.plot(xy.iloc[:, 0], xy.iloc[:, 1], 'o', markerfacecolor=tuple(col),
            #              markeredgecolor='k', markersize=14)
            #
            #     xy = result_df[class_member_mask & ~core_samples_mask]
            #     plt.plot(xy.iloc[:, 0], xy.iloc[:, 1], 'o', markerfacecolor=tuple(col),
            #              markeredgecolor='k', markersize=6)
            #
            # plt.title('DBSCAN number of clusters: %d' % len(unique_labels))
            # plt.savefig('result/Clustering.png', dpi=300)
            # wandb.log({"Clustered_Patients": wandb.Image("result/Clustered_Patients.png")})
    except ValueError:
        print("No Clusters found")
        return None, None

    return pd.Series(labels, name='Labels'), sil_score


def clustering_points_agglo(df):
    k = range(3, 6)
    ac_list = [AgglomerativeClustering(n_clusters=i) for i in k]
    # Appending the silhouette scores
    silhouette_scores = {}
    silhouette_scores.fromkeys(k)

    for i, j in enumerate(k):
        silhouette_scores[j] = silhouette_score(df,
                                                ac_list[i].fit_predict(df))

    # Plotting
    y = list(silhouette_scores.values())
    fig = plt.figure()
    plt.bar(k, y)
    plt.xlabel('Number of clusters', fontsize=20)
    plt.ylabel('Mean_Silhoutte_Score(i)', fontsize=20)
    plt.title('Comparing different cluster amounts with Agglomerative')
    # wandb
    fig.savefig('result/Clustering_agglo_comparison.png', dpi=300)
    wandb.log({"Clustered_PaClustering_agglo_comparisontien": wandb.Image("result/Clustering_agglo_comparison.png")})

    # get best vis
    k_opt = max(silhouette_scores, key=silhouette_scores.get)
    wandb.log({"k_opt_agglo": k_opt})  # exclude -1 unclustered
    clustering = AgglomerativeClustering(n_clusters=k_opt).fit_predict(df)
    labels = clustering.tolist()
    # increment one to get rid of 0
    labels = [x + i for x in labels]
    labels = int_to_roman(labels)
    # Number of clusters in labels, ignoring noise if present.
    print(f'Agglomerative Clustering best k:{k_opt}, sil_score: {silhouette_scores[k_opt]}')

    return pd.Series(labels, name='Labels'), silhouette_scores[k_opt]


def plot_embedding(df, labels, type="", title="", names=""):
    """
    Plots dataframe after dimensionality reduction.
    """

    if len(df.columns) > 2:  # 3D +labels plotly
        df.columns = ['Dimension 0', 'Dimension 1', 'Dimension 2']
        df['labels'] = labels
        fig = px.scatter_3d(df, x='Dimension 0', y='Dimension 1', z='Dimension 2',
                            color=labels, title=title, opacity=0.8,
                            hover_name=names if isinstance(names, pd.Series) else labels)
        wandb.log({type: fig})
        df.drop('labels', axis=1, inplace=True)

    else:  # 2D +labels plt
        # plt.style.use('classic')  # or extensys
        fig = plt.figure()
        colors = [
            '#ff0000', '#4169e1', '#065535', '#ffd700', '#092345', '#ffc456', '#01ff00', '#ffa500', '#69b1b3',
            '#76c474', '#ebdf6a', '#009392', '#FFACC7', '#F0FF42', '#379237', '#FF6464', '#395144', '#C48557', '#00C40C'
        ]
        value_countings = labels.value_counts().sort_index()
        labels = labels.reset_index(drop=True)
        for i, x in enumerate(value_countings.index):
            df_i = df.iloc[labels[labels == x].index]
            plt.scatter(df_i.iloc[:, 0], df_i.iloc[:, 1], s=20, color=colors[i],
                        label=f'Cluster {x} (n={value_countings[x]})', alpha=0.8)
        plt.title(title)
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.legend()
        fig.savefig(f'result/{type}.png', dpi=300)
        wandb.log({type: wandb.Image(f'result/{type}.png')})


def plot_auc(aucs):
    # plt.plot(losses)
    plt.plot(aucs)
    title = '{}, Model: {}, Features: {}, AUC: {}'.format(config.dataset, model_name, data.num_features, max(aucs))
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    # plt.show()

    print('Done')


def survival_analysis(df_survival, labels):
    # Make df ready for survival analysis
    df = df_survival
    df.rename(columns={'OS.time': 'days_to_death', 'OS': 'vital_status'}, inplace=True)
    df.index.name = None
    df.to_csv('result/survival_data.csv', index=True, sep="\t")

    # Survival analysis via this method or extensive analysis with get_KM_plot_survival_clusters.R
    figure_survival = create_survival_plot(df, labels)
    figure_survival.savefig('result/Survival.png', dpi=300)
    wandb.log({"Survival": wandb.Image("result/Survival.png")})


def create_survival_plot(df, labels, path_csv=None):
    """
    Survival analysis on a dataframe containing patients ID, vital_status, days_to_death and grouped label from pipeline
    Returns plot figure
    Based on https://pub.towardsai.net/survival-analysis-with-python-tutorial-how-what-when-and-why-19a5cfb3c312
    and the lifelines package.
    """
    df['Labels'] = labels
    # kick -1 label unsorted, and empty death or survival
    df = df[df.labels != -1]
    df = df.dropna(how='any', axis=0)

    # Organize our data:
    df.loc[df.vital_status == 0, 'dead'] = 0
    df.loc[df.vital_status == 1, 'dead'] = 1

    # Create objects for groups:
    groups = sorted(set(df.labels))

    fitters = []
    # Dividing data into groups:
    for k in groups:
        class_member_mask = (df.labels == k)
        df_groups = df[class_member_mask]
        fitters.append(
            KaplanMeierFitter().fit(durations=df_groups["days_to_death"], event_observed=df_groups["dead"], label=k))

    print(fitters[0].event_table)  # event_table, predict(days), survival_function_, cumulative_density_

    # wilcoxon multi log rank test
    results_logrank = multivariate_logrank_test(df['days_to_death'], df['Labels'], df['dead'])
    # results_logrank.print_summary()

    # plot KMP
    fig = plt.figure()
    for fit in fitters:
        fit.plot(ci_show=False)  # ci_show=False
    x_ticks = [day for day in df['days_to_death'] if day % 365 == 0]  # Only pull out full years
    x_labels = ['Year ' + str(i) for i in range(len(x_ticks))]
    plt.xticks(x_ticks, x_labels)
    plt.xlim([0, 3650])  # 1825
    plt.ylim([0, 1.05])
    plt.xlabel("Time (in Years)")
    plt.ylabel("Survival Probability")
    plt.title("10 Year OS for clustered patient groups")
    plt.text(50, 0.05,
             f'multi-log-rank: {str(round(results_logrank.test_statistic, 4))}, p: {str(round(results_logrank.p_value, 4))}',
             fontsize=8)

    return fig


def pfs_analysis(df_survival, labels):
    # Make df ready for survival analysis
    df_survival.rename(columns={'PFI.time': 'days', 'PFI': 'progression'}, inplace=True)
    df_survival.index.name = None
    df_survival.to_csv('result/pfs_data.csv', index=True, sep="\t")

    # Survival analysis via this method or extensive analysis with get_KM_plot_survival_clusters.R
    figure_survival = create_pfs_plot(df_survival, labels)
    figure_survival.savefig('result/PFS.png', dpi=300)
    wandb.log({"PFS": wandb.Image("result/PFS.png")})


def create_pfs_plot(df, labels, path_csv=None):
    """
    Survival analysis on a dataframe containing patients ID, vital_status, days_to_death and grouped label from pipeline
    Returns plot figure
    Based on https://pub.towardsai.net/survival-analysis-with-python-tutorial-how-what-when-and-why-19a5cfb3c312
    and the lifelines package.
    """
    df['labels'] = labels
    # kick -1 label unsorted, and empty death or survival
    df = df[df.labels != -1]
    df = df.dropna(how='any', axis=0)

    # Create objects for groups:
    groups = sorted(set(df.labels))

    fitters = []
    # Dividing data into groups:
    for k in groups:
        class_member_mask = (df.labels == k)
        df_groups = df[class_member_mask]
        fitters.append(
            KaplanMeierFitter().fit(durations=df_groups["days"], event_observed=df_groups["progression"], label=k))

    print(fitters[0].event_table)  # event_table, predict(days), survival_function_, cumulative_density_

    # wilcoxon multi log rank test
    results_logrank = multivariate_logrank_test(df['days'], df['labels'], df['progression'])
    # results_logrank.print_summary()

    # plot KMP
    fig = plt.figure()
    for fit in fitters:
        fit.plot(ci_show=False)  # ci_show=False
    x_ticks = [day for day in df['days'] if day % 365 == 0]  # Only pull out full years
    x_labels = ['Year ' + str(i) for i in range(len(x_ticks))]
    plt.xticks(x_ticks, x_labels)
    plt.xlim([0, 3650])  # 1825
    plt.ylim([0, 1.05])
    plt.xlabel("Time (in Years)")
    plt.ylabel("Survival Probability")
    plt.title("10 Year OS for clustered patient groups")
    plt.text(50, 0.05,
             f'multi-log-rank: {str(round(results_logrank.test_statistic, 4))}, p: {str(round(results_logrank.p_value, 4))}',
             fontsize=8)

    return fig


def corrupt_features(clean_data, noise=None, variance=0.1):
    """
        Input noise for the MGAE
        """
    data = clean_data.detach()
    if noise:
        data = data + (variance ** 0.5) * torch.randn_like(data)
    else:
        # example 10% of features get 0.1 variance
        data[torch.randn_like(clean_data) < noise] = data[torch.randn_like(clean_data) < noise] + (
                variance ** 0.5) * torch.randn_like(clean_data)
    return data


def add_and_remove_edges(G, p_new_connection, p_remove_connection):
    '''
    for each node,
      add a new connection to random other node, with prob p_new_connection,
      remove a connection, with prob p_remove_connection

    operates on G in-place
    '''
    new_edges = []
    rem_edges = []

    for node in G.nodes():
        # find the other nodes this one is connected to
        connected = [to for (fr, to) in G.edges(node)]
        # and find the remainder of nodes, which are candidates for new edges
        unconnected = [n for n in G.nodes() if not n in connected]

        # probabilistically add a random edge
        if len(unconnected):  # only try if new edge is possible
            if random.random() < p_new_connection:
                new = random.choice(unconnected)
                G.add_edge(node, new)
                new_edges.append((node, new))
                # book-keeping, in case both add and remove done in same cycle
                unconnected.remove(new)
                connected.append(new)

        # probabilistically remove a random edge
        if len(connected):  # only try if an edge exists to remove
            if random.random() < p_remove_connection:
                remove = random.choice(connected)
                G.remove_edge(node, remove)
                rem_edges.append((node, remove))
                # book-keeping, in case lists are important later?
                connected.remove(remove)
                unconnected.append(remove)

    return G, rem_edges, new_edges


def calculate_overlap(x, y, labels):
    from sklearn.metrics import adjusted_rand_score

    clustering = AgglomerativeClustering(n_clusters=2).fit_predict(list(map(list, zip(*[x, y]))))
    cluster_labels = clustering.tolist()

    score = adjusted_rand_score(labels, cluster_labels)

    minority_indices = get_minority_indices(labels, cluster_labels)

    return score, minority_indices


def get_minority_indices(labels, clusters):
    minority_indices = []
    for cluster in set(clusters):
        cluster_labels = [labels[i] for i in range(len(clusters)) if clusters[i] == cluster]
        label_counts = {label: cluster_labels.count(label) for label in set(cluster_labels)}
        minority_label = min(label_counts, key=label_counts.get)
        minority_indices += [i for i in range(len(clusters)) if clusters[i] == cluster and labels[i] == minority_label]
    return minority_indices


def replace_with_zeros(df, percentage):
    # Make a copy of the original dataframe
    input_array = df.to_numpy()

    mask = np.random.random(input_array.shape)
    mask[mask < percentage] = 0
    input_array[mask == 0] = 0

    df_modified = pd.DataFrame(input_array, index=df.index, columns=df.columns)

    return df_modified


def replace_with_zeros_non_row(df, percentage):
    """
    Sets values to 0 but no row completely
    :param df:
    :param percentage:
    """
    replaced = collections.defaultdict(set)
    ix = [(row, col) for row in range(df.shape[0]) for col in range(df.shape[1])]
    random.shuffle(ix)
    to_replace = int(round(percentage * len(ix)))
    for row, col in ix:
        if len(replaced[row]) < df.shape[1] - 1:
            df.iloc[row, col] = np.nan
            to_replace -= 1
            replaced[row].add(col)
            if to_replace == 0:
                break
    df.fillna(0, inplace=True)

    return df


def int_to_roman(list):
    dict = {1: 'I',
            2: 'II',
            3: 'III',
            4: 'IV',
            5: 'V',
            6: 'VI',
            7: 'VII',
            8: 'VIII',
            9: 'IX',
            10: 'X'}
    return [dict.get(key) for key in list]


def parse_arguments():
    # first the overall settings and then the settings for AE SNF and GCN
    parser = argparse.ArgumentParser()

    ###################Check before running!!!!############################################################################
    # dataset
    parser.add_argument('--data_name', default='LUAD')
    parser.add_argument('--methods_omics', '-mo', nargs='+', type=str, help='Methods used',
                        default=['RNAseq', 'CNV', 'Methylation', 'Somatic_mutation',
                                 'Protein_array'])  # 'RNAseq', 'Somatic_mutation', 'RNAseq' #, 'CNV', 'Methylation', 'Protein_array'
    parser.add_argument('--paths_omics', '-po', nargs='+', type=str, help='The first omics file name.',
                        default=[
                            r'data\LUAD_RNA_seq.csv',
                            r'data\TCGA_LUAD_CNV_gene.csv',
                            r'data\TCGA_LUAD_Methylation_450.csv',
                            r"data\TCGA_LUAD_mutation2.csv",
                            r"data\LUAD_Protein_Array_Gene_Level.csv"
                        ])
    parser.add_argument('--path_overview', '-p_overview', type=str, help='The clinical file including survival.',
                        default=r"data\TCGA_LUNG_overview_table.csv")
    parser.add_argument('--append_clinical_features', '-p_clinical_feat', type=str, help='The clinical file features.',
                        default=r"data\TCGA_LUAD_clinical_input_features.csv")
    parser.add_argument('--labeldata', '-ld', type=str, help='Optional label file for supervised run',
                        default='')

    # overall settings
    parser.add_argument('--newdataset', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--run_similarity', type=str, default='SNF', choices=['SNF', 'Cosine_distance', 'Random'])
    parser.add_argument('--graph_from', type=str, default='Input_features', choices=['Input_features', 'Latent'])
    parser.add_argument('--supervised', default=False, action=argparse.BooleanOptionalAction)

    # supervised run
    parser.add_argument('--regression', default=False, action=argparse.BooleanOptionalAction)

    # unsupervised run
    parser.add_argument('--model', type=str, default='GAT',
                        choices=['GCN', 'VarGCN', 'Linear', 'VarLinear', 'GAT', 'LinGAT', 'GraphSAGE', 'GraphConv',
                                 'VarGraphConv', 'FeatureConv'])
    parser.add_argument('--loss_target', type=str, default='Adjacency',
                        choices=['Adjacency', 'Features'])
    parser.add_argument('--verbose', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--mask_percent', type=float, default=0.0,
                        help='Percent mask input, change the blindness type in script')  # change the blindness type in script
    parser.add_argument('--alter_graph', default=False, action=argparse.BooleanOptionalAction,
                        help='Alter the graph adjacency information for testing')

    parser.add_argument('--patient_subset', type=str, default='Overview_LUAD',
                        choices=['Complete', 'Overview_LUAD'])

    # Autoencoder
    parser.add_argument('--AE_model', type=str, default='All_deep', choices=['Old', 'All', 'All_deep'])
    parser.add_argument('--AEmode', '-aem', type=int, choices=[0, 1, 2], default=0,
                        help='Mode 0: train&integrate, Mode 1: just train, Mode 2: just integrate, default: 0.')
    parser.add_argument('--seed', '-s', type=int, default=0, help='Random seed, default=0.')
    parser.add_argument('--AEbatchsize', '-aebs', type=int, default=32, help='Training batchsize, default: 32.')
    parser.add_argument('--AElearningrate', '-aelr', type=float, default=0.001, help='Learning rate, default: 0.001.')
    parser.add_argument('--AEepoch', '-aee', type=int, default=120, help='Training epochs, default: 120.')
    parser.add_argument('--AElatent', '-ael', type=int, default=300, help='The latent layer dim.')
    parser.add_argument('--modality_weights', '-mw', nargs='+',
                        help='How to weight the modalities as input from 1 as list or "even" ',
                        default='even')
    parser.add_argument('--device', '-d', type=str, choices=['cpu', 'gpu'], default='gpu',
                        help='Training on cpu or gpu, default: cpu.')
    parser.add_argument('--topn', '-n', type=int, default=100,
                        help='Extract top N features every 10 epochs, default: 100.')
    # SNF
    parser.add_argument('--metric', '-me', type=str, choices=['correlation', 'cosine', 'dice', 'euclidean', 'hamming',
                                                              'jaccard', 'seuclidean',
                                                              'sqeuclidean',
                                                              'yule'], default='sqeuclidean',
                        help='Distance metric to compute. Must be one of available metrics in :py:func scipy.spatial.distance.pdist.')
    parser.add_argument('--K', '-k', type=int, default=20,
                        help='(0, N) int, number of neighbors to consider when creating affinity matrix. See Notes of :py:func snf.compute.affinity_matrix for more details. Default: 20.')
    parser.add_argument('--mu', '-mu', type=float, default=0.6,
                        help='(0, 1) float, Normalization factor to scale similarity kernel when constructing affinity matrix. See Notes of :py:func snf.compute.affinity_matrix for more details. Default: 0.6.')
    # GCN
    parser.add_argument('--adjdata', '-ad', type=str,
                        help='The adjacency matrix file. If newdataset = False, it uses the previous one',
                        default=r'result/Similarity_fused_matrix.csv')
    parser.add_argument('--featuredata', '-fd', type=str,
                        help='The vector feature file. If newdataset = False, it uses the previous one',
                        default=r'result/latent_data.csv')
    parser.add_argument('--mode', '-m', type=int, choices=[0, 1], default=1,
                        help='mode 0: 10-fold cross validation; mode 1: train and test a model.')
    parser.add_argument('--epochs', '-e', type=int, default=350, help='Training epochs, default: 350.')
    parser.add_argument('--learningrate', '-lr', type=float, default=0.001, help='Learning rate, default: 0.001.')
    parser.add_argument('--weight_decay', '-w', type=float, default=0.014,
                        help='Weight decay (L2 loss on parameters), methods to avoid overfitting, default: 0.014')
    parser.add_argument('--hidden', '-hd', type=int, default=80, help='Hidden layer dimension, default: 80.')
    parser.add_argument('--out_channels', type=int, default=30, help='Output layer dimension, default: 30')
    parser.add_argument('--dropout', '-dp', type=float, default=0.1,
                        help='Dropout rate, methods to avoid overfitting, default: 0.2')
    parser.add_argument('--threshold', '-t', type=float, default=0.1,  # 3 mods 0.0028
                        help='Threshold to filter edges')
    parser.add_argument('--projection', type=str, default='UMAP',
                        choices=['TSNE', 'UMAP', 'MDS'])

    args = parser.parse_args()

    # input check
    if args.data_name not in args.paths_omics[0]:
        raise ValueError('Wrong dataset!')
    if len(args.methods_omics) != len(args.paths_omics):
        raise ValueError('Check modalities')
    args.modality_weights = [1 / len(args.methods_omics)] * len(
        args.methods_omics) if args.modality_weights == 'even' else args.modality_weights

    # check inputs and setting
    if round(sum(config.modality_weights)) != 1.0:
        print('The sum of weights must be 1. Currently: ' + str(sum(config.modality_weights)))
        exit(1)

    return args


if __name__ == '__main__':

    # get settings and inputs
    args = parse_arguments()

    # initiate wandb project and run
    wandb.init(project=f"MoGCN_{args.data_name}_unsuper_thesis", config=args, notes='Complete patients')
    config = wandb.config

    # Check whether GPUs are available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set random seed
    setup_seed(config.seed)

    # create new dataset or skip and load data from last run
    if config.newdataset:
        # load data from omics (merge) and clinical data
        Merge_data, in_feas, clinical_append_df = load_data_AE_all(
            config.paths_omics, config.path_overview, config.patient_subset, config.append_clinical_features)
        # log data
        wandb.log({'Input_feature_dimensions': wandb.Table(data=[in_feas], columns=config.methods_omics)})

        # labels for plotting results
        if config.data_name == 'LUNG':
            print('Plotting input data')
            df_labelling = pd.read_csv(config.path_overview, index_col=1)
            df_labelling = df_labelling.loc[Merge_data['Sample_ID'].to_list()]
            labels_plotting = df_labelling['disease_code']
        else:
            labels_plotting = pd.Series(['Patient'] * Merge_data.shape[0])

        # mask input features
        blindness = '' if config.mask_percent != 0 else ''
        if blindness == 'random':
            Merge_data.set_index('Sample_ID', inplace=True)
            # check test in GAE
            Merge_data = replace_with_zeros(Merge_data, config.mask_percent)
            Merge_data.reset_index(inplace=True)
        if blindness == 'modality':
            Merge_data.set_index('Sample_ID', inplace=True)

            lookup = pd.DataFrame(np.ones((Merge_data.shape[0], len(in_feas))), columns=config.methods_omics,
                                  index=Merge_data.index)

            lookup_blind = replace_with_zeros_non_row(lookup, config.mask_percent)
            lookup_blind.columns = in_feas

            mask_df = pd.DataFrame()

            # Iterate over columns in lookup DataFrame
            for col in lookup_blind.columns:
                # Create a new 10-column block with the same values as the current column
                block = pd.concat([lookup_blind[col]] * col.name, axis=1)
                # Concatenate the block to the output DataFrame
                mask_df = pd.concat([mask_df, block], axis=1)

            Merge_data = Merge_data.mul(mask_df)

            Merge_data.reset_index(inplace=True)
        if blindness == 'clinical':
            # replace config.append_clinical_df with '' but add the path to load_data_AE_all as index overlap is needed for plotting
            pass

        if config.verbose:
            print(labels_plotting.value_counts())

            # plot input data
            '''Dimensionality reduction via projection'''
            input_data_2D = projection(Merge_data.iloc[:, 1:].values, dimensions=2,
                                       projection_type=config.projection)  # UMAP, TSNE, MDS
            # Plot clustering with subtypes
            title = f'{config.data_name} Input Data UMAP 2D Representation from {" ".join(config.methods_omics)}'
            plot_embedding(input_data_2D, labels=labels_plotting,
                           type='Projection_input_data_2D', title=title)

            score, minority_indices = calculate_overlap(input_data_2D[0].to_list(), input_data_2D[1].to_list(),
                                                        labels_plotting)
            print('Adj_rand_Score_input:' + str(score))
            print(
                'Inspect these patients as they dont group right:' + str(
                    labels_plotting[minority_indices].index.to_list()))
            silhouette_avg = silhouette_score(input_data_2D.values, labels_plotting.to_list())
            print('Silhoutte_Score_LUAD_LUSC_input:' + str(silhouette_avg))
            wandb.log({'Adj_rand_Score_input': score, 'Silhoutte_Score_LUAD_LUSC_input': silhouette_avg})

        # train AE
        latent_data, encoded_omics_list, _ = train_AE_all(Merge_data,
                                                          in_feas,
                                                          model=config.AE_model,
                                                          latent_dim=config.AElatent,
                                                          lr=config.AElearningrate,
                                                          bs=config.AEbatchsize,
                                                          epochs=config.AEepoch,
                                                          device=device,
                                                          modality_weights=config.modality_weights,
                                                          mode=config.AEmode,
                                                          topn=config.topn)

        # plot latent data
        '''Dimensionality reduction via projection'''
        latent_data_2D = projection(latent_data.iloc[:, 1:].values, dimensions=2, projection_type=config.projection)
        # Plot clustering with subtypes
        title = f'{config.data_name} Latent_AE_features UMAP 2D Representation from {" ".join(config.methods_omics)}'
        plot_embedding(latent_data_2D, labels=labels_plotting,
                       type='Projection_latent_AE_2D', title=title)

        # check encoded omics list!
        if config.verbose:
            for i, method in enumerate(config.methods_omics):
                latent_data = encoded_omics_list[i]
                '''Dimensionality reduction via projection'''
                latent_data_2D = projection(latent_data.iloc[:, 1:].values, dimensions=2,
                                            projection_type=config.projection)
                # Plot clustering with subtypes
                title = f'{config.data_name} Latent_AE_features UMAP 2D Representation from {" ".join(config.methods_omics)}'
                plot_embedding(latent_data_2D, labels=labels_plotting,
                               type=f'Projection_latent_AE_2D_{method}', title=title)
                # scores for each latent encoder
                if config.verbose:
                    score, minority_indices = calculate_overlap(latent_data_2D[0].to_list(),
                                                                latent_data_2D[1].to_list(),
                                                                labels_plotting)
                    print('Adj_rand_Score_AE:' + str(score))
                    print(
                        'Inspect these patients as they dont group right:' + str(
                            labels_plotting[minority_indices].index.to_list()))
                    silhouette_avg = silhouette_score(latent_data_2D.values, labels_plotting.to_list())
                    print('Silhoutte_Score_LUAD_LUSC_AE:' + str(silhouette_avg))
                    wandb.log(
                        {f'Adj_rand_Score_AE_{method}': score,
                         f'Silhoutte_Score_LUAD_LUSC_AE_{method}': silhouette_avg})
        # scores for latent
        if config.verbose:
            score, minority_indices = calculate_overlap(latent_data_2D[0].to_list(), latent_data_2D[1].to_list(),
                                                        labels_plotting)
            print('Adj_rand_Score_AE:' + str(score))
            print(
                'Inspect these patients as they dont group right:' + str(
                    labels_plotting[minority_indices].index.to_list()))
            silhouette_avg = silhouette_score(latent_data_2D.values, labels_plotting.to_list())
            print('Silhoutte_Score_LUAD_LUSC_AE:' + str(silhouette_avg))
            wandb.log({'Adj_rand_Score_AE': score, 'Silhoutte_Score_LUAD_LUSC_AE': silhouette_avg})

        # add the clinical features to the latent features from AE
        if config.append_clinical_features:
            latent_data = pd.concat(
                [latent_data.reset_index(drop=True), clinical_append_df.iloc[:, 1:].reset_index(drop=True)], axis=1)
        # save the integrated data
        latent_data.to_csv('result/latent_data.csv', header=True, index=False)
        table = wandb.Table(data=latent_data.values, columns=[str(x) for x in latent_data.columns])
        wandb.log({'df_latent_transposed': table})

        # Adjacency generation
        if config.run_similarity == 'SNF':
            if config.graph_from == 'Input_features':
                # SNF to get similarities
                data = Merge_data
                samples = data.pop('Sample_ID').to_list()
                prev = 0
                omics_list = []
                for i, feas in enumerate(in_feas):
                    omics_list.append(data.iloc[:, prev:prev + in_feas[i]].values.astype(np.float64))
                    prev += in_feas[i]
                if config.append_clinical_features:
                    omics_list.append(clinical_append_df.iloc[:, 1:].values)
                fused_df = train_SNF(omics_list, samples, metric=config.metric, K=config.K, mu=config.mu)
            else:
                print('SNF cannot run on latent data')
        elif config.run_similarity == 'Cosine_distance':
            if config.graph_from == 'Input_features':
                # Cosine distance to get similarities
                if config.append_clinical_features:
                    Merge_data = pd.concat(
                        [Merge_data.reset_index(drop=True), clinical_append_df.iloc[:, 1:].reset_index(drop=True)],
                        axis=1)
                fused_df = train_similarity(Merge_data)
            else:
                fused_df = train_similarity(latent_data)
        elif config.run_similarity == 'Random':
            print('Random Graph used')
            samples = Merge_data.pop('Sample_ID').to_list()
            # generate random sampled graph
            G_rand = nx.gnp_random_graph(len(samples), p=0.05).to_undirected()
            # push graph into adjacency df
            fused_df = pd.DataFrame(nx.to_numpy_matrix(G_rand))
            fused_df.columns = samples
            fused_df.index = samples
            print(fused_df)
        else:
            print('Please enter a similarity approach')

        # remove self loops
        np.fill_diagonal(fused_df.values, 0)
        fused_df.to_csv('result/Similarity_fused_matrix.csv', header=True, index=True)

        fig = sns.clustermap(fused_df.iloc[:, :], cmap='vlag', figsize=(9, 9))
        fig.fig.suptitle(config.run_similarity)
        fig.savefig('result/Similarity_fused_clustermap.png', dpi=300)
        wandb.log({"Similarity_fused_clustermap": wandb.Image("result/Similarity_fused_clustermap.png")})
        table = wandb.Table(dataframe=fused_df)
        wandb.log({'Fused_Similarities': table})

        # pickle.dump((Merge_data, in_feas, patient_similarity), open(r"data\merge_data_feas_snf.pkl", "wb"))
        # _dataset_unused, filepath = create_dataset(datasetname=config.dataset, df_adj=df_adj, df_features=df_features,
        #                                           df_y=df_y, df_labels=df_labels)  # contains .survival redundant

    # ! if supervised run
    if config.supervised and config.labeldata != '':
        regression = config.regression
        # GCN learn --> load input files
        adj_df, data, label_df, clinical_df = load_GCN_input(config.adjdata, config.featuredata,
                                                             lab=config.labeldata if config.labeldata != '' else None,
                                                             clinical=config.path_overview if config.path_overview != '' else None)

        # generate graph from similarities
        _, adj = generate_graph_from_similarity(adj_df, config.threshold, labels=clinical_df.disease_code)

        # change dataframe to Tensor
        adj = torch.tensor(adj, dtype=torch.float, device=device)
        features = torch.tensor(data.iloc[:, 1:].values, dtype=torch.float, device=device)
        if isinstance(label_df['labels'][0], int):
            labels = torch.tensor(label_df['labels'].to_list(), dtype=torch.long, device=device)  # change dtype
        elif isinstance(label_df['labels'][0], float):
            labels = torch.tensor(label_df['labels'].to_list(), dtype=torch.float, device=device)  # change dtype
        elif isinstance(label_df['labels'][0], str):
            label_df['labels'] = label_df['labels'].astype('category')
            label_df['categories'] = label_df['labels'].cat.codes
            labels = torch.tensor(label_df['categories'].to_list(), dtype=torch.int64, device=device)  # change dtype
        else:
            print('Please check type of labels for tensor')

        # for testing
        nclass = 1 if regression else len(label_df['labels'].value_counts())
        wandb.config.update({"labels": nclass})

        print('Begin training model...')

        # n-fold cross validation
        if config.mode == 0:
            skf = StratifiedKFold(n_splits=2, shuffle=True)

            acc_res, f1_res = [], []  # record accuracy and f1 score

            # split train and test data
            for idx_train, idx_test in skf.split(data.iloc[:, 1:], label.iloc[:, 1]):
                # initialize a model
                GCN_model = GCN(n_in=features.shape[1], n_hid=config.hidden, n_out=nclass, dropout=config.dropout)
                GCN_model.to(device)
                wandb.watch(GCN_model)

                # define the optimizer
                optimizer = torch.optim.Adam(GCN_model.parameters(), lr=config.learningrate,
                                             weight_decay=config.weight_decay)

                idx_train, idx_test = torch.tensor(idx_train, dtype=torch.long, device=device), torch.tensor(idx_test,
                                                                                                             dtype=torch.long,
                                                                                                             device=device)
                # train network
                train_GCN(config.epochs, optimizer, features, adj, labels, idx_train)

                # calculate the final accuracy and f1 score
                ac, f1, _, _, _ = test_GCN_class(features, adj, labels, idx_test)
                acc_res.append(ac)
                f1_res.append(f1)
            print('10-fold  Acc(%.4f, %.4f)  F1(%.4f, %.4f)' % (
                np.mean(acc_res), np.std(acc_res), np.mean(f1_res), np.std(f1_res)))
            wandb.log({'Mean_accuracy_fold': np.mean(acc_res), 'SD_accuracy_fold': np.std(acc_res),
                       'Mean_F1_fold': np.mean(f1_res), 'SD_F1_fold': np.std(f1_res)})
            predict(features, adj, data['Sample_ID'].tolist(), data.index.tolist())

        elif config.mode == 1:
            X_train, X_test, y_train, y_test = train_test_split(data.iloc[:, 1:], label_df['labels'],
                                                                test_size=0.20)  # for class stratify=label_df['labels'], for regression vertack package
            idx_train, idx_test = X_train.index.tolist(), X_test.index.tolist()

            y = np.array(label_df['labels'])
            class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
            class_weights = torch.tensor(class_weights, dtype=torch.float, device=device)

            GCN_model = GCN(n_in=features.shape[1], n_hid=config.hidden, n_out=nclass, dropout=config.dropout)
            GCN_model.to(device)
            wandb.watch(GCN_model)

            optimizer = torch.optim.Adam(GCN_model.parameters(), lr=config.learningrate,
                                         weight_decay=config.weight_decay)
            idx_train, idx_test = torch.tensor(idx_train, dtype=torch.long, device=device), torch.tensor(idx_test,
                                                                                                         dtype=torch.long,
                                                                                                         device=device)
            # callbacks = [EarlyStopping(monitor='val_loss', patience=5)]
            # GCN_model.set_callbacks(callbacks)

            '''
            save a best model (with the minimum loss value)
            if the loss didn't decrease in N epochs，stop the train process.
            N can be set by config.patience 
            '''
            # train early stopping

            if regression:
                train_GCN(config.epochs, optimizer, features, adj, labels, idx_train, regression=regression)

                GCN_model.eval()
                output = GCN_model(features, adj)
                loss_test = F.mse_loss(torch.flatten(output[idx_test]), labels[idx_test])
                wandb.log({'MSE': loss_test})
                print(f'Loss_test {loss_test}')

                fig = plt.figure()
                predicted_values = torch.flatten(output[idx_test]).detach().cpu().numpy()
                true_values = labels[idx_test].detach().cpu().numpy()
                plt.scatter(true_values, predicted_values, c='crimson')
                plt.yscale('log')
                plt.xscale('log')

                p1 = max(max(predicted_values), max(true_values))
                p2 = min(min(predicted_values), min(true_values))
                plt.plot([p1, p2], [p1, p2], 'b-')
                plt.xlabel('True Values', fontsize=15)
                plt.ylabel('Predictions', fontsize=15)
                plt.axis('equal')
                fig.savefig('result/True_vs_predicted_values.png', dpi=300)
                wandb.log({"True_vs_predicted_values": wandb.Image("result/True_vs_predicted_values.png")})

            else:
                # class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
                # class_weights = torch.tensor(class_weights, dtype=torch.float, device=device)

                train_GCN(config.epochs, optimizer, features, adj, labels, idx_train)  # class_weights=class_weights

                test_GCN_class(features, adj, labels, idx_test)

            # choose model
            # GCN_model.load_state_dict(torch.load('model/GCN/{}.pkl'.format(best_epoch)))
            # predict(features, adj, data['Sample_ID'].tolist(), idx_test.detach().cpu().numpy())

    # ! if unsupervised run
    if config.supervised == False:
        print('running GAE')
        # GCN learn --> load input files, create overlap and sort them; optional labels and clinical
        adj_df, data, label_df, clinical_df = load_GCN_input(config.adjdata, config.featuredata,
                                                             lab=config.labeldata if config.labeldata != '' else None,
                                                             clinical=config.path_overview if config.path_overview != '' else None)

        # generate graph from similarities
        adj_1, adj_norm = generate_graph_from_similarity(adj_df, config.threshold, clinical_df.disease_code)

        # alter graph information
        if config.alter_graph:
            old_DF_As = adj_1
            print('Altering graph adjacency')
            G = nx.from_pandas_adjacency(adj_1)
            # operates inplace on G
            add_and_remove_edges(G, p_new_connection=0.9, p_remove_connection=0.1)
            # to do function doesnt work right....
            # safety check
            G.remove_edges_from(nx.selfloop_edges(G))

            adj_1.values[:] = nx.to_numpy_matrix(G)

            node_color = ['red' if node == 'LUAD' else 'blue' for node in labels] if labels is not None else '#3120E0'
            nx.draw_spring(G, arrows=False, node_color=node_color, with_labels=False, node_size=15,
                           linewidths=0.2, width=0.2, label=f'Patient_graph')
            plt.savefig("result/Patient_Graph_altered.png", format="png", dpi=300)
            wandb.log({"Patient_Graph_altered": wandb.Image("result/Patient_Graph_altered.png")})
            print('Edges_altered: ' + str(len(G.edges)))
            wandb.log({'Edges_altered': len(G.edges)})

        # create pyg dataobject
        df_survival = clinical_df[
            ['Sample_ID', 'OS', 'OS.time', 'PFI', 'PFI.time']]  # remaining clinical data not used yet
        data, filepath = create_dataset(config.data_name, df_adj=adj_1, df_features=data, df_survival=df_survival)

        # settings for graph layers
        num_features = data.num_features
        out_channels = config.out_channels

        '''Selection of Model'''
        model_name = config.model
        dropout = config.dropout

        if config.model == 'GCN':
            model = GAE(GCNEncoder(num_features, out_channels, dropout))
        elif config.model == 'Linear':
            model = GAE(LinearEncoder(num_features, out_channels))
        elif config.model == 'VarLinear':
            model = VGAE(VariationalLinearEncoder(num_features, out_channels))
        elif config.model == 'VarGCN':
            model = VGAE(VariationalGCNEncoder(num_features, out_channels, dropout))
        elif config.model == 'GAT':
            model = GAE(GATEncoder(num_features, out_channels, dropout))
        elif config.model == 'LinGAT':
            model = GAE(LinGATEncoder(num_features, out_channels, dropout))
        elif config.model == 'GraphSAGE':
            model = GAE(GraphSAGE(num_features, out_channels, dropout))
        elif config.model == 'GraphConv':
            model = GAE(GraphConvEncoder(num_features, out_channels, dropout))
        elif config.model == 'VarGraphConv':
            model = VGAE(VariationalGraphConvEncoder(num_features, out_channels, dropout))
        elif config.model == 'FeatureConv':
            model = GAE(GCNEncoder(num_features, out_channels), MLP(out_channels, num_features))
        else:
            print('Please enter model')

        '''GPU CUDA Connection and prep&send data'''
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learningrate,
                                     weight_decay=config.weight_decay)  # optional config.decay
        wandb.watch(model)
        data.train_mask = data.val_mask = data.test_mask = None
        if config.loss_target == 'Adjacency':
            data = train_test_split_edges(data)
            train_pos_edge_index = data.train_pos_edge_index.to(device)
        elif config.loss_target == 'Features':
            assert data.edge_index.max() < data.x.size(0)
            x_train_cor = corrupt_features(data.x, 0.1).to(device)

            # drop node --> adapted node and edge --> failed as edge.max > nodes
            # train_edge_index, split_edge_mask, train_node_mask = dropout_node(data.edge_index, p=0.15)
            # train_edge_index = train_edge_index.to(device)
            # x_train = data.x[train_node_mask].to(device)
            # x_train_cor = corrupt_features(x_train, 0.1).to(device)

            # random node split -> edge not adapted
            # data = RandomNodeSplit(split='train_rest', num_val=0.1, num_test=0.1, key='x')(data)
            # x_train = (data.x[data.train_mask]).to(device) #or data.x[data.train_mask]
            # x_train_cor = corrupt_features(x_train, 0.1).to(device)
            # x_test = (data.x[data.test_mask]).to(device) #or x[data.train_mask]
            # x_test_cor = corrupt_features(x_test, 0.1).to(device)
        x = data.x.to(device)
        data = data.to(device)

        ''' Execute model training and testing'''
        best_val_auc = 0
        for epoch in range(1, config.epochs + 1):

            if config.loss_target == 'Adjacency':
                loss = train_GAE_adj(x, train_pos_edge_index)
                wandb.log({'Train_Loss': loss})

                if (epoch + 1) % 10 == 0:
                    # testing masked egdes against the reconstructed graph from the projection
                    auc, ap = test_GAE(x, train_pos_edge_index, data.test_pos_edge_index, data.test_neg_edge_index)
                    if auc > best_val_auc:
                        best_val_auc = auc
                    print(f'Training Epoch: {epoch} AUC: {auc}')
                    wandb.log({'AUC': auc, 'AP': ap})

            elif config.loss_target == 'Features':
                # assert train_edge_index.max() < x_train_cor.size(0)
                loss = train_GAE_feature(x_train_cor, data.x, data.edge_index)
                wandb.log({'Train_Loss': loss})

                if (epoch + 1) % 10 == 0:
                    test_loss = test_GAE_feature(data.x, data.x, data.edge_index)
                    wandb.log({'Test_loss': test_loss})
            else:
                raise ValueError('What should the GAE loss train for?')
        print('Training finished')

        '''Grab latent representation'''
        with torch.no_grad():
            # get latent representation {nodes, outputchannels (feature dimensions)}
            z = model.encode(x, train_pos_edge_index if config.loss_target == 'Adjacency' else data.edge_index)
            z_0 = z.cpu().numpy()  # copies it to CPU

        ''' optional save trained model '''
        # modelpath = os.path.join(os.getwd(), 'models')
        # torch.save(model.state_dict(), modelpath)

        # 3D analysis
        if config.verbose:
            # FOR ALL
            '''Dimensionality reduction via projection'''
            result_df = projection(z_0, dimensions=3, projection_type=config.projection)

            # Plot clustering with LUAD LUSC
            title = '{}, Model: {}, Features: {}, AUC: {}, Silhouette Score:{}'.format(config.projection, model_name,
                                                                                       data.num_features,
                                                                                       round(best_val_auc, 3),
                                                                                       0)
            plot_embedding(result_df, labels=label_df.labels, type='Projection_3D', title=title,
                           names=label_df.Sample_ID)

            # Clustering and Plot of the projection into patient groups
            labels, sil_score = clustering_points(result_df, min_samples_per_group=30)
            nclass = len(labels.value_counts())
            if list(labels).count(-1) > 100:
                labels, sil_score = clustering_points(result_df, min_samples_per_group=20)
                nclass = len(labels.value_counts())
            wandb.log({"clusters_3D": nclass - 1})  # exclude -1 unclustered
            # Log the table to your W&B workspace
            output_df = pd.concat([data.survival.Sample_ID, result_df, labels], axis=1)
            output_df.columns = output_df.columns.astype(str)
            table = wandb.Table(dataframe=output_df)
            wandb.log({'result_clustered_3D': table})

            title = '{}, Model: {}, Features: {}, AUC: {}, Silhouette Score:{}'.format(config.projection, model_name,
                                                                                       data.num_features,
                                                                                       round(best_val_auc, 3),
                                                                                       round(sil_score, 3))
            plot_embedding(result_df, labels=labels, type='Clustering_Patients_3D', title=title,
                           names=label_df.Sample_ID)

        # 2D analysis
        '''Dimensionality reduction via projection'''
        result_df_2D = projection(z_0, dimensions=2, projection_type=config.projection)
        result_df = result_df_2D

        # Plot 2D embedding
        title = '{}, Model: {}, Features: {}, AUC: {}, Silhouette Score:{}'.format(config.projection, model_name,
                                                                                   data.num_features,
                                                                                   round(best_val_auc, 3),
                                                                                   0)
        plot_embedding(result_df, labels=clinical_df.disease_code, type='Projection_2D', title=title,
                       names=label_df.Sample_ID)

        if config.verbose:
            score, minority_indices = calculate_overlap(result_df[0].to_list(), result_df[1].to_list(),
                                                        clinical_df.disease_code)
            print('Adj_rand_Score_GAE:' + str(score))
            print(
                'Inspect these patients as they dont group right:' + str(
                    clinical_df.disease_code[minority_indices].index.to_list()))
            silhouette_avg = silhouette_score(result_df.values, clinical_df.disease_code.to_list())
            print('Silhoutte_Score_GAE:' + str(silhouette_avg))
            wandb.log({'Adj_rand_Score_GAE': score, 'Silhoutte_Score_GAE': silhouette_avg})

        # Clustering and Plot of the projection into patient groups
        labels, sil_score = clustering_points(result_df, min_samples_per_group=20)
        nclass = len(labels.value_counts())
        if list(labels).count(-1) > 100:
            labels, sil_score = clustering_points(result_df, min_samples_per_group=15)
            nclass = len(labels.value_counts())
        wandb.log({"clusters_2D": nclass - 1})  # exclude -1 unclustered
        # Log the table to your W&B workspace
        output_df = pd.concat([data.survival.Sample_ID, result_df, labels], axis=1)
        output_df.columns = output_df.columns.astype(str)
        table = wandb.Table(dataframe=output_df)
        wandb.log({'result_clustered_2D': table})

        title = '{}, Model: {}, Features: {}, AUC: {}, Silhouette Score:{}'.format(config.projection, model_name,
                                                                                   data.num_features,
                                                                                   round(best_val_auc, 3),
                                                                                   round(sil_score, 3))
        plot_embedding(result_df, labels=labels, type='Clustering_Patients_2D', title=title, names=label_df.Sample_ID)

        # agglo clustering if DBSCAN not efficient
        result_df = result_df_2D
        labels, sil_score = clustering_points_agglo(result_df)
        output_df = pd.concat([data.survival.Sample_ID, result_df, labels], axis=1)
        output_df.columns = output_df.columns.astype(str)
        table = wandb.Table(dataframe=output_df)
        wandb.log({'result_clustered_agglo_2D': table})
        title = 'Agglomerative Clusteirng: {}, Model: {}, Features: {}, AUC: {}, Silhouette Score:{}'.format(
            config.projection, model_name,
            data.num_features,
            round(best_val_auc, 3),
            round(sil_score, 3))
        plot_embedding(result_df.iloc[:, :2], labels=labels, type='Clustering_Patients_Agglo_2D', title=title)

        # Plot average groups silhouette score
        plot_silhouette_comparison(result_df.iloc[:, :2], labels)

        # Survival analysis
        df_surv = data.survival
        survival_analysis(df_surv, labels)
        plt.clf()
        plt.cla()
        plt.close()

        # PFS analysis
        pfs_analysis(data.survival, labels)
        plt.clf()
        plt.cla()
        plt.close()

        '''
        Clinical selectivity analysis run separatedly 
        '''

wandb.finish()

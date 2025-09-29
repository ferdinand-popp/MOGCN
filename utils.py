import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from sklearn.manifold import TSNE, MDS
import umap
import wandb
import collections

# Plotting and clustering utilities

def plot_in_out_degree_distributions(edge_index, num_of_nodes):
    if isinstance(edge_index, torch.Tensor):
        edge_index = edge_index.cpu().numpy()
    assert isinstance(edge_index, np.ndarray), f'Expected NumPy array got {type(edge_index)}.'
    in_degrees = np.zeros(num_of_nodes, dtype=int)
    out_degrees = np.zeros(num_of_nodes, dtype=int)
    num_of_edges = edge_index.shape[1]
    for cnt in range(num_of_edges):
        source_node_id = edge_index[0, cnt]
        target_node_id = edge_index[1, cnt]
        out_degrees[source_node_id] += 1
        in_degrees[target_node_id] += 1
    hist = np.zeros(np.max(out_degrees) + 1)
    for out_degree in out_degrees:
        hist[out_degree] += 1
    fig = plt.figure(figsize=(12, 8), dpi=200)
    plt.plot(hist, color='blue')
    plt.xlabel('node degree')
    plt.ylabel('# nodes for a given out-degree')
    plt.title(f'Node out-degree distribution, Unconnected {str(np.count_nonzero(hist == 0))} from {str(num_of_nodes)}')
    plt.xticks(np.arange(0, len(hist), 10.0))
    return fig

def projection(z, dimensions, projection_type):
    print('Projection')
    if projection_type == 'TSNE':
        projection = TSNE(n_components=dimensions, random_state=123)
    elif projection_type == 'UMAP':
        projection = umap.UMAP(n_components=dimensions, random_state=42)
    elif projection_type == 'MDS':
        projection = MDS(n_components=dimensions)
    else:
        projection = None
    result = projection.fit_transform(z)
    return pd.DataFrame(result)

def plot_silhouette_comparison(df, labels):
    df['labels'] = labels
    n_clusters = len(set(df.labels))
    X = df.iloc[:, [0, 1]].to_numpy()
    y = df.labels.to_numpy()
    fig, ax1 = plt.subplots()
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    silhouette_avg = silhouette_score(X, y)
    sample_silhouette_values = silhouette_samples(X, y)
    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[y == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.suptitle(("Silhouette analysis for clustering on sample data "
                  "with n_clusters = %d" % n_clusters), fontsize=14, fontweight='bold')
    plt.savefig('result/Silhoutte_avg_groups.png', dpi=300)
    wandb.log({'Silhoutte_avg_score': silhouette_avg})
    wandb.log({"Silhoutte_avg_groups": wandb.Image("result/Silhoutte_avg_groups.png")})
    return fig

def clustering_points(result_df, min_samples_per_group=30):
    try:
        clustering = DBSCAN(min_samples=min_samples_per_group).fit(result_df)
        labels = clustering.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        print(f'DBSCAN: Clusters: {n_clusters_}, Excluded points:{n_noise_}')
        sil_score = 0
        if n_clusters_ > 1:
            sil_score = silhouette_score(result_df, labels)
    except ValueError:
        print("No Clusters found")
        return None, None
    return pd.Series(labels, name='Labels'), sil_score

def clustering_points_agglo(df):
    k = range(3, 6)
    ac_list = [AgglomerativeClustering(n_clusters=i) for i in k]
    silhouette_scores = {}
    silhouette_scores.fromkeys(k)
    for i, j in enumerate(k):
        silhouette_scores[j] = silhouette_score(df, ac_list[i].fit_predict(df))
    y = list(silhouette_scores.values())
    fig = plt.figure()
    plt.bar(k, y)
    plt.xlabel('Number of clusters', fontsize=20)
    plt.ylabel('Mean_Silhoutte_Score(i)', fontsize=20)
    plt.title('Comparing different cluster amounts with Agglomerative')
    fig.savefig('result/Clustering_agglo_comparison.png', dpi=300)
    wandb.log({"Clustered_PaClustering_agglo_comparisontien": wandb.Image("result/Clustering_agglo_comparison.png")})
    k_opt = max(silhouette_scores, key=silhouette_scores.get)
    wandb.log({"k_opt_agglo": k_opt})
    clustering = AgglomerativeClustering(n_clusters=k_opt).fit_predict(df)
    labels = clustering.tolist()
    labels = [x + i for x in labels]
    labels = int_to_roman(labels)
    print(f'Agglomerative Clustering best k:{k_opt}, sil_score: {silhouette_scores[k_opt]}')
    return pd.Series(labels, name='Labels'), silhouette_scores[k_opt]

def plot_embedding(df, labels, type="", title="", names=""):
    if len(df.columns) > 2:
        df.columns = ['Dimension 0', 'Dimension 1', 'Dimension 2']
        df['labels'] = labels
        fig = px.scatter_3d(df, x='Dimension 0', y='Dimension 1', z='Dimension 2',
                            color=labels, title=title, opacity=0.8,
                            hover_name=names if isinstance(names, pd.Series) else labels)
        wandb.log({type: fig})
        df.drop('labels', axis=1, inplace=True)
    else:
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
    plt.plot(aucs)
    title = '{}, Model: {}, Features: {}, AUC: {}'.format(config.dataset, model_name, data.num_features, max(aucs))
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    print('Done')

def survival_analysis(df_survival, labels):
    df = df_survival
    df.rename(columns={'OS.time': 'days_to_death', 'OS': 'vital_status'}, inplace=True)
    df.index.name = None
    df.to_csv('result/survival_data.csv', index=True, sep="\t")
    figure_survival = create_survival_plot(df, labels)
    figure_survival.savefig('result/Survival.png', dpi=300)
    wandb.log({"Survival": wandb.Image("result/Survival.png")})

def create_survival_plot(df, labels, path_csv=None):
    df['Labels'] = labels
    df = df[df.labels != -1]
    df = df.dropna(how='any', axis=0)
    df.loc[df.vital_status == 0, 'dead'] = 0
    df.loc[df.vital_status == 1, 'dead'] = 1
    groups = sorted(set(df.labels))
    fitters = []
    for k in groups:
        class_member_mask = (df.labels == k)
        df_groups = df[class_member_mask]
        fitters.append(KaplanMeierFitter().fit(durations=df_groups["days_to_death"], event_observed=df_groups["dead"], label=k))
    results_logrank = multivariate_logrank_test(df['days_to_death'], df['Labels'], df['dead'])
    fig = plt.figure()
    for fit in fitters:
        fit.plot(ci_show=False)
    x_ticks = [day for day in df['days_to_death'] if day % 365 == 0]
    x_labels = ['Year ' + str(i) for i in range(len(x_ticks))]
    plt.xticks(x_ticks, x_labels)
    plt.xlim([0, 3650])
    plt.ylim([0, 1.05])
    plt.xlabel("Time (in Years)")
    plt.ylabel("Survival Probability")
    plt.title("10 Year OS for clustered patient groups")
    plt.text(50, 0.05,
             f'multi-log-rank: {str(round(results_logrank.test_statistic, 4))}, p: {str(round(results_logrank.p_value, 4))}',
             fontsize=8)
    return fig

def pfs_analysis(df_survival, labels):
    df_survival.rename(columns={'PFI.time': 'days', 'PFI': 'progression'}, inplace=True)
    df_survival.index.name = None
    df_survival.to_csv('result/pfs_data.csv', index=True, sep="\t")
    figure_survival = create_pfs_plot(df_survival, labels)
    figure_survival.savefig('result/PFS.png', dpi=300)
    wandb.log({"PFS": wandb.Image("result/PFS.png")})

def create_pfs_plot(df, labels, path_csv=None):
    df['labels'] = labels
    df = df[df.labels != -1]
    df = df.dropna(how='any', axis=0)
    groups = sorted(set(df.labels))
    fitters = []
    for k in groups:
        class_member_mask = (df.labels == k)
        df_groups = df[class_member_mask]
        fitters.append(KaplanMeierFitter().fit(durations=df_groups["days"], event_observed=df_groups["progression"], label=k))
    results_logrank = multivariate_logrank_test(df['days'], df['labels'], df['progression'])
    fig = plt.figure()
    for fit in fitters:
        fit.plot(ci_show=False)
    x_ticks = [day for day in df['days'] if day % 365 == 0]
    x_labels = ['Year ' + str(i) for i in range(len(x_ticks))]
    plt.xticks(x_ticks, x_labels)
    plt.xlim([0, 3650])
    plt.ylim([0, 1.05])
    plt.xlabel("Time (in Years)")
    plt.ylabel("Survival Probability")
    plt.title("10 Year OS for clustered patient groups")
    plt.text(50, 0.05,
             f'multi-log-rank: {str(round(results_logrank.test_statistic, 4))}, p: {str(round(results_logrank.p_value, 4))}',
             fontsize=8)
    return fig

def corrupt_features(clean_data, noise=None, variance=0.1):
    data = clean_data.detach()
    if noise:
        data = data + (variance ** 0.5) * torch.randn_like(data)
    else:
        data[torch.randn_like(clean_data) < noise] = data[torch.randn_like(clean_data) < noise] + (
                variance ** 0.5) * torch.randn_like(clean_data)
    return data

def add_and_remove_edges(G, p_new_connection, p_remove_connection):
    new_edges = []
    rem_edges = []
    for node in G.nodes():
        connected = [to for (fr, to) in G.edges(node)]
        unconnected = [n for n in G.nodes() if not n in connected]
        if len(unconnected):
            if random.random() < p_new_connection:
                new = random.choice(unconnected)
                G.add_edge(node, new)
                new_edges.append((node, new))
                unconnected.remove(new)
                connected.append(new)
        if len(connected):
            if random.random() < p_remove_connection:
                remove = random.choice(connected)
                G.remove_edge(node, remove)
                rem_edges.append((node, remove))
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
    input_array = df.to_numpy()
    mask = np.random.random(input_array.shape)
    mask[mask < percentage] = 0
    input_array[mask == 0] = 0
    df_modified = pd.DataFrame(input_array, index=df.index, columns=df.columns)
    return df_modified

def replace_with_zeros_non_row(df, percentage):
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
    dict = {1: 'I', 2: 'II', 3: 'III', 4: 'IV', 5: 'V', 6: 'VI', 7: 'VII', 8: 'VIII', 9: 'IX', 10: 'X'}
    return [dict.get(key) for key in list]


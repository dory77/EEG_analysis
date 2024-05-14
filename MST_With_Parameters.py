#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 11:48:05 2024

@author: monir
"""


import matplotlib.pyplot as plt
import networkx as nx

def plot_max_spanning_tree(max_spanning_tree, key):
    # Position the nodes using the spring layout algorithm
    pos = nx.spring_layout(max_spanning_tree)

    # Extract edge weights
    edge_weights = nx.get_edge_attributes(max_spanning_tree, 'weight').values()
    min_weight = min(edge_weights)
    max_weight = max(edge_weights)

    # Define colormap
    cmap = plt.cm.get_cmap('coolwarm')

    # Normalize edge weights to [0, 1]
    norm = plt.Normalize(vmin=min_weight, vmax=max_weight)

    # Create a plot
    plt.figure(key, figsize=(15, 10))

    # Draw the maximum spanning tree with labels and styles
    nx.draw(max_spanning_tree, pos, with_labels=True, node_size=300, node_color='lightpink', font_size=6,
            edge_color=[cmap(norm(max_spanning_tree[u][v]['weight'])) for u, v in max_spanning_tree.edges()],
            width=2.0)  # adjust the width of the edges as needed

    # Add edge labels showing the weights with two decimals
    labels = {(u, v): f"{w:.2f}" for u, v, w in max_spanning_tree.edges(data='weight')}
    nx.draw_networkx_edge_labels(max_spanning_tree, pos, edge_labels=labels)

    # Set the title of the plot
    plt.title(f"Maximum Spanning Tree for {key} group")

    # Show the color bar for the colormap
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, label='Connectivity')

    # Show the plot
    plt.show()

# Example usage:
# plot_max_spanning_tree(max_spanning_tree, 'Group')
# plot_max_spanning_tree(max_spanning_tree,31)
#%%
def plot_max_spanning_tree(max_spanning_tree, key):
    # Position the nodes using the spring layout algorithm
    pos = nx.spring_layout(max_spanning_tree)

    # Create a plot
    plt.figure(key, figsize=(15, 10))

    # Draw the maximum spanning tree with labels and styles
    nx.draw(max_spanning_tree, pos, with_labels=True, node_size=300, node_color='pink', font_size=6)

    # Add edge labels showing the weights with two decimals
    labels = {(u, v): f"{w:.2f}" for u, v, w in max_spanning_tree.edges(data='weight')}
    nx.draw_networkx_edge_labels(max_spanning_tree, pos, edge_labels=labels)

    # Set the title of the plot
    plt.title(f"Maximum Spanning Tree for {key} group")

    # Show the plot
    plt.show()

#%%
def connectivity_calculator_per_subject(connectivity_dict, electrode_names=None, scalp_regions=None):
    """
    Calculate mean connectivity values across specified scalp regions from a dictionary of connectivity data.

    Parameters:
    - connectivity_dict (dict): Dictionary containing connectivity data for each subject.
                                Keys are subject IDs, and values are 2D arrays of connectivity data.
    - electrode_names (list, optional): List of electrode names. Defaults to None.
    - scalp_regions (dict, optional): Dictionary mapping scalp region names to lists of corresponding electrode names.
                                      Defaults to None.

    Returns:
    - dict: Dictionary containing mean connectivity DataFrames for each subject.
            Keys are subject IDs, and values are DataFrames containing mean connectivity values between scalp regions.

    Example:
    # Assuming delta_conn_ctrl is a dictionary containing connectivity data
    mean_connectivity = connectivity_calculator(delta_conn_ctrl)
    """
    if electrode_names is None:
        electrode_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3',
                           'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
    
    if scalp_regions is None:
        scalp_regions = {
            'frontal': ['Fp1', 'Fp2', 'F3', 'F4', 'Fz'],
            'occipital': ['O1', 'O2'],
            'parietal': ['P3', 'P4', 'Pz'],
            'central': ['C3', 'C4', 'Cz'],
            'temporal': ['T3', 'T4', 'T5', 'T6', 'F7', 'F8']
        }


    # Initialize an empty dictionary to store mean connectivity DataFrames for each subject
    mean_connectivity_per_subject = {}

    # Iterate over each subject's connectivity data
    for subject_id, conn_data in connectivity_dict.items():
        # Convert 2D array of connectivity data into a DataFrame
        conn_df = pd.DataFrame(conn_data, index=electrode_names, columns=electrode_names)
        
        # Initialize an empty DataFrame to store mean connectivity values between scalp regions
        mean_df = pd.DataFrame(index=scalp_regions.keys(), columns=scalp_regions.keys())

# """
# region_row, electrodes_row = ('frontal', ['Fp1', 'Fp2', 'F3', 'F4', 'Fz'])
# region_col, electrodes_col = ('occipital', ['O1', 'O2'])

# region_row, electrodes_row = ('occipital', ['O1', 'O2'])
# region_col, electrodes_col = ('frontal', ['Fp1', 'Fp2', 'F3', 'F4', 'Fz'])

# region_row, electrodes_row = ('frontal', ['Fp1', 'Fp2', 'F3', 'F4', 'Fz'])
# region_col, electrodes_col = ('frontal', ['Fp1', 'Fp2', 'F3', 'F4', 'Fz'])

# """
        # Iterate over scalp regions and calculate mean connectivity values
        for region_row, electrodes_row in scalp_regions.items():
            for region_col, electrodes_col in scalp_regions.items():
                # Extract connectivity data for the specified electrodes
                region_conn_values = conn_df.loc[electrodes_row, electrodes_col].values
                              
                if region_row != region_col:
                    # Calculate mean connectivity value for the current region
                    mean_values = np.nanmean(region_conn_values)
                
                    # Store the mean connectivity value in the DataFrame
                    mean_df.loc[region_row, region_col] = mean_values
                    
                elif region_row == region_col: # en cas de matrice carrée
                    region_conn_values[region_conn_values == 0] = np.nan
                    mean_values = np.nanmean(region_conn_values)
                    mean_df.loc[region_row, region_col] = mean_values
                            
        # Convert DataFrame to numeric type
        mean_df = mean_df.apply(pd.to_numeric)
        
        # Ensure symmetry
        mean_df[mean_df == 0] = np.nan
        for i in range(len(mean_df)):
            for j in range(len(mean_df.columns)):
                # Check if the value is NaN
                if pd.isnull(mean_df.iloc[i, j]):
                    # Check the corresponding cell in the symmetric position
                    if not pd.isnull(mean_df.iloc[j, i]):
                        # Fill the NaN cell with the corresponding non-NaN value
                        mean_df.iloc[i, j] = mean_df.iloc[j, i]
        
        # Store the mean connectivity DataFrame for the current subject
        mean_connectivity_per_subject[subject_id] = mean_df
    
    return mean_connectivity_per_subject


def calculate_psd_results(folder_path):
    psd_results = []

    for eeg_file in glob.glob(folder_path + '/*'):
        #%%%%%% étape 1: création d'époques de durées identiques
        eeg = mne.io.read_raw_fif(eeg_file)
        eeg.load_data()
        #Divide continuous raw data into equal-sized consecutive epochs, Creating Fixed-Length Epochs.
        epochs = mne.make_fixed_length_epochs(eeg, duration=2, preload=True)

        #%%%%%% étape 2: calcul des puissances de chaque bande pour toutes les époques et tous les canaux
        #Calculating Power Spectral Density (PSD)
        # frequency_bands: A dictionary defining frequency bands of interest
        frequency_bands = {"delta": [0.5, 4], "theta": [4, 8], "alpha": [8, 12], "beta": [12, 30], "gamma": [30, 45]}
        spectrum = epochs.compute_psd("welch", average="mean", picks="eeg", fmin=0.5, fmax=45.0) # 91 epochs × 19 channels × 90 freqs, 0.5-45.0 Hz
        # Normalizing the PSDs
        psds, freqs = spectrum.get_data(return_freqs=True)
        psds = psds / np.sum(psds, axis=-1, keepdims=True) # normalize the PSDs

        bandpower = []  #Calculating Band Powers
        for fmin, fmax in frequency_bands.values():
            psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
            bandpower.append(psds_band.reshape(len(psds), -1))
        bandpower_global_means = [np.mean(band) for band in bandpower]
        delta_power, theta_power, alpha_power, beta_power, gamma_power = bandpower_global_means
        # Extract subject ID from the filename
        subject_id = os.path.basename(eeg_file).split('_')[1].split('.')[0]

        # Append results to the list
        psd_results.append([subject_id, delta_power, theta_power, alpha_power, beta_power, gamma_power])

    psd_results = pd.DataFrame(psd_results, columns=['subject_id', 'delta', 'theta', 'alpha', 'beta', 'gamma'])
    psd_results['subject_id'] = pd.to_numeric(psd_results['subject_id'], errors='coerce')
    psd_results.sort_values(by='subject_id', inplace=True)
    psd_results.set_index('subject_id', inplace=True)
    
    return psd_results



def plot_heatmap(subjects_dict, frequency_band, index):
    """
    Plot a heatmap of overall connectivity values for a given frequency band.

    Parameters:
    - subjects_dict (dict): Dictionary containing connectivity DataFrames for each subject.
                            Keys are subject IDs, and values are DataFrames containing connectivity values.
    - frequency_band (str): Frequency band for which the heatmap is plotted.
    - index (int): Index of the plot (for labeling purposes).

    Returns:
    None

    Example:
    # subjects_dict contains connectivity data for multiple subjects
    plot_heatmap(subjects_dict, 'Theta', 1)
    """
    # Calculate the overall mean connectivity across all subjects
    overall_mean_connectivity = None
    
    for subject_id, df in subjects_dict.items():
        
        if overall_mean_connectivity is None:
            overall_mean_connectivity = df
        else:
            overall_mean_connectivity += df
    
    overall_mean_connectivity /= len(subjects_dict)  # Calculate the mean
    
    # Plot the heatmap
    plt.figure(figsize=(15, 10))
    mask = np.triu(np.ones_like(overall_mean_connectivity, dtype=bool))
    np.fill_diagonal(mask, False)
    ax = plt.axes()
    ax.set_title(f'Overall Mean Connectivity Heatmap - {frequency_band}')
    sns.heatmap(overall_mean_connectivity, annot=True, cmap='coolwarm', fmt=".2f", mask=mask, ax=ax, vmin=0.05, vmax=0.45) 

    # # Define the filename for the plot
    # filename = f'overall_heatmap_{index}_{frequency_band}.png'
    
    # # Check if the plot file already exists
    # if not os.path.exists(filename):
    #     plt.savefig(filename)  # Save the plot as an image file
    plt.show()
#%%

from pycrostates.metrics import silhouette_score, calinski_harabasz_score, dunn_score, davies_bouldin_score
from mne.channels import make_standard_montage
from mne_bids import BIDSPath, read_raw_bids, print_dir_tree, make_report
import matplotlib
from pycrostates.io import ChData
from pycrostates.preprocessing import extract_gfp_peaks, resample, apply_spatial_filter
from pycrostates.datasets import lemon
from pycrostates.cluster import ModKMeans
import pycrostates
from mne.viz import plot_ch_adjacency, plot_topomap
from mne.channels import find_ch_adjacency
from mne.datasets import fetch_fsaverage
from mne.datasets import eegbci
import mne
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import os.path as op
import re
from mne_connectivity import spectral_connectivity_epochs 
from mne_connectivity.viz import plot_sensors_connectivity
from mne.datasets import sample
import os
import glob
import pandas as pd
import mne
import seaborn as sns
import numpy as np
from IPython.display import display
from scipy.stats import mannwhitneyu
import scipy
from scipy.stats import kruskal
import dabest
import networkx as nx

%matplotlib qt
matplotlib.use('Qt5Agg')
print("mne version", mne.__version__)  # mne version 1.6.0


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

montage = make_standard_montage('standard_1020')  # montage.plot()

ord_channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3','C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']

#montage.plot(kind="3d", show=False)   
    
    
#%%
  
  
    # Control group
    
pheno = pd.read_csv('/home/monir/Monir/IPNP/Phenotype/phenotype.csv') #phenotype data
pheno_ctrl = pheno[pheno['status']=='ctrl']

input_ctrl_group = '/home/monir/Monir/IPNP/22q11_analysis/Data_controls' #EEG recording
psd_results_ctrl = calculate_psd_results(input_ctrl_group)


# Extract the numeric part of the subject_id column
pheno_ctrl['subject_id'] = pheno_ctrl['subject_id'].str.extract(r'(\d+)')

# Convert the subject_id column to numeric type
pheno_ctrl['subject_id'] = pd.to_numeric(pheno_ctrl['subject_id'])
merged_ctrl = pheno_ctrl.merge(psd_results_ctrl, left_on='subject_id', right_index=True)  
    
#%%   
    
    # Deletion group
    
pheno = pd.read_csv('/home/monir/Monir/IPNP/Phenotype/phenotype.csv')
pheno_del = pheno[pheno['status']=='del']

input_del_group = '/home/monir/Monir/IPNP/22q11_analysis/Data_del'
psd_results_del = calculate_psd_results(input_del_group)


# Extract the numeric part of the subject_id column
pheno_del['subject_id'] = pheno_del['subject_id'].str.extract(r'(\d+)')

# Convert the subject_id column to numeric type
pheno_del['subject_id'] = pd.to_numeric(pheno_del['subject_id'])
merged_del = pheno_del.merge(psd_results_del, left_on='subject_id', right_index=True)
 
    
  #%%    
    # Del + Scz 
    
    
pheno = pd.read_csv('/home/monir/Monir/IPNP/Phenotype/phenotype.csv')
pheno_del_scz = pheno[pheno['status']=='del_scz']

input_del_scz_group = '/home/monir/Monir/IPNP/22q11_analysis/Data_del_scz'
psd_results_del_scz = calculate_psd_results(input_del_scz_group)

wpli_results_del_scz = []

# Extract the numeric part of the subject_id column
pheno_del_scz['subject_id'] = pheno_del_scz['subject_id'].str.extract(r'(\d+)')

# Convert the subject_id column to numeric type
pheno_del_scz['subject_id'] = pd.to_numeric(pheno_del_scz['subject_id'])
merged_del_scz = pheno_del_scz.merge(psd_results_del_scz, left_on='subject_id', right_index=True)
    
    
#%%
    
electrode_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3',
                           'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']

# Initialize dictionaries to store connectivity results for each subject
wpli_results_del = {}  # Dictionary to store connectivity results for each subject
delta_conn_del = {}    # Dictionary to store delta band connectivity for each subject
theta_conn_del = {}  
alpha_conn_del = {}  
beta_conn_del = {}  
gamma_conn_del = {}  
# Iterate over each subject
for subject in pheno_del['subject_id']:
    subject_id_str = 'subject_' + str(subject) + '.fif'
    eeg_file = os.path.join(input_del_group, subject_id_str)
    
    # Load EEG data for the subject
    eeg = mne.io.read_raw_fif(eeg_file, preload=True)
    
    # Create fixed length epochs
    epochs = mne.make_fixed_length_epochs(eeg, duration=2, preload=True)
    
    # Compute connectivity
    conn_del = spectral_connectivity_epochs(epochs, method='wpli', fmin=(0.5, 4, 8, 12, 30), fmax=(4, 8, 12, 30, 45),
                                             faverage=True).get_data(output='dense')
     # Store the connectivity results in the dictionaries
    wpli_results_del[subject] = conn_del  # All frequency bands
    delta_conn_del[subject] = conn_del[:, :, 0]    # Storing only the delta band connectivity for each subject
    theta_conn_del[subject] = conn_del[:, :, 1]
    alpha_conn_del[subject] = conn_del[:, :, 2]
    beta_conn_del[subject] = conn_del[:, :, 3]
    gamma_conn_del[subject] = conn_del[:, :, 4]
    
    
# Create a dictionary to store DataFrames for each subject
alpha_conn_df_dict_del = {}
delta_conn_df_dict_del = {}
theta_conn_df_dict_del = {}
for subject_id, conn_data in theta_conn_del.items():
         #Convert 2D array of connectivity data into a DataFrame
    conn_df = pd.DataFrame(conn_data, index=electrode_names, columns=electrode_names)
    # Convert DataFrame to numeric type
    conn_df = conn_df.apply(pd.to_numeric)

    # Ensure symmetry
    conn_df[conn_df == 0] = np.nan
    for i in range(len(conn_df)):
        for j in range(len(conn_df.columns)):
            # Check if the value is NaN
            if pd.isnull(conn_df.iloc[i, j]):
                # Check the corresponding cell in the symmetric position
                if not pd.isnull(conn_df.iloc[j, i]):
                    # Fill the NaN cell with the corresponding non-NaN value
                    conn_df.iloc[i, j] = conn_df.iloc[j, i]

    np.fill_diagonal(conn_df.values, 0)
    # Store the DataFrame in the dictionary with the subject ID as the key
    theta_conn_df_dict_del[subject_id] = conn_df  
    

# Initialize dictionaries to store results
betweenness_centrality_results_Del = {} # Measure of node
degree_results_Del = {} # Measure of node
leaf_fraction_results_Del = {} # Measure of network
diameter_results_Del = {} # Measure of network   # The diameter of a graph is defined as the maximum shortest path length between any pair of nodes in the graph. In simpler terms, it represents the longest shortest path that can be found within the graph.
# dictionary where the keys represent subjects, and the values represent the diameter of the maximum spanning tree for each subject??
eccentricity_results_Del = {} # Measure of node
tree_hierarchy_results_Del = {} # Measure of network

# Loop through each dataframe in the alpha_conn_df_dict
for key, df in theta_conn_df_dict_del.items():
    # Create a graph
    G = nx.Graph()

    # Loop through each region in the dataframe
    for i in range(len(df)):
        # Loop through each subsequent region to avoid duplicates
        for j in range(i + 1, len(df)):
            # Add an edge between two regions with the weight being the similarity value
            G.add_edge(df.index[i], df.index[j], weight=df.iloc[i, j])

    # Compute the maximum spanning tree of the graph
    max_spanning_tree = nx.maximum_spanning_tree(G)

    # Compute betweenness centrality
    betweenness_centrality = nx.betweenness_centrality(max_spanning_tree)

    # Compute degree
    degree = dict(max_spanning_tree.degree())

    # Compute leaf fraction
    leaves = [node for node in max_spanning_tree.nodes() if max_spanning_tree.degree(node) == 1]
    leaf_fraction = len(leaves) / len(max_spanning_tree.nodes())

    # Compute diameter
    diameter = nx.diameter(max_spanning_tree)

    # Compute eccentricity
    eccentricity = nx.eccentricity(max_spanning_tree)

    # Compute tree hierarchy (depth-first search tree rooted at an arbitrary node)
    root_node = next(iter(max_spanning_tree.nodes()))  # Choose an arbitrary root node
    tree_hierarchy = nx.dfs_tree(max_spanning_tree, root_node)

    # Store results in dictionaries
    betweenness_centrality_results_Del[key] = betweenness_centrality
    degree_results_Del[key] = degree
    leaf_fraction_results_Del[key] = leaf_fraction
    diameter_results_Del[key] = diameter
    eccentricity_results_Del[key] = eccentricity
    tree_hierarchy_results_Del[key] = tree_hierarchy
    
#%%
# deletion + schizophrenia
electrode_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3',
                           'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']

# Initialize dictionaries to store connectivity results for each subject
wpli_results_del_scz = {}  # Dictionary to store connectivity results for each subject
delta_conn_del_scz = {}    # Dictionary to store delta band connectivity for each subject
theta_conn_del_scz = {}  
alpha_conn_del_scz = {}  
beta_conn_del_scz = {}  
gamma_conn_del_scz = {}  
# Iterate over each subject
for subject in pheno_del_scz['subject_id']:
    subject_id_str = 'subject_' + str(subject) + '.fif'
    eeg_file = os.path.join(input_del_scz_group, subject_id_str)
    
    # Load EEG data for the subject
    eeg = mne.io.read_raw_fif(eeg_file, preload=True)
    
    # Create fixed length epochs
    epochs = mne.make_fixed_length_epochs(eeg, duration=2, preload=True)
    
    # Compute connectivity
    conn_del_scz = spectral_connectivity_epochs(epochs, method='wpli', fmin=(0.5, 4, 8, 12, 30), fmax=(4, 8, 12, 30, 45),
                                             faverage=True).get_data(output='dense')
    
    # Store the connectivity results in the dictionaries
    wpli_results_del_scz[subject] = conn_del_scz  # All frequency bands
    delta_conn_del_scz[subject] = conn_del_scz[:, :, 0]    # Storing only the delta band connectivity for each subject
    theta_conn_del_scz[subject] = conn_del_scz[:, :, 1]
    alpha_conn_del_scz[subject] = conn_del_scz[:, :, 2]
    beta_conn_del_scz[subject] = conn_del_scz[:, :, 3]
    gamma_conn_del_scz[subject] = conn_del_scz[:, :, 4]
    
    
    
# Create a dictionary to store DataFrames for each subject
alpha_conn_df_dict_del_scz = {}
delta_conn_df_dict_del_scz = {}
theta_conn_df_dict_del_scz = {}
for subject_id, conn_data in theta_conn_del_scz.items():
         #Convert 2D array of connectivity data into a DataFrame
    conn_df = pd.DataFrame(conn_data, index=electrode_names, columns=electrode_names)
    # Convert DataFrame to numeric type
    conn_df = conn_df.apply(pd.to_numeric)

    # Ensure symmetry
    conn_df[conn_df == 0] = np.nan
    for i in range(len(conn_df)):
        for j in range(len(conn_df.columns)):
            # Check if the value is NaN
            if pd.isnull(conn_df.iloc[i, j]):
                # Check the corresponding cell in the symmetric position
                if not pd.isnull(conn_df.iloc[j, i]):
                    # Fill the NaN cell with the corresponding non-NaN value
                    conn_df.iloc[i, j] = conn_df.iloc[j, i]

    np.fill_diagonal(conn_df.values, 0)
    # Store the DataFrame in the dictionary with the subject ID as the key
    theta_conn_df_dict_del_scz[subject_id] = conn_df
    
    
    
# Initialize dictionaries to store results
# In graph theory, betweenness centrality is a measure of centrality in a graph based on shortest paths.
                                # https://en.wikipedia.org/wiki/Betweenness_centrality
betweenness_centrality_results_Del_Scz = {} # Measure of node
degree_results_Del_Scz = {} # Measure of node
leaf_fraction_results_Del_Scz = {} # Measure of network
diameter_results_Del_Scz = {} # Measure of network   # The diameter of a graph is defined as the maximum shortest path length between any pair of nodes in the graph. In simpler terms, it represents the longest shortest path that can be found within the graph.
# dictionary where the keys represent subjects, and the values represent the diameter of the maximum spanning tree for each subject??
eccentricity_results_Del_Scz = {} # Measure of node
tree_hierarchy_results_Del_Scz = {} # Measure of network

# Loop through each dataframe in the alpha_conn_df_dict
for key, df in delta_conn_df_dict_del_scz.items():
    # Create a graph
    G = nx.Graph()

    # Loop through each region in the dataframe
    for i in range(len(df)):
        # Loop through each subsequent region to avoid duplicates
        for j in range(i + 1, len(df)):
            # Add an edge between two regions with the weight being the similarity value
            G.add_edge(df.index[i], df.index[j], weight=df.iloc[i, j])

    # Compute the maximum spanning tree of the graph
    max_spanning_tree = nx.maximum_spanning_tree(G)

    # Compute betweenness centrality
    betweenness_centrality = nx.betweenness_centrality(max_spanning_tree)

    # Compute degree
    degree = dict(max_spanning_tree.degree())

    # Compute leaf fraction
    leaves = [node for node in max_spanning_tree.nodes() if max_spanning_tree.degree(node) == 1]
    leaf_fraction = len(leaves) / len(max_spanning_tree.nodes())

    # Compute diameter
    diameter = nx.diameter(max_spanning_tree)

    # Compute eccentricity
    eccentricity = nx.eccentricity(max_spanning_tree)

    # Compute tree hierarchy (depth-first search tree rooted at an arbitrary node)
    root_node = next(iter(max_spanning_tree.nodes()))  # Choose an arbitrary root node
    tree_hierarchy = nx.dfs_tree(max_spanning_tree, root_node)

    # Store results in dictionaries
    betweenness_centrality_results_Del_Scz[key] = betweenness_centrality
    degree_results_Del_Scz[key] = degree
    leaf_fraction_results_Del_Scz[key] = leaf_fraction
    diameter_results_Del_Scz[key] = diameter
    eccentricity_results_Del_Scz[key] = eccentricity
    tree_hierarchy_results_Del_Scz[key] = tree_hierarchy
    
    
#%% 
    
wpli_results_ctrl = {}  # Dictionary to store connectivity results for each subject
delta_conn_ctrl = {}    # Dictionary to store delta band connectivity for each subject
theta_conn_ctrl = {}  
alpha_conn_ctrl = {}  # Dictionary of arrays
beta_conn_ctrl = {}  
gamma_conn_ctrl = {}

# Iterate over each subject
for subject in pheno_ctrl['subject_id']:
    subject_id_str = 'subject_' + str(subject) + '.fif'
    eeg_file = os.path.join(input_ctrl_group, subject_id_str)
    
    # Load EEG data for the subject
    eeg = mne.io.read_raw_fif(eeg_file, preload=True)
    
    # Create fixed length epochs
    epochs = mne.make_fixed_length_epochs(eeg, duration=2, preload=True)
    
    # Compute connectivity
    conn_ctrl = spectral_connectivity_epochs(epochs, method='wpli', fmin=(0.5, 4, 8, 12, 30), fmax=(4, 8, 12, 30, 45),
                                             faverage=True).get_data(output='dense')
    

    # Store the connectivity results in the dictionaries
    wpli_results_ctrl[subject] = conn_ctrl  # All frequency bands
    delta_conn_ctrl[subject] = conn_ctrl[:, :, 0]    # Storing only the delta band connectivity for each subject
    theta_conn_ctrl[subject] = conn_ctrl[:, :, 1]
    alpha_conn_ctrl[subject] = conn_ctrl[:, :, 2]
    beta_conn_ctrl[subject] = conn_ctrl[:, :, 3]
    gamma_conn_ctrl[subject] = conn_ctrl[:, :, 4]
    
    
# Create a dictionary to store DataFrames for each subject
alpha_conn_df_dict = {}
delta_conn_df_dict = {}
theta_conn_df_dict = {}
for subject_id, conn_data in theta_conn_ctrl.items():
         #Convert 2D array of connectivity data into a DataFrame
    conn_df = pd.DataFrame(conn_data, index=electrode_names, columns=electrode_names)
    # Convert DataFrame to numeric type
    conn_df = conn_df.apply(pd.to_numeric)

    # Ensure symmetry
    conn_df[conn_df == 0] = np.nan
    for i in range(len(conn_df)):
        for j in range(len(conn_df.columns)):
            # Check if the value is NaN
            if pd.isnull(conn_df.iloc[i, j]):
                # Check the corresponding cell in the symmetric position
                if not pd.isnull(conn_df.iloc[j, i]):
                    # Fill the NaN cell with the corresponding non-NaN value
                    conn_df.iloc[i, j] = conn_df.iloc[j, i]

    np.fill_diagonal(conn_df.values, 0)
    # Store the DataFrame in the dictionary with the subject ID as the key
    theta_conn_df_dict[subject_id] = conn_df
    
    
# Initialize dictionaries to store results
betweenness_centrality_results_ctrl = {} # Measure of node
degree_results_ctrl = {} # Measure of node
leaf_fraction_results_ctrl = {} # Measure of network
diameter_results_ctrl = {} # Measure of network   # The diameter of a graph is defined as the maximum shortest path length between any pair of nodes in the graph. In simpler terms, it represents the longest shortest path that can be found within the graph.
# dictionary where the keys represent subjects, and the values represent the diameter of the maximum spanning tree for each subject??
eccentricity_results_ctrl = {} # Measure of node
tree_hierarchy_results_ctrl = {} # Measure of network

# Loop through each dataframe in the alpha_conn_df_dict
for key, df in delta_conn_df_dict.items():
    # Create a graph
    G = nx.Graph()

    # Loop through each region in the dataframe
    for i in range(len(df)):
        # Loop through each subsequent region to avoid duplicates
        for j in range(i + 1, len(df)):
            # Add an edge between two regions with the weight being the similarity value
            G.add_edge(df.index[i], df.index[j], weight=df.iloc[i, j])

    # Compute the maximum spanning tree of the graph
    max_spanning_tree = nx.maximum_spanning_tree(G)

    # Compute betweenness centrality
    betweenness_centrality = nx.betweenness_centrality(max_spanning_tree)

    # Compute degree
    degree = dict(max_spanning_tree.degree())

    # Compute leaf fraction
    leaves = [node for node in max_spanning_tree.nodes() if max_spanning_tree.degree(node) == 1]
    leaf_fraction = len(leaves) / len(max_spanning_tree.nodes())

    # Compute diameter
    diameter = nx.diameter(max_spanning_tree)

    # Compute eccentricity
    eccentricity = nx.eccentricity(max_spanning_tree)

    # Compute tree hierarchy (depth-first search tree rooted at an arbitrary node)
    root_node = next(iter(max_spanning_tree.nodes()))  # Choose an arbitrary root node
    tree_hierarchy = nx.dfs_tree(max_spanning_tree, root_node)

    # Store results in dictionaries
    betweenness_centrality_results_ctrl[key] = betweenness_centrality
    degree_results_ctrl[key] = degree
    leaf_fraction_results_ctrl[key] = leaf_fraction
    diameter_results_ctrl[key] = diameter
    eccentricity_results_ctrl[key] = eccentricity
    tree_hierarchy_results_ctrl[key] = tree_hierarchy   
  
    
delta_conn_df_dict.keys()
plot_max_spanning_tree(max_spanning_tree,31)
    

#%% Stat for Leaf Fraction

data_group1 = list(leaf_fraction_results_ctrl.values())
data_group2 = list(leaf_fraction_results_Del.values())
data_group3 = list(leaf_fraction_results_Del_Scz.values())


statistic, p_value = kruskal(data_group1, data_group2, data_group3)


print("Kruskal-Wallis Test:")
print("Statistic:", statistic)
print("P-value:", p_value)


alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis. There are significant differences between groups.")
else:
    print("Fail to reject the null hypothesis. There are no significant differences between groups.")
    


mean_1 = sum(data_group1)/len(data_group1)
mena_2 = sum(data_group2)/len(data_group2)
mean_3 = sum(data_group3)/len(data_group3)


d = sum(diameter_results_Del)/len(diameter_results_Del)
d_s = sum(diameter_results_Del_Scz)/len(diameter_results_Del_Scz)
c = sum(diameter_results_ctrl)/len(diameter_results_ctrl)


#%% Stat for Diameter


# Collect the diameter values from the dictionaries into lists
diameter_values_1 = list(diameter_results_ctrl.values())
diameter_values_2 = list(diameter_results_Del.values())
diameter_values_3 = list(diameter_results_Del_Scz.values())

# Perform Kruskal-Wallis test
statistic, p_value = kruskal(diameter_values_1, diameter_values_2, diameter_values_3)

# Interpret the results
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis. There are significant differences between groups.")
else:
    print("Fail to reject the null hypothesis. There are no significant differences between groups.")


#%%
plot_max_spanning_tree()

# Plotting violin plots side by side

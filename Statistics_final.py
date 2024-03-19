#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 13:40:14 2024

@author: monir
"""

#%%


def get_values_for_pair(data_dict, index_name, column_name):
    """
    Extracts values from DataFrames in a dictionary for a specific pair of index and column names.
    
    Parameters:
        data_dict (dict): A dictionary containing DataFrames.
        index_name (str): The name of the index.
        column_name (str): The name of the column.
        
    Returns:
        list: A list containing values corresponding to the specified index and column pair.
    """
    list_control = []
    
    for df in data_dict.values():
        if index_name in df.index and column_name in df.columns:
            value = df.loc[index_name, column_name]
            list_control.append(value)
    
    return list_control



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
        
        # Iterate over scalp regions and calculate mean connectivity values
        for region_row, electrodes_row in scalp_regions.items():
            for region_col, electrodes_col in scalp_regions.items():
                # Extract connectivity data for the specified electrodes
                region_conn_values = conn_df.loc[electrodes_row, electrodes_col].values
                
                # Calculate mean connectivity value for the current region
                mean_values = np.nanmean(region_conn_values)
                
                # Store the mean connectivity value in the DataFrame
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





#%%



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



#%%

# Provide the function with desired dataframe
# delta_conn_ctrl_df = pd.DataFrame(conn_ctrl[:,:,0]) #make a df out of a list for example

def connectivity_calculator(connectivity_dataframe, electrode_names=None, scalp_regions=None):
    """
    Calculate mean connectivity values across specified scalp regions.

    Parameters:
    - connectivity_dataframe (pd.DataFrame): DataFrame containing connectivity values between electrodes.
    - electrode_names (list, optional): List of electrode names. Defaults to None.
    - scalp_regions (dict, optional): Dictionary mapping scalp region names to lists of corresponding electrode names.
                                      Defaults to None.

    Returns:
    - pd.DataFrame: DataFrame containing mean connectivity values between scalp regions.

    Example:
    # Assuming connectivity_df is a DataFrame containing connectivity values
    mean_connectivity = connectivity_calculator(connectivity_df)
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
    
    connectivity_dataframe.columns = electrode_names
    connectivity_dataframe.index = electrode_names
    
    mean_df = pd.DataFrame(index=scalp_regions.keys(), columns=scalp_regions.keys())
    
    for region_row, electrodes_row in scalp_regions.items():
        for region_col, electrodes_col in scalp_regions.items():
            mean_values = connectivity_dataframe.loc[electrodes_row, electrodes_col].mean().mean()
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
    
    return mean_df



#%%


def plot_heatmap(df, frequency_band, index):
    """
    Plot a heatmap of connectivity values for a given frequency band.

    Parameters:
    - df (pd.DataFrame): DataFrame containing connectivity values.
    - frequency_band (str): Frequency band for which the heatmap is plotted.
    - index (int): Index of the plot (for labeling purposes).

    Returns:
    None

    Example:
    # Assuming df contains connectivity values for a specific frequency band
    plot_heatmap(df, 'Theta', 1)
    
    """
    plt.figure(figsize=(15, 10))
    mask = np.triu(np.ones_like(df, dtype=bool))
    np.fill_diagonal(mask, False)
    ax = plt.axes()
    ax.set_title(f'Heatmap {index} - {frequency_band}')
    sns.heatmap(df, annot=True, cmap='coolwarm', fmt=".2f", mask=mask, ax=ax, vmin=0.05, vmax=0.45) 

    # Define the filename for the plot
    filename = f'heatmap_{index}_{frequency_band}.png'
    
    # Check if the plot file already exists
    if not os.path.exists(filename):
        plt.savefig(filename)  # Save the plot as an image file
    plt.show()








#%% Imports


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



%matplotlib qt
matplotlib.use('Qt5Agg')
print("mne version", mne.__version__)  # mne version 1.1.0




def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

montage = make_standard_montage('standard_1020')  # montage.plot()

#ord_channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3','C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']

#montage.plot(kind="3d", show=False)
#%% Control Group

pheno = pd.read_csv('/home/monir/Monir/IPNP/Phenotype/phenotype.csv') #phenotype data
pheno_ctrl = pheno[pheno['status']=='ctrl']

input_ctrl_group = '/home/monir/Monir/IPNP/22q11_analysis/Data_controls' #EEG recording
psd_results_ctrl = calculate_psd_results(input_ctrl_group)


# Extract the numeric part of the subject_id column
pheno_ctrl['subject_id'] = pheno_ctrl['subject_id'].str.extract(r'(\d+)')

# Convert the subject_id column to numeric type
pheno_ctrl['subject_id'] = pd.to_numeric(pheno_ctrl['subject_id'])
merged_ctrl = pheno_ctrl.merge(psd_results_ctrl, left_on='subject_id', right_index=True)





#%% connectivity results for each subject


# Initialize dictionaries to store connectivity results for each subject

wpli_results_ctrl = {}  # Dictionary to store connectivity results for each subject
delta_conn_ctrl = {}    # Dictionary to store delta band connectivity for each subject
theta_conn_ctrl = {}  
alpha_conn_ctrl = {}  
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

# Now wpli_results_ctrl is a dictionary where each key is a subject ID, 
    #and the corresponding value is a 3D array containing connectivity results.  ----> print( wpli_results_ctrl)
    # delta_conn_ctrl is a similar dictionary containing only the delta band connectivity for each subject.




#delta_conn_ctrl.keys()
delta_ctrl = connectivity_calculator_per_subject(delta_conn_ctrl)
theta_ctrl = connectivity_calculator_per_subject(theta_conn_ctrl)
alpha_ctrl = connectivity_calculator_per_subject(alpha_conn_ctrl)
beta_ctrl = connectivity_calculator_per_subject(beta_conn_ctrl)
gamma_ctrl = connectivity_calculator_per_subject(gamma_conn_ctrl)
alpha_ctrl

# Example
alpha_ctrl[29] # subject=29 # frequency band=alpha group=control





#%% Deletion Group

pheno = pd.read_csv('/home/monir/Monir/IPNP/Phenotype/phenotype.csv')
pheno_del = pheno[pheno['status']=='del']

input_del_group = '/home/monir/Monir/IPNP/22q11_analysis/Data_del'
psd_results_del = calculate_psd_results(input_del_group)


# Extract the numeric part of the subject_id column
pheno_del['subject_id'] = pheno_del['subject_id'].str.extract(r'(\d+)')

# Convert the subject_id column to numeric type
pheno_del['subject_id'] = pd.to_numeric(pheno_del['subject_id'])
merged_del = pheno_del.merge(psd_results_del, left_on='subject_id', right_index=True)



#%% connectivity result for each subject

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



delta_del = connectivity_calculator_per_subject(delta_conn_del)
theta_del = connectivity_calculator_per_subject(theta_conn_del)
alpha_del = connectivity_calculator_per_subject(alpha_conn_del)
beta_del = connectivity_calculator_per_subject(beta_conn_del)
gamma_del = connectivity_calculator_per_subject(gamma_conn_del)





#%%  Del + Scz Group

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





#%% connectivity result for each subject

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


#%%

delta_del_scz = connectivity_calculator_per_subject(delta_conn_del_scz)
theta_del_scz = connectivity_calculator_per_subject(theta_conn_del_scz)
alpha_del_scz = connectivity_calculator_per_subject(alpha_conn_del_scz)
beta_del_scz = connectivity_calculator_per_subject(beta_conn_del_scz)
gamma_del_scz = connectivity_calculator_per_subject(gamma_conn_del_scz)
# you can have access to each subject's dataframe by printing the keys and then : alpha_del_scz[28] replace 28 by the key you want.
#delta_del_scz.keys()
#delta_del_scz[1]


#%% Alpha Freq band

# List of all possible combinations of indexes and columns

combinations = [
    ('frontal', 'occipital'),
    ('frontal', 'parietal'),
    ('frontal', 'central'),
    ('frontal', 'temporal'),
    ('occipital', 'parietal'),
    ('occipital', 'central'),
    ('occipital', 'temporal'),
    ('parietal', 'central'),
    ('parietal', 'temporal'),
    ('central', 'temporal')
]

# Dictionary to store results
alpha_ctrl_values = {}


# Loop through combinations
for index_name, column_name in combinations:
    key_name = f"{index_name}_{column_name}_alpha_ctrl"
    values = get_values_for_pair(alpha_ctrl, index_name, column_name)
    alpha_ctrl_values[key_name] = values

# Print or use alpha_ctrl_values dictionary as needed
alpha_ctrl_values


# Dictionary to store results
alpha_del_values = {}


# Loop through combinations
for index_name, column_name in combinations:
    key_name = f"{index_name}_{column_name}_alpha_del"
    values = get_values_for_pair(alpha_del, index_name, column_name)
    alpha_del_values[key_name] = values

# Print or use alpha_ctrl_values dictionary as needed
alpha_del_values


# Dictionary to store results
alpha_del_scz_values = {}


# Loop through combinations
for index_name, column_name in combinations:
    key_name = f"{index_name}_{column_name}_alpha_del_scz"
    values = get_values_for_pair(alpha_del_scz, index_name, column_name)
    alpha_del_scz_values[key_name] = values

# Print or use alpha_ctrl_values dictionary as needed
alpha_del_scz_values


# Iterate over combinations
for index_name, column_name in combinations:
    # Construct key names
    alpha_del_key = f"{index_name}_{column_name}_alpha_del"
    alpha_ctrl_key = f"{index_name}_{column_name}_alpha_ctrl"
    alpha_del_scz_key = f"{index_name}_{column_name}_alpha_del_scz"
    
    # Perform Kruskal-Wallis test
    H, p_value = kruskal(
        alpha_del_values[alpha_del_key],
        alpha_ctrl_values[alpha_ctrl_key],
        alpha_del_scz_values[alpha_del_scz_key]
    )
    
    # Output the results
    print(f"\nFor combination {index_name} - {column_name}:")
    print("Kruskal-Wallis H statistic:", H)
    print("p-value:", p_value)

    # Interpret the results
    alpha = 0.05
    if p_value < alpha:
        print("Reject the null hypothesis: There are significant differences between the groups.")
    else:
        print("Fail to reject the null hypothesis: There are no significant differences between the groups.")
        
        
        
#%% Delta Freq band

# List of all possible combinations of indexes and columns

combinations = [
    ('frontal', 'occipital'),
    ('frontal', 'parietal'),
    ('frontal', 'central'),
    ('frontal', 'temporal'),
    ('occipital', 'parietal'),
    ('occipital', 'central'),
    ('occipital', 'temporal'),
    ('parietal', 'central'),
    ('parietal', 'temporal'),
    ('central', 'temporal')
]

# Dictionary to store results
delta_ctrl_values = {}


# Loop through combinations
for index_name, column_name in combinations:
    key_name = f"{index_name}_{column_name}_delta_ctrl"
    values = get_values_for_pair(delta_ctrl, index_name, column_name)
    delta_ctrl_values[key_name] = values

# Print or use alpha_ctrl_values dictionary as needed
delta_ctrl_values



# Dictionary to store results
delta_del_values = {}


# Loop through combinations
for index_name, column_name in combinations:
    key_name = f"{index_name}_{column_name}_delta_del"
    values = get_values_for_pair(delta_del, index_name, column_name)
    delta_del_values[key_name] = values

# Print or use alpha_ctrl_values dictionary as needed
delta_del_values


# Dictionary to store results
delta_del_scz_values = {}


# Loop through combinations
for index_name, column_name in combinations:
    key_name = f"{index_name}_{column_name}_delta_del_scz"
    values = get_values_for_pair(delta_del_scz, index_name, column_name)
    delta_del_scz_values[key_name] = values

# Print or use alpha_ctrl_values dictionary as needed
delta_del_scz_values


# Iterate over combinations
for index_name, column_name in combinations:
    # Construct key names
    delta_del_key = f"{index_name}_{column_name}_delta_del"
    delta_ctrl_key = f"{index_name}_{column_name}_delta_ctrl"
    delta_del_scz_key = f"{index_name}_{column_name}_delta_del_scz"
    
    # Perform Kruskal-Wallis test
    H, p_value = kruskal(
        delta_del_values[delta_del_key],
        delta_ctrl_values[delta_ctrl_key],
        delta_del_scz_values[delta_del_scz_key]
    )
    
    # Output the results
    print(f"\nFor combination {index_name} - {column_name}:")
    print("Kruskal-Wallis H statistic:", H)
    print("p-value:", p_value)

    # Interpret the results
    alpha = 0.05
    if p_value < alpha:
        print("Reject the null hypothesis: There are significant differences between the groups.")
    else:
        print("Fail to reject the null hypothesis: There are no significant differences between the groups.")




#%% Theta Freq band

# List of all possible combinations of indexes and columns

combinations = [
    ('frontal', 'occipital'),
    ('frontal', 'parietal'),
    ('frontal', 'central'),
    ('frontal', 'temporal'),
    ('occipital', 'parietal'),
    ('occipital', 'central'),
    ('occipital', 'temporal'),
    ('parietal', 'central'),
    ('parietal', 'temporal'),
    ('central', 'temporal')
]

# Dictionary to store results
theta_ctrl_values = {}


# Loop through combinations
for index_name, column_name in combinations:
    key_name = f"{index_name}_{column_name}_theta_ctrl"
    values = get_values_for_pair(theta_ctrl, index_name, column_name)
    theta_ctrl_values[key_name] = values

# Print or use alpha_ctrl_values dictionary as needed
theta_ctrl_values



# Dictionary to store results
theta_del_values = {}


# Loop through combinations
for index_name, column_name in combinations:
    key_name = f"{index_name}_{column_name}_theta_del"
    values = get_values_for_pair(beta_del, index_name, column_name)
    theta_del_values[key_name] = values

# Print or use alpha_ctrl_values dictionary as needed
theta_del_values



# Dictionary to store results
theta_del_scz_values = {}


# Loop through combinations
for index_name, column_name in combinations:
    key_name = f"{index_name}_{column_name}_theta_del_scz"
    values = get_values_for_pair(theta_del_scz, index_name, column_name)
    theta_del_scz_values[key_name] = values

# Print or use alpha_ctrl_values dictionary as needed
theta_del_scz_values


# Iterate over combinations
for index_name, column_name in combinations:
    # Construct key names
    theta_del_key = f"{index_name}_{column_name}_theta_del"
    theta_ctrl_key = f"{index_name}_{column_name}_theta_ctrl"
    theta_del_scz_key = f"{index_name}_{column_name}_theta_del_scz"
    
    # Perform Kruskal-Wallis test
    H, p_value = kruskal(
        theta_del_values[theta_del_key],
        theta_ctrl_values[theta_ctrl_key],
        theta_del_scz_values[theta_del_scz_key]
    )
    
    # Output the results
    print(f"\nFor combination {index_name} - {column_name}:")
    print("Kruskal-Wallis H statistic:", H)
    print("p-value:", p_value)

    # Interpret the results
    alpha = 0.05
    if p_value < alpha:
        print("Reject the null hypothesis: There are significant differences between the groups.")
    else:
        print("Fail to reject the null hypothesis: There are no significant differences between the groups.")
        
#%% Beta Freq band



# Dictionary to store results
beta_ctrl_values = {}


# Loop through combinations
for index_name, column_name in combinations:
    key_name = f"{index_name}_{column_name}_beta_ctrl"
    values = get_values_for_pair(beta_ctrl, index_name, column_name)
    beta_ctrl_values[key_name] = values

# Print or use alpha_ctrl_values dictionary as needed
beta_ctrl_values


# Dictionary to store results
beta_del_values = {}


# Loop through combinations
for index_name, column_name in combinations:
    key_name = f"{index_name}_{column_name}_beta_del"
    values = get_values_for_pair(beta_del, index_name, column_name)
    beta_del_values[key_name] = values

# Print or use alpha_ctrl_values dictionary as needed
beta_del_values


# Dictionary to store results
beta_del_scz_values = {}


# Loop through combinations
for index_name, column_name in combinations:
    key_name = f"{index_name}_{column_name}_beta_del_scz"
    values = get_values_for_pair(beta_del_scz, index_name, column_name)
    beta_del_scz_values[key_name] = values

# Print or use alpha_ctrl_values dictionary as needed
beta_del_scz_values


# Iterate over combinations
for index_name, column_name in combinations:
    # Construct key names
    beta_del_key = f"{index_name}_{column_name}_beta_del"
    beta_ctrl_key = f"{index_name}_{column_name}_beta_ctrl"
    beta_del_scz_key = f"{index_name}_{column_name}_beta_del_scz"
    
    # Perform Kruskal-Wallis test
    H, p_value = kruskal(
        beta_del_values[beta_del_key],
        beta_ctrl_values[beta_ctrl_key],
        beta_del_scz_values[beta_del_scz_key]
    )
    
    # Output the results
    print(f"\nFor combination {index_name} - {column_name}:")
    print("Kruskal-Wallis H statistic:", H)
    print("p-value:", p_value)

    # Interpret the results
    alpha = 0.05
    if p_value < alpha:
        print("Reject the null hypothesis: There are significant differences between the groups.")
    else:
        print("Fail to reject the null hypothesis: There are no significant differences between the groups.")
        
        
        


#%% Gamma Freq band


# List of all possible combinations of indexes and columns

combinations = [
    ('frontal', 'occipital'),
    ('frontal', 'parietal'),
    ('frontal', 'central'),
    ('frontal', 'temporal'),
    ('occipital', 'parietal'),
    ('occipital', 'central'),
    ('occipital', 'temporal'),
    ('parietal', 'central'),
    ('parietal', 'temporal'),
    ('central', 'temporal')
]

# Dictionary to store results
gamma_ctrl_values = {}


# Loop through combinations
for index_name, column_name in combinations:
    key_name = f"{index_name}_{column_name}_gamma_ctrl"
    values = get_values_for_pair(gamma_ctrl, index_name, column_name)
    gamma_ctrl_values[key_name] = values

# Print or use alpha_ctrl_values dictionary as needed
gamma_ctrl_values




# Dictionary to store results
gamma_del_values = {}


# Loop through combinations
for index_name, column_name in combinations:
    key_name = f"{index_name}_{column_name}_gamma_del"
    values = get_values_for_pair(gamma_del, index_name, column_name)
    gamma_del_values[key_name] = values

# Print or use alpha_ctrl_values dictionary as needed
gamma_del_values



# Dictionary to store results
gamma_del_scz_values = {}


# Loop through combinations
for index_name, column_name in combinations:
    key_name = f"{index_name}_{column_name}_gamma_del_scz"
    values = get_values_for_pair(gamma_del_scz, index_name, column_name)
    gamma_del_scz_values[key_name] = values

# Print or use alpha_ctrl_values dictionary as needed
gamma_del_scz_values


# Iterate over combinations
for index_name, column_name in combinations:
    # Construct key names
    gamma_del_key = f"{index_name}_{column_name}_gamma_del"
    gamma_ctrl_key = f"{index_name}_{column_name}_gamma_ctrl"
    gamma_del_scz_key = f"{index_name}_{column_name}_gamma_del_scz"
    
    # Perform Kruskal-Wallis test
    H, p_value = kruskal(
        gamma_del_values[gamma_del_key],
        gamma_ctrl_values[gamma_ctrl_key],
        gamma_del_scz_values[gamma_del_scz_key]
    )
    
    # Output the results
    print(f"\nFor combination {index_name} - {column_name}:")
    print("Kruskal-Wallis H statistic:", H)
    print("p-value:", p_value)

    # Interpret the results
    alpha = 0.05
    if p_value < alpha:
        print("Reject the null hypothesis: There are significant differences between the groups.")
    else:
        print("Fail to reject the null hypothesis: There are no significant differences between the groups.")



# Significant difference : For combination frontal - occipital, For combination frontal - central, For combination frontal - temporal





#%%

















#%%













#%%














#%%











#%%













#%%



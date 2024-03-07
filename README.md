EEG Analysis and Connectivity Calculation

This repository contains code for analyzing EEG data, calculating power spectral density (PSD), computing connectivity between electrodes, and visualizing the results. The code is written in Python and utilizes various libraries such as MNE, NumPy, Pandas, and Matplotlib.
Table of Contents

    Introduction
    Requirements
    Usage
    Functions
    Examples
    Contributing
    License

Introduction

Electroencephalography (EEG) is a non-invasive technique used to record electrical activity in the brain. This project aims to analyze EEG data and extract useful information regarding power spectral density and connectivity between different brain regions.
Requirements

To run the code in this repository, you need to have the following dependencies installed:

    Python 3.x
    MNE (>= 0.23.0)
    NumPy
    Pandas
    Matplotlib
    Seaborn

You can install the required dependencies using pip:

bash

pip install mne numpy pandas matplotlib seaborn

Usage

    Clone the repository to your local machine:

bash

git clone https://github.com/your-username/eeg-analysis.git

    Navigate to the project directory:

bash

cd eeg-analysis

    Run the Python scripts to perform EEG analysis, PSD calculation, connectivity computation, and visualization.

Functions
calculate_psd_results(folder_path)

This function calculates the power spectral density (PSD) of EEG signals stored in the specified folder path. It returns a Pandas DataFrame containing the PSD results.
connectivity_calculator(connectivity_dataframe, electrode_names=None, scalp_regions=None)

Calculates mean connectivity values across specified scalp regions using connectivity data stored in the provided DataFrame. It returns a DataFrame containing the mean connectivity values between scalp regions.
plot_heatmap(df, frequency_band, index)

Plots a heatmap of connectivity values for a given frequency band. The function takes a DataFrame containing connectivity values, the frequency band name, and the index of the plot.
Examples

python

# Example usage of calculate_psd_results
psd_results = calculate_psd_results('/path/to/eeg/data')

# Example usage of connectivity_calculator
mean_connectivity = connectivity_calculator(connectivity_df)

# Example usage of plot_heatmap
plot_heatmap(connectivity_df, 'Theta', 1)

Contributing

Contributions to this project are welcome. You can contribute by opening issues, suggesting improvements, or submitting pull requests.

License

This project is licensed under the MIT License - see the LICENSE file for details.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 19:38:38 2024

@author: davidmikhael
"""

"""
The dataset used here is from OpenNeuro. 

'Tommaso Fedele and Ece Boran and Valeri Chirkov and Peter Hilfiker and
Thomas Grunwald and Lennart Stieglitz and Hennric Jokeit and
Johannes Sarnthein (2020). Dataset of neurons and intracranial EEG from human
amygdala during aversive dynamic visual stimulation. OpenNeuro.
[Dataset] doi: 10.18112/openneuro.ds003374.v1.1.1'

It is an ECoG dataset collected from the amygdalae of nine subjects attending
a visual dynamic stimulation of emotional aversive vs neutral content. We are
interested in comparing fitting the data to an LSTM (to capture temporal
dependencies in the timeseries) and comparing the model's activations in
response to aversive vs neutral stimuli. Mainly, we are interesting in adding
a sparsity constraint and looking at the sparsity level for each stimulus type
"""
#%%
!pip install mne pyedflib torch numpy pandas

import pyedflib
import numpy as np
import mne
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind, mannwhitneyu, shapiro, levene
from mne.time_frequency import tfr_morlet

#%% Loads data

# Load EDF file
edf_file = 'sub-01/ses-01/ieeg/sub-01_ses-01_task-jokeit_run-01_ieeg.edf'
f = pyedflib.EdfReader(edf_file)

# Extract information
n_channels = f.signals_in_file
signal_labels = f.getSignalLabels()
sampling_frequency = f.getSampleFrequency(0) # Same frequency
# Extract data
signals = np.zeros((n_channels, f.getNSamples()[0]))
for i in range(n_channels):
    signals[i, :] = f.readSignal(i)

f.close()

print("Number of Channels:", n_channels)
print("Signal Labels:", signal_labels)
print("Sampling Frequency:", sampling_frequency)

#%%
# Load BrainVision file
vhdr_file = 'sub-01/ses-01/ieeg/sub-01_ses-01_task-jokeit_run-01_ieeg.vhdr'
raw = mne.io.read_raw_brainvision(vhdr_file, preload=True)

# Access data
data, times = raw[:]

print("Data Shape:", data.shape)  # (n_channels, n_times)
print("Time Shape:", times.shape)
print("Sampling Frequency:", raw.info['sfreq'])
print("Channel Names:", raw.info['ch_names'])

#%%

# Load event metadata
events_tsv = 'sub-01/ses-01/ieeg/sub-01_ses-01_task-jokeit_run-01_events.tsv'
events_df = pd.read_csv(events_tsv, sep='\t')

print(events_df.head())

#%%

# Extract the sampling frequency from the raw data
sfreq = raw.info['sfreq']

# Map trial types to integer codes
event_id_mapping = {'Aversive': 1, 'Neutral': 2}  # Example mapping

# Convert events to MNE-compatible format
events = []
for _, row in events_df.iterrows():
    onset_sample = int(row['onset'] * sfreq)  # Convert seconds to sample index
    event_type = event_id_mapping.get(row['trial_type'], 0) 
    events.append([onset_sample, 0, event_type])

events = np.array(events)

#%%

# Add events to MNE object
event_dict = {'aversive': 1, 'neutral': 2}  
epochs = mne.Epochs(raw, events, event_id=event_dict, tmin=-0.2, tmax=0.8, baseline=(None, 0), preload=True)

#%%

# Plot epochs for visual inspection
epochs.plot()

aversive_evoked = epochs['aversive'].average()
neutral_evoked = epochs['neutral'].average()

# Plot the evoked response
aversive_evoked.plot()
neutral_evoked.plot()

aversive_data = epochs['aversive'].get_data()  # Shape: (n_aversive_epochs, n_channels, n_times)
neutral_data = epochs['neutral'].get_data()    # Shape: (n_neutral_epochs, n_channels, n_times)

data_reshaped = aversive_data.reshape(aversive_data.shape[0], -1)  # Flatten channels and timepoints
scaler = StandardScaler().fit(data_reshaped)
data_normalized = scaler.transform(data_reshaped)
aversive_data = data_normalized.reshape(aversive_data.shape)  # Reshape back to original shape

data_reshaped = neutral_data.reshape(neutral_data.shape[0], -1)  # Flatten channels and timepoints
scaler = StandardScaler().fit(data_reshaped)
data_normalized = scaler.transform(data_reshaped)
neutral_data = data_normalized.reshape(neutral_data.shape)  # Reshape back to original shape


#%%

# Create labels: 1 for aversive, 0 for neutral
aversive_labels = np.ones((aversive_data.shape[0],), dtype=int)
neutral_labels = np.zeros((neutral_data.shape[0],), dtype=int)

# Concatenate data and labels
data = np.concatenate([aversive_data, neutral_data], axis=0)
labels = np.concatenate([aversive_labels, neutral_labels], axis=0)

# Reshape to 2D for normalization, then reshape back
data_reshaped = data.reshape(data.shape[0], -1)  # Flatten channels and timepoints
scaler = StandardScaler().fit(data_reshaped)
data_normalized = scaler.transform(data_reshaped)
data = data_normalized.reshape(data.shape)  # Reshape back to original shape


# Convert all data to PyTorch tensors
data_tensor = torch.tensor(data, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.long)
# Create DataLoaders
batch_size = 16
dataset = TensorDataset(data_tensor, labels_tensor)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Convert aversive and neutral data to PyTorch tensors
# Convert to PyTorch tensors
aversive_tensor = torch.tensor(aversive_data, dtype=torch.float32)
neutral_tensor = torch.tensor(neutral_data, dtype=torch.float32)
avlabels_tensor = torch.tensor(aversive_labels, dtype=torch.long)
ntlabels_tensor = torch.tensor(neutral_labels, dtype=torch.long)
# Create DataLoaders
batch_size = 16
avdataset = TensorDataset(aversive_tensor, avlabels_tensor)
ntdataset = TensorDataset(neutral_tensor, ntlabels_tensor)
aversive_loader = DataLoader(avdataset, batch_size=batch_size, shuffle=True)
neutral_loader = DataLoader(ntdataset, batch_size=batch_size, shuffle=True)


#%% Creates the model

class SparseLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, sparsity_lambda=0.1):
        super(SparseLSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sparsity_lambda = sparsity_lambda

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Batch Normalization (conditionally applied)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        
        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM forward pass
        out, (hn, cn) = self.lstm(x, (h0, c0))
        l1_penalty = self.sparsity_lambda * torch.sum(torch.abs(out))

        # Apply Batch Normalization only if batch size > 1
        if out.size(0) > 1:
            # Permute out to (batch_size, hidden_size) for BatchNorm1d
            out_last_step = out[:, -1, :]  # Take the last time step
            out_last_step = self.batch_norm(out_last_step)
        else:
            out_last_step = out[:, -1, :]  # Skip batch norm if batch size is 1

        # Pass through the fully connected layer
        out = self.fc(out_last_step)

        return out, l1_penalty


#%% Training

input_size = data.shape[2]  # Number of channels
hidden_size = 32            # Hidden state size of LSTM
num_layers = 1              # Number of LSTM layers
output_size = 2             # Number of classes (aversive, neutral)
sparsity_lambda = 0.1       # Sparsity regularization strength
num_epochs = 100             # Number of training epochs
learning_rate = 0.0001      # Learning rate

# Instantiate model, loss function, and optimizer
model = SparseLSTMClassifier(input_size, hidden_size, num_layers, output_size, sparsity_lambda)
criterion = nn.CrossEntropyLoss()  # For classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_l1_penalty = 0
    correct = 0
    total = 0
    
    for inputs, labels in data_loader:
        inputs = inputs.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        labels = labels.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Forward pass
        outputs, l1_penalty = model(inputs)
        loss = criterion(outputs, labels) + l1_penalty
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update loss and L1 penalty
        total_loss += loss.item()
        total_l1_penalty += l1_penalty.item()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    # Print epoch metrics
    avg_loss = total_loss / len(data_loader)
    avg_l1_penalty = total_l1_penalty / len(data_loader)
    accuracy = 100 * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, L1 Penalty: {avg_l1_penalty:.4f}, Accuracy: {accuracy:.2f}%")

#%% Compares sparsity in activations for neutral vs aversive stimuli responses

def calculate_sparsity(data_loader, model):
    model.eval()
    sparsity_scores = []
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            
            # Get hidden activations from the LSTM
            out, (hn, cn) = model.lstm(inputs)
            
            # Calculate sparsity as the fraction of zero (or near-zero) activations
            sparsity_ratio = (torch.abs(out) < 1e-4).float().mean().item()
            sparsity_scores.append(sparsity_ratio)
    
    avg_sparsity = np.mean(sparsity_scores)
    print(f"Average Sparsity: {avg_sparsity:.4f}")
    return avg_sparsity


# Calculate sparsity for both conditions
print("Sparsity for aversive data:")
sparsity_aversive = calculate_sparsity(aversive_loader, model)
print("Sparsity for neutral data:")
sparsity_neutral = calculate_sparsity(neutral_loader, model)

#%% Another code for sparsity calculation with plotting and stats for quantification

def get_lstm_activations(data_loader, model):
    model.eval()
    all_activations = []
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            out, _ = model.lstm(inputs)
            activations = out.cpu().numpy()
            all_activations.append(activations)
    all_activations = np.concatenate(all_activations, axis=0)
    return all_activations

# Get activations for aversive and neutral data
activations_aversive = get_lstm_activations(aversive_loader, model)
activations_neutral = get_lstm_activations(neutral_loader, model)

def calculate_sparsity(activations, threshold=1e-4):
    # activations shape: (samples, time_steps, hidden_size)
    num_elements = activations.size
    num_zero_elements = np.sum(np.abs(activations) < threshold)
    sparsity_ratio = num_zero_elements / num_elements
    return sparsity_ratio

# Calculate sparsity for each sample
sparsity_aversive = []
for activation in activations_aversive:
    sparsity = calculate_sparsity(activation)
    sparsity_aversive.append(sparsity)

sparsity_neutral = []
for activation in activations_neutral:
    sparsity = calculate_sparsity(activation)
    sparsity_neutral.append(sparsity)


plt.figure(figsize=(10,6))
plt.hist(sparsity_aversive, bins=30, alpha=0.5, label='Aversive')
plt.hist(sparsity_neutral, bins=30, alpha=0.5, label='Neutral')
plt.title('Sparsity Distribution')
plt.xlabel('Sparsity Ratio')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# Check for normality
stat, p_aversive = shapiro(sparsity_aversive)
stat, p_neutral = shapiro(sparsity_neutral)
print(f"P-value for normality (Aversive): {p_aversive:.4f}")
print(f"P-value for normality (Neutral): {p_neutral:.4f}")

# If p > 0.05, data is normally distributed
# Check for equal variances
stat, p_levene = levene(sparsity_aversive, sparsity_neutral)
print(f"P-value for equal variances: {p_levene:.4f}")

# Perform t-test or Mann-Whitney U test based on normality
if p_aversive > 0.05 and p_neutral > 0.05:
    # Data is normally distributed
    if p_levene > 0.05:
        # Equal variances
        stat, p_value = ttest_ind(sparsity_aversive, sparsity_neutral)
    else:
        # Unequal variances
        stat, p_value = ttest_ind(sparsity_aversive, sparsity_neutral, equal_var=False)
else:
    # Non-parametric test
    stat, p_value = mannwhitneyu(sparsity_aversive, sparsity_neutral)
    
print(f"Test Statistic: {stat:.4f}, P-value: {p_value:.4f}")


# Cohen's d for effect size
def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1)*np.std(x, ddof=1)**2 + (ny - 1)*np.std(y, ddof=1)**2) / dof)
    return (np.mean(x) - np.mean(y)) / pooled_std

effect_size = cohen_d(sparsity_aversive, sparsity_neutral)
print(f"Cohen's d effect size: {effect_size:.4f}")

#%% Time-frequency analysis for aversive vs neutral responses
# 
# We'll separate epochs by condition
epochs_aversive = epochs['aversive']
epochs_neutral = epochs['neutral']


# Define frequencies of interest
freqs = np.logspace(*np.log10([1, 100]), num=70)  # From 1 to 100 Hz
n_cycles = freqs / 2.  # Different number of cycles per frequency

# Perform time-frequency decomposition
power_aversive = tfr_morlet(epochs_aversive, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                            return_itc=False, decim=3, n_jobs=1)

power_neutral = tfr_morlet(epochs_neutral, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                           return_itc=False, decim=3, n_jobs=1)

baseline = (-0.2, 0)  # Pre-stimulus period based on when stimulus is presented
power_aversive.apply_baseline(baseline=baseline, mode='logratio')
power_neutral.apply_baseline(baseline=baseline, mode='logratio')

# Plot for aversive condition
power_aversive.plot([0], baseline=baseline, title='Aversive Condition')

# Plot for neutral condition
power_neutral.plot([0], baseline=baseline, title='Neutral Condition')

# Compute difference between conditions
power_diff = power_aversive.copy()
power_diff.data = power_aversive.data - power_neutral.data

# Plot the difference
power_diff.plot(title='Aversive - Neutral')

#%%

"""
I need to add more data and separate into training and testing sets. But
as a preliminary finding, aversive stimuli activations in the amygdala as
respresented by an LSTM as more sparse than neutral stimuli.

"""
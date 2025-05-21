#!/usr/bin/env python
# coding: utf-8

# Importing packages

import pandas as pd                                         # type: ignore 
import numpy as np                                          # type: ignore


parent_path = '/gws/nopw/j04/wiser_ewsa/mrakotomanga/EPS/Data/Senegal/64x64/full'


location = 'senegal'
# Test data
test_data_X = pd.read_csv(f'{parent_path}/test-data-{location}-input-t0.csv')
test_data_y_lt0 = pd.read_csv(f'{parent_path}/test-data-{location}-output-map-t0.csv', header=None)
test_data_y_lt1 = pd.read_csv(f'{parent_path}/test-data-{location}-output-map-t1.csv', header=None)


# Train data
train_data_X = pd.read_csv(f'{parent_path}/train-data-{location}-input-t0.csv')
train_data_y_lt0 = pd.read_csv(f'{parent_path}/train-data-{location}-output-map-t0.csv', header=None)
train_data_y_lt1 = pd.read_csv(f'{parent_path}/train-data-{location}-output-map-t1.csv', header=None)


# Choosing lead time
lead_time = 6

# Choosing data split

dataset = "train"
if dataset == "train":
    data = train_data_X
    target = train_data_y_lt1
    target_t0 = train_data_y_lt0
else:
    data = test_data_X 
    target = test_data_y_lt1
    target_t0 = test_data_y_lt0


original_data = data.copy()

data['datetime'] = pd.to_datetime(data[['year', 'month', 'day', 'hour', 'minute']])

def find_exact_row_after_given_hours(row, hours, minutes, df):
    target_time = row['datetime'] + pd.Timedelta(hours=hours, minutes=minutes)
    corresponding_row = df[df['datetime'] == target_time]
    if not corresponding_row.empty:
        return corresponding_row.index[0]  # Return the index of the corresponding row
    else:
        return None

data['row_index_X0_30'] = data.apply(find_exact_row_after_given_hours, args=(0, 30, data), axis=1)
data['row_index_X0_60'] = data.apply(find_exact_row_after_given_hours, args=(1, 0, data), axis=1)
data['row_index_X0_90'] = data.apply(find_exact_row_after_given_hours, args=(1, 30, data), axis=1)
data['row_index_X0_120'] = data.apply(find_exact_row_after_given_hours, args=(2, 0, data), axis=1)

data['row_index_Cb'] = data.apply(find_exact_row_after_given_hours, args=(lead_time + 1 , 0, data), axis=1)      # since the target is at t0+1 h

columns_to_keep = original_data.keys().to_list()

# Function to combine current row with rows based on indices, retaining original column order
def combine_current_and_rows(row, data):
    # Extract the row indices from the current row
    indices = [row['row_index_X0_30'], row['row_index_X0_60'], row['row_index_X0_90'], row['row_index_X0_120']]
    
    # Fetch the current row (filter only relevant columns)
    current_row = row[columns_to_keep].copy()
    
    # List to store the rows
    rows_to_combine = [current_row]
    
    # Fetch rows corresponding to the indices, rename columns with suffix to avoid duplicates
    for i, idx in enumerate(indices):
        if pd.notna(idx):
            # Fetch the row, keep only relevant columns, and rename them with suffix
            fetched_row = data.loc[idx, columns_to_keep].rename(lambda col: f"{col}_{(i+1)*30}")
            rows_to_combine.append(fetched_row)
        else:
            # If the index is NaN, create an empty Series with the same columns as the current row
            empty_row = pd.Series(index=[f"{col}_{(i+1)*30}" for col in columns_to_keep])
            rows_to_combine.append(empty_row)
    
    # Concatenate the current row with the fetched rows side by side (maintain column order)
    combined_row = pd.concat(rows_to_combine, axis=0)
    
    return combined_row
# Apply the function row by row to get combined data
combined_data = data.apply(combine_current_and_rows, args=(data,), axis=1)

# Convert the combined series into a DataFrame while keeping the original index
combined_df = pd.DataFrame(combined_data, index=data.index)

combined_df = pd.DataFrame(combined_data, index=data.index)
combined_df = combined_df.dropna()

target_index = data['row_index_Cb'][combined_df.index].dropna()
target = target.loc[target_index]
target_t0 = target_t0.loc[target_index]
combined_df = combined_df.loc[target_index.index]

target = target.to_numpy()
target_t0 = target_t0.to_numpy()

target.shape

resolution_y, resolution_x = 64, 64
target = target.reshape(len(target), resolution_y, resolution_x)
target_t0 = target_t0.reshape(len(target_t0), resolution_y, resolution_x)


output_dir = '/gws/nopw/j04/wiser_ewsa/mrakotomanga/EPS/Data/Senegal/64x64/full/timesteps'
combined_df.to_csv(f'{output_dir}/{dataset}-senegal-input-t0-for-lt{lead_time}.csv', index=False)

np.save(f'{output_dir}/{dataset}-senegal-output-at-lt{lead_time}.npy', target)
np.save(f'{output_dir}/{dataset}-senegal-output-t0-for-lt{lead_time}.npy', target_t0)
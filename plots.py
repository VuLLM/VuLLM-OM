import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from collections import Counter

# Ensure the directory exists
save_directory = "Charts"
# os.makedirs(save_directory, exist_ok=True)

# # Data
# models = [
#     "VulGen", 
#     "Shorter than 5", 
#     "Shorter than 10", 
#     "Shorter than 10 only CWE", 
#     "Shorter than 20 only CWE", 
#     "Shorter than 30"
# ]
# accuracy = [39.63, 50, 40.37, 39.26, 38.4, 33.3]
# train_set_size = [7426, 6272, 12547, 4895, 7964, 23084]
# test_set_size = [775, 262, 540, 540, 888, 1024]
# generation_length = [59.51, 33.4, 43.88, 47.62, 59.51, 71.72]
# cwe_coverage = [29, 40, 51, 53, 54, 57]  # New data

# # Create a DataFrame
# data = pd.DataFrame({
#     "Model": models,
#     "Accuracy (%)": accuracy,
#     "Train Set Size": train_set_size,
#     "Test Set Size": test_set_size,
#     "Generation Length": generation_length,
#     "CWE Coverage": cwe_coverage  # Adding the new column
# })

# # Colors
# colors = plt.cm.viridis(np.linspace(0, 1, len(models)))

# # Plotting
# def plot_data(x, y, title, ylabel, filename):
#     fig, ax = plt.subplots(figsize=(10, 5))
#     bars = ax.bar(x, y, color=colors)
#     ax.set_xlabel("Models")
#     ax.set_ylabel(ylabel)
#     ax.set_title(title)
#     ax.set_xticklabels(x, rotation=45, ha="right")

#     # Adding numbers above bars
#     for bar in bars:
#         yval = bar.get_height()
#         ax.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), ha='center', va='bottom')

#     plt.tight_layout()
#     plt.savefig(os.path.join(save_directory, filename))
#     plt.close()

# # Create and save plots
# plot_data(data['Model'], data['Accuracy (%)'], 'Model Accuracy', 'Accuracy (%)', 'accuracy_chart.png')
# plot_data(data['Model'], data['Train Set Size'], 'Training Set Size by Model', 'Train Set Size', 'train_set_size_chart.png')
# plot_data(data['Model'], data['Test Set Size'], 'Test Set Size by Model', 'Test Set Size', 'test_set_size_chart.png')
# plot_data(data['Model'], data['Generation Length'], 'Generation Length by Model', 'Generation Length', 'generation_length_chart.png')
# plot_data(data['Model'], data['CWE Coverage'], 'CWE Coverage by Model', 'CWE Coverage', 'cwe_coverage_chart.png')

# print("Charts have been created and saved successfully in the directory 'Charts/'.")

# Read the pickle file
with open('cwes.pickle', 'rb') as file:
    counter_data = pickle.load(file)

# Sort the items by keys
counter_data = Counter(counter_data)
items = sorted(counter_data.items())
keys = [str(k) for k, v in items]
values = [v for k, v in items]

# Positions for the bars on the x-axis
positions = range(len(keys))

# Create the plot
plt.figure(figsize=(12, 6))
plt.bar(positions, values)

# Label the x-axis with the keys
plt.xticks(positions, keys, rotation='vertical')

# Set labels and title
plt.xlabel('CWEs')
plt.ylabel('Frequency')
plt.title('Histogram of CWEs Coverage')

# Annotate each bar with its value
for pos, y in zip(positions, values):
    plt.text(pos, y + 0.5, str(y), ha='center', va='bottom')

# Adjust layout to prevent clipping of tick-labels
plt.tight_layout()

# Display the plot
plt.savefig('cwe_hist_TM.png')

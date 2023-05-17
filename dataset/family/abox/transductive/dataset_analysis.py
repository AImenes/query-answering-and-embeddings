import pandas as pd
import os

df = pd.read_csv("all.txt", delimiter='\t', header=None)

# Concatenate the 0th and 2nd columns of the dataframe
df_concatenated = pd.concat([df.iloc[:,0], df.iloc[:,2]])

# Count the frequency of the entities
entity_frequencies = df_concatenated.value_counts()

# Write the results to a file
entity_frequencies.to_csv("freq_count_entities.txt")

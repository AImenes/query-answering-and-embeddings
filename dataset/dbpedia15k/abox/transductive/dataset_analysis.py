import pandas as pd

df = pd.read_csv("all.txt", delimiter='\t', header=None)

# Filter the dataframe where the predicate (column 1) is equal to the rdf:type property
df_filtered = df[df.iloc[:,1] == "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"]

# Count the frequency of the entities in the 3rd column (column 2) of the filtered dataframe
entity_frequencies = df_filtered.iloc[:,2].value_counts()

# Write the results to a file
entity_frequencies.to_csv("freq_count_concepts.txt", header=False)

import pandas as pd
import os

df = pd.read_csv("all.txt", delimiter='\t', header=None)
print("stop")

#Using pandas way, Series.value_counts()
df1 = df.iloc[:,1].value_counts()
df1.to_csv("freq_count.txt")


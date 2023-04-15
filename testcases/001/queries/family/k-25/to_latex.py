import pandas as pd
import json

# Read the JSON data
with open("testcases/001/queries/family/k-25/family-BoxE-dim192-epoch24-k100-numbofqueries:25-1p-final_results.json", "r") as file:
    data = json.load(file)

# Convert the JSON data to a pandas DataFrame
df = pd.DataFrame(data)

# Generate the LaTeX table from the DataFrame
latex_table = df.to_latex(index=False)

# Save the LaTeX table to a file
with open("testcases/001/queries/family/k-25/family-BoxE-dim192-epoch24-k100-numbofqueries:25-1p-final_results.tex", "w") as file:
    file.write(latex_table)

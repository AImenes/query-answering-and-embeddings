import os
import pickle

base_cases_path = "testcases/001/base_cases.pickle"

if os.path.exists(base_cases_path):
    with open(base_cases_path, 'rb') as file:
            base_cases = pickle.load(file)
else:
    base_cases = list()

print("stop")
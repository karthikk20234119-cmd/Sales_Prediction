import sys
import subprocess
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

notebook_path = r"c:\Projects\OASIS INFOBYTE Internship\sales_prediction\sales_prediction.ipynb"

print("Starting Notebook Execution ...")
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

try:
    ep.preprocess(nb, {'metadata': {'path': r"c:\Projects\OASIS INFOBYTE Internship\sales_prediction"}})
    print("Notebook executed successfully!")
except Exception as e:
    print(f"Error executing notebook: {e}")
    sys.exit(1)

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print(f"Executed notebook saved to {notebook_path}")

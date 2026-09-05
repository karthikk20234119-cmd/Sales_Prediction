import os
import urllib.request
import pandas as pd

# Directories
base_dir = r"c:\Projects\OASIS INFOBYTE Internship\sales_prediction"
data_dir = os.path.join(base_dir, "data")
outputs_dir = os.path.join(base_dir, "outputs")

os.makedirs(data_dir, exist_ok=True)
os.makedirs(outputs_dir, exist_ok=True)

csv_url = "https://raw.githubusercontent.com/justmarkham/scikit-learn-videos/master/data/Advertising.csv"
csv_path = os.path.join(data_dir, "Advertising.csv")

print(f"Downloading Advertising dataset from {csv_url} ...")
urllib.request.urlretrieve(csv_url, csv_path)

if os.path.exists(csv_path):
    print("Download successful!")
    df = pd.read_csv(csv_path)
    print("Shape:", df.shape)
    print("Columns:", list(df.columns))
    print("Head:\n", df.head())
else:
    print("Download failed!")

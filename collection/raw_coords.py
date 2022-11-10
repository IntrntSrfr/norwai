import pandas as pd
import requests
import matplotlib.pyplot as plt
from tqdm import trange

URL = 'https://randomstreetview.com/data'

def get_locs(n: int) -> list:
  all_locs = []
  for _ in trange(n):
    r = requests.post(URL, data={'country':'no'}, files=[])
    all_locs.extend(r.json()['locations'])
  return all_locs

df = pd.DataFrame(get_locs(600))
df = df.drop('formatted_address', axis=1)
print("initial length: ", len(df))
df = df.drop_duplicates()
print("length after removing dupes: ", len(df))

df['lat'] = pd.to_numeric(df['lat'])
df['lng'] = pd.to_numeric(df['lng'])

df.to_json('data/raw_coords.json', orient='records')
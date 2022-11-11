import requests
import json
import pandas as pd
from PIL import Image
from tqdm import tqdm
from config import MAPS_API_KEY

with open('./data/deg_clean_coords.json', 'rb') as f:
  data = json.load(f)

complete_data = []

for i, r in enumerate(tqdm(data)):
  if 'angle' not in r or r['county_name'] == 'oslo':
    continue
  if r['county'] > 1:
    r['county'] -= 1
  l = [(r['angle'] - 90)%360, (r['angle'] + 90)%360]
  for j, h in enumerate(l):
    url = "https://maps.googleapis.com/maps/api/streetview?size=615x640&location={},{}&fov=90&pitch=15&heading={}&key={}".format(r['lat'], r['lng'], h, MAPS_API_KEY)
    r_c = r.copy()
    try:
      img = Image.open(requests.get(url, stream=True).raw)
      img = img.crop((0, 0, 615, 615))
      img.save('./data/images/{:05d}_{}.jpg'.format(i+1, j))
      r_c['filename'] = '{:05d}_{}.jpg'.format(i+1, j)
      r_c['angle'] = h
      complete_data.append(r_c)
    except:
      continue

df = pd.DataFrame(complete_data)
df.to_csv('./data/data.csv', index=False)
import json
import pandas as pd
from geopandas import GeoDataFrame
from shapely.geometry import Point

# read geojson data
with open('./data/norge.geojson', 'rb') as f:
  geo_data = json.load(f)

# convert projection
gdf = GeoDataFrame.from_features(geo_data["administrative_enheter.fylke"]["features"], crs=25833)
gdf = gdf.to_crs(4326)

counties = gdf[['fylkesnummer', 'navn', 'geometry']].to_dict(orient='records')
mapping = {v:i for i,v in enumerate(gdf['fylkesnummer'])}

def find_county(row):
  for county in counties:
    if county['geometry'].contains(Point(row['lng'], row['lat'])):
      return mapping[county['fylkesnummer']], county['navn'][0]['navn'].lower()

cdf = pd.read_json('./data/raw_coords.json')
cdf[['county', 'county_name']] = cdf.apply(lambda x: find_county(x), axis=1, result_type='expand')
cdf = cdf.dropna()
cdf.to_json('./data/clean_coords.json', orient='records')
cdf.to_csv('./data/clean_coords.csv', index=False)
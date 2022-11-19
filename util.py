import os

def make_data_folders():
  os.makedirs('./data', exist_ok=True)
  os.makedirs('./data/trained_models/county/metrics', exist_ok=True)
  os.makedirs('./data/trained_models/distance/metrics', exist_ok=True)
  os.makedirs('./data/trained_models/zone/metrics', exist_ok=True)

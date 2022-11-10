import pandas as pd
import os
import pathlib
import shutil
import numpy as np

path = pathlib.Path('./data/images')
files = [x.name for x in path.iterdir() if x.is_file()]

df = pd.read_csv('./data/locations_county.csv', index_col=3)

idx_to_county = {0:'agder', 1:'oslo', 2:'troms-og-finnmark', 3:'more-og-romsdal',4:'vestfold-og-telemark',5:'trondelag', 6:'rogaland',7:'innlandet',8:'viken',9:'nordland',10:'vestland'}

ye = {x:[] for x in idx_to_county}
print(ye)
for f in files:
  try:
    county = int(df.loc[f]['fylkesnummer'])
  except:
    continue
  ye[county].append(f)

full_len = np.array([len(ye[e]) for e in ye])
train_len = np.array([int(len(ye[e])*0.85) for e in ye])
test_len = full_len-train_len

print(full_len)
print(train_len)
print(test_len)

#print(ye)

train_set = {i:ye[i][test_len[i]:] for i in ye}
test_set = {i:ye[i][:test_len[i]] for i in ye}

print([len(train_set[e]) for e in train_set])
print([len(test_set[e]) for e in test_set])

print(sum([len(train_set[e]) for e in train_set]))
print(sum([len(test_set[e]) for e in test_set]))


''' 
for county in test_set:
  for im in test_set[county]:
    src = pathlib.Path('./data/images') / im
    dst = pathlib.Path('./data/NSVD/test') / idx_to_county[county] / im
    shutil.copy(src, dst)
 '''
#print('----------------------------------')
#print(test_set)

''' 
for i in idx_to_county.values():
  os.makedirs('./data/NSVD/train/'+i)
  os.makedirs('./data/NSVD/test/'+i)
 '''
''' 
FYLKESNUMMER_INDEX = [42, 3, 54, 15, 38, 50, 11, 34, 30, 18, 46]
42 - Agder
03 - Oslo
54 - Troms og Finnmark
15 - Møre og Romsdal
38 - Vestfold og Telemark
50 - Trøndelag
11 - Rogaland
34 - Innlandet
30 - Viken
18 - Nordland
46 - Vestland
 '''
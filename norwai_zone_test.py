import pandas as pd
import torch
from pyproj import Proj, transform
from torch import nn
import numpy as np
import geopy.distance

def weighted_centroid(points, weights):
    res = points
    while len(res) > 1:
        new_res = []
        new_weights = []
        for i in range(0,len(res),2):
            if i+1 == len(res):
              continue
            p1 = res[i]
            p2 = res[i+1]
            w1 = weights[i]
            w2 = weights[i+1]
            
            vector = np.array(p2) - np.array(p1)
            #length = np.linalg.norm(vector)
            
            percentage = w1 / (w1 + w2)
            p = p1 + percentage*vector

            new_res.append(p)
            new_weights.append(w1 + w2)
        res = new_res
        weights = new_weights
    return res

def centroid(points):
  if type(points[0]) == str:
    points = [tuple(map(float, point[1:-1].split(','))) for point in points]

  x = [p[0] for p in points]
  y = [p[1] for p in points]
  return (sum(x) / len(points), sum(y) / len(points))

def distance_from_point(y, labels, amt = 3): 
  #grad_fn = y.grad_fn
  dist = []
  BOX_INDEXES = [61, 18, 9, 35, 19, 43, 17, 44, 25, 62, 52, 27, 26, 10, 34, 60, 36]
  #print(len(y))
  for i in range(len(y)):
    # Get the four best y values with in each batch
    y_ = y[i].topk(amt, dim=0)[1]
    lat, lng = labels["lat"][i], labels["lng"][i]
    # Get y_ to the cpu
    y_ = y_.cpu().numpy()
    # Get the box cords for each of the four best y values
    box_cords = n_list.iloc[[BOX_INDEXES[y_[idx]] for idx in range(len(y_))]]
  
    box_centers = []
    for _, row in box_cords.iterrows():
      box_centers.append(centroid([row['1'], row['0']]))
    
    #print(torch.softmax(y[i], dim=0).detach().cpu().numpy())
    percentages =  torch.softmax(y[i], dim=0).detach().cpu().numpy()
    #get percentage of each box selected
    percentages = percentages[[y_[idx] for idx in range(len(y_))]]
    #calculate new percentages so that they add up to 1
    percentages = percentages / percentages.sum()

    estimated_point = weighted_centroid(box_centers,percentages)[0]
    #print(estimated_point)
    esitmated_point = transform(zone34N, gps, estimated_point[0], estimated_point[1])
    estimated_point = (esitmated_point[1], esitmated_point[0])
    #y[i] = geopy.distance.distance(estimated_point, (lat, lng)).km
    dist.append(geopy.distance.geodesic((lat, lng), estimated_point).km)

  #res = torch.mean(torch.tensor(dist))#.to(device)
  #res.grad_fn = grad_fn
  return dist


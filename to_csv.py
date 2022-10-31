import pandas as pd
import os

images = os.listdir('./data/images')
print("image files", len(images))

bad_images = ["0009.jpg","0024.jpg","0049.jpg","0104.jpg","0146.jpg","0153.jpg","0164.jpg","0167.jpg","0239.jpg","0240.jpg","0248.jpg","0280.jpg","0335.jpg","0350.jpg","0354.jpg","0382.jpg","0443.jpg","0487.jpg","0489.jpg","0513.jpg","0533.jpg","0564.jpg","0605.jpg","0608.jpg","0613.jpg","0615.jpg","0656.jpg","0688.jpg","0753.jpg","0755.jpg","0774.jpg","0872.jpg","0876.jpg","0910.jpg","0938.jpg","0942.jpg","0981.jpg","1033.jpg","1034.jpg","1036.jpg","1068.jpg","1083.jpg","1105.jpg","1124.jpg","1142.jpg","1159.jpg","1170.jpg","1188.jpg","1259.jpg","1286.jpg","1354.jpg","1364.jpg","1393.jpg","1434.jpg","1447.jpg","1454.jpg","1527.jpg","1553.jpg","1645.jpg","1647.jpg","1744.jpg","1757.jpg","1767.jpg","1772.jpg","1862.jpg","1871.jpg","1881.jpg","1937.jpg","1940.jpg","1959.jpg","1994.jpg","2020.jpg","2045.jpg","2084.jpg","2126.jpg","2129.jpg","2133.jpg","2246.jpg","2314.jpg","2425.jpg","2478.jpg","2479.jpg","2483.jpg","2509.jpg","2524.jpg","2614.jpg","2636.jpg","2662.jpg","2682.jpg","2700.jpg","2747.jpg","2772.jpg","2788.jpg","2888.jpg","2918.jpg","2969.jpg","2998.jpg","3000.jpg","3110.jpg","3131.jpg","3132.jpg","3138.jpg","3159.jpg","3250.jpg","3381.jpg","3430.jpg","3474.jpg","3580.jpg","3625.jpg","3627.jpg","3629.jpg","3645.jpg","3660.jpg","3675.jpg","3720.jpg","3744.jpg"]

df = pd.read_json('./data/locations_county.json')
df['filename'] = images
print(df.iloc[:10])
df = df[df['filename'].isin(bad_images) == False]
print(df.iloc[:10])
df.to_csv('./data/locations_county.csv', index=False)
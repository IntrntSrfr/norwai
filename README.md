# NorwAI - Geoguessing in Norway

Metaverse, Web3, NFT, AI, Blockchain, Big data, Crypto, Data mining, IoT, Ecosystem, Quantum computing

## Approaches

We have made different approaches, attempting to solve the problem. We have a regression model, and a classifier.
There will be some files that belong to each approach. A script for training the model, and notebooks for testing.

## The files and their usecases

- [`norwai_regression.py`](norwai_regression.py) and [`norwai_regression_test.ipynb`](norwai_regression_test.ipynb)
  - Script for training, and notebook for testing the regression .
- [`norwai_county.py`](norwai_county.py) and [`norwai_county_test.ipynb`](norwai_county_test.ipynb)
  - Script for training, and notebook for testing the classifier model.
- [`nsvd.py`](nsvd.py)
  - Pytorch Dataset class for the data. NSVD - Norwegian StreetView Dataset.
- [`locations.ipynb`](locations.ipynb)
  - Grabs random streetview locations, cleans and saves them to a json file.
- [`images.ipynb`](images.ipynb)
  - Grabs images from streetview API using all the locations grabbed by the file above.
- [`graph.ipynb`](graph.ipynb) and [`fylker.ipynb`](fylker.ipynb)
  - Both serve as a means of visualizing the locations we have gathered on a map.
  The first only plots locations, while the second also shows their counties. The latter also adds county data to the dataset.
- [`to_csv.py`](to_csv.py)
  - As some images are not useful (tunnels or locations with broken images), this file
  connects the locations with the image files, and removes a defined array of bad images.
  It then saves the cleaned data to a csv file.

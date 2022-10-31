# NorwAI - Geoguessing in Norway

Metaverse, Web3, NFT, AI, Blockchain, Big data, Crypto, Data mining, IoT, Ecosystem, Quantum computing

## The files and their usecases

- [`norwai.ipynb`](norwai.ipynb)
  - Notebook for training and testing. Testing will be moved later.
- [`norwai_test.ipynb`](norwai_test.ipynb)
  - Notebook that will contain the testing in the future.
- [`locations.ipynb`](locations.ipynb)
  - Grabs random streetview locations, cleans and saves them to a json file.
- [`images.ipynb`](images.ipynb)
  - Grabs images from streetview API using all the locations grabbed by the file above.
- [`graph.ipynb`](graph.ipynb) and [`fylker.ipynb`](fylker.ipynb)
  - Both serve as a means of visualizing the locations we have gathered on a map.
  The first only plots locations, while the second shows them in their respective municipality as well.
- [`to_csv.py`](to_csv.py)
  - As some images are not useful (tunnels or locations with broken images), this file
  connects the locations with the image files, and removes a defined array of bad images.
  It then saves the cleaned data to a csv file.
- [`nsvd.py`](nsvd.py)
  - Pytorch Dataset class for the data. NSVD - Norwegian StreetView Dataset.

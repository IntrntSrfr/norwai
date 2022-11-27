# NorwAI - Geoguessing in Norway

An attempt at teaching a machine how to geolocate images in Norway using different approaches and methods.
We have made our own dataset, containing over 12000 streetview images in Norway. NSVD - Norwegian StreetView Dataset.

Training metrics has been logged using [Weights and Biases](https://wandb.ai/).

## Approaches

We have made three approaches. A purely coordinate/distance-based model, a county classifier, and dividing geography
into zones and classifying those zones.

### Distance

[`norwai_regression_train.py`](norwai_regression_train.py) contains training code. With this approach, we tested several models,
including preexisting models, with and without pretrained weights, as well as our own model. It uses a custom loss function (Haversine) based on
the distance between coordinates on a great circle. This is computationally expensive, but will give the most accurate loss.

Trial and error based on the limited resources we have available, showed a pretrained EfficientNet (large) to be the best model
with an average loss of 182km on the test set.

[`norwai_regression_train.ipynb`](norwai_regression_test.ipynb) is a notebook for testing this approach.

### County

[`norwai_county_train.py`](norwai_county_train.py) contains training code. This approach was tested using different models. Our own,
preexisting models, with and without pretrained weights. This approach is a classification problem, so cross entropy loss is used,
with 10 classes, one for each county, excluding Oslo. This is due to Oslo having too few training samples to be useful, so it was dropped.

For this approach, we found ResNet152 and EfficientNet (large) to do best, with an accuracy of around 58% on the test set.
ResNet152 seemed to have a very slight edge at times compared to EfficientNet, but not enough to say it is definitively better.
They have an accuracy difference of 1.2%, and similar metrics in terms of loss.

[`norwai_county_test.ipynb`](norwai_county_test.ipynb) is a notebook for testing this approach.

### Zone

[`norwai_zone_train.py`](norwai_zone_train.py) contains training code with wandb. This approach was tested using different models and county sizes. We tested both 400km^2 and 200km^2. The models that did the best was vgg19 and EfficientNet (small) with an accuracy of around 300km. 

[`norwai_zone.py`](norwai_zone.py) contains training code.

[`norwai_zone_test.ipynb`](norwai_zone_test.ipynb) isa a notebook for testing the zone approach.

## Other files and folders

- [`collection`](collection/) contains all code for collecting images and putting together the dataset.
- [`config`](config/) contains WandB config files.
- [`nsvd.py`](nsvd.py) contains the dataset class to be used with PyTorch.
- [`norwai_models.py`](norwai_models.py) contains code to fetch models we wish to train and related code.
- [`util.py`](util.py) small code to create folders to save finished models.

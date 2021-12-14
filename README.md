# Movie-Review-Dataset
## Problem description

This problem is about predicting the weather of the contenent of Austerilia.

The [data](https://ai.stanford.edu/~amaas/data/sentiment/)
IMDB movie review dataset, which is a dataset for binary sentiment classification. The IMDB dataset was first proposed by Maas. As a benchmark for sentiment analysis. The core dataset contains 50,000 reviews split evenly into 25k training and 25k testing sets. The overall distribution of labels is balanced in both the training and testing sets (25k positive and 25k negative). There are additional 50,000 unlabeled reviews that may be used for unsupervised learning.

# Problem Thinking

* Download Data and Apply Text Pre-processing.
* Loading the data which is the reviews from the given files.
Text pre-processing is essential for NLP tasks. I used the open source NLTK for available text-preprocessing operations such as: tokenization, stop words removal, stemming, lemmatization, etc.
* Create a Data Matrix.
Needed to convert the text of each review (after pre-processing) into a vector form to construct the data matrix.
I convert the text into its BoW representaional model.
* Trying different ML calssification algorithms such as:
    * KNN (unsupervised)
    * XGB random forest (supervised)
* tunning the models parameters
* selecting the model with the best Accuracy score

# Building virtual environment 
```bash
# installing PIPENV virtual environment creator
pip install pipenv

# installing the dependancies
pipenv intall numpy pandas sklearn xgboost flask requests gunicorn

# for Running the virtual environment
pipenv shell
```
# How to build a container (Docker)
```bash
# Make sure you have docker first
# After building your own image use the following command to build the image
# the '.' assumes that you run the terminal from the same directory where the Dockerfile exists
sudo docker build . -t weather:1.0
# to run the docker image
sudo docker run --name <Container name> -it -p <continer port>:<forwarding port> -e <entry point if needed> <docker image name :version(latest by default)>  
```


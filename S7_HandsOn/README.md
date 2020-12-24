# Hands On 2 - Sentiment Analysis using LSTM/RNN

## Problem Statement
- Download the StanfordSentimentAnalysis Dataset. This dataset contains just over 10,000 pieces of Stanford data from HTML files of Rotten Tomatoes. The sentiments are rated between 1 and 25, where one is the most negative and 25 is the most positive.
- Apply "Back Translate", i.e. using Google translate to convert the sentences. "random_swap" "random_delete" to augment the data you are training on
- Train your model and try and achieve 60%+ accuracy. Upload your colab file on git with the the training logs

# Results

- RNN's LSTM and GRU are unable to achieve the results
- Applying augmentation takes too much time on a large dataset. Specially Random Translate - Even after 2 hours it still shows no results
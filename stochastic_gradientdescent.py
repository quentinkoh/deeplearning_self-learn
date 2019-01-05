# import the necessary packages
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse


def sigmoid_activation(x):
    # to compute sigmoid activation value for a given input
    return 1.0 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    # to compute the derivative of sigmoid function assuming that
    # input x has already passed through sigmoid_activation()
    return x * (1 - x)


def predict(X, W):
    # dot product between our features and weight matrices
    preds = sigmoid_activation(X.dot(W))

    # apply step function to threshold the outputs to binary
    # class labels
    preds[preds <= 0.5] = 0
    preds[preds > 0.5] = 1

    # return predictions
    return preds


def next_batch(X, y, batchSize):
    # loop over our dataset 'X' in mini-batches, yielding a tuple of
    # the current batched data and labels
    for i in np.arange(0, X.shape[0], batchSize):
        yield (X[i:i + batchSize], y[i:i + batchSize])


# construct argument parse and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument('-e', '--epochs', type=float, default=100,
                help='# of epochs')
ap.add_argument('-a', '--alpha', type=float, default=0.01,
                help='learning rate')
ap.add_argument('-b', '--batch-size', type=int, default=32,
                help='size of SGD mini-batches')
args = vars(ap.parse_args())

# generate a binary classification problem with 1,000 data points,
# where each data point is a 2D feature vector
(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2,
                    cluster_std=1.5, random_state=1)
y = y.reshape((y.shape[0], 1))

# insert a column of 1's as the last entry in the feature
# matrix (known as bias trick) so as to allow us to treat
# bias vector as a trainable parameter within the weight matrix
X = np.c_[X, np.ones((X.shape[0]))]

# train test split by 50 / 50 ratio
(trainX, testX, trainY, testY) = train_test_split(X, y,
                                                  test_size=0.5, random_state=42)

# initialize our weight matrix and list of losses
print('[INFO] training...')
W = np.random.randn(X.shape[1], 1)
losses = []

# loop over the desired number of epochs
for epoch in np.arange(0, args['epochs']):
    # initialize the total loss for the epoch
    epochLoss = []

    # loop over our data in batches
    for (batchX, batchY) in next_batch(trainX, trainY, args['batch_size']):
        # take the dot product between our current batch of features
        # and the weight matrix, then pass this value through our
        # activation function
        preds = sigmoid_activation(batchX.dot(W))

        # using preds vector, we shall determine our error
        error = preds - batchY
        epochLoss.append(np.sum(error**2))

        # gradient descent update
        d = error * sigmoid_deriv(preds)
        gradient = batchX.T.dot(d)

        # weight matrix update where we take a small step towards
        # the local / global minimum of loss function. the size of
        # this step will be determined by our learning rate, alpha
        W += -args['alpha'] * gradient

    # update our loss history by taking the average loss across all
    # batches
    loss = np.average(epochLoss)
    losses.append(loss)

    # check to see if an update should be displayed
    if epoch == 0 or (epoch + 1) % 5 == 0:
        print('[INFO] epoch={}, loss={:.7f}'.format(int(epoch+1),
                                                    loss))

# evaluate our model
print('[INFO] evaluating...')
preds = predict(testX, W)
print(classification_report(testY, preds))

# plot the (testing) classification data
plt.style.use('ggplot')
plt.figure()
plt.title('Data')
plt.scatter(testX[:, 0], testX[:, 1], marker='o', c=testY[:, 0], s=30)

# construct a figure that plots the loss over time
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, args['epochs']), losses)
plt.title('Training Loss')
plt.xlabel('Epoch #')
plt.ylabel('Loss')
plt.show()

# import the necessary packages
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from quentindl.preprocessing import DataPreprocessing
from quentindl.datasets import DatasetLoader
from imutils import paths
import argparse

# construct argument parse and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True,
                help='path to input dataset')
args = vars(ap.parse_args())

# pull the list of images that we'll be analyzing
print('[INFO] loading images...')
imagePaths = list(paths.list_images(args['dataset']))

# initialize the image preprocessor, load the dataset from disk,
# and reshape the data matrix
dp = DataPreprocessing(32, 32)
dsl = DatasetLoader(preprocessors=[sp])
(data, labels) = dsl.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], 3072))

# encode the labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# train test split by 75 / 25 ratio
(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.25, random_state=5)

# loop over our set of regularizers
for r in (None, 'l1', 'l2'):
    # train a SGD classifier using softmax loss function and the
    # specified regularizer for 10 epochs
    print('[INFO] training model with {} penalty'.format(r))
    model = SGDClassifier(loss='log', penalty=r, max_iter=10,
                          learning_rate='constant', eta0=0.01, random_state=42)
    model.fit(trainX, trainY)

    # evaluate the classifier
    acc = model.score(testX, testY)
    print('[INFO] {} penalty accuracy: {:.2f}%'.format(r,
                                                       acc * 100))

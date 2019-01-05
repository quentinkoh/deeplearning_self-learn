# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from quentindl.preprocessing import DataPreprocessing
from quentindl.datasets import DatasetLoader
from imutils import paths
import argparse

# construct argument parse and parse arguments
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True,
                help='path to input dataset')
ap.add_argument('-k', '--neighbors', type=int, default=1,
                help='# of nearest neighbors for classification')
ap.add_argument('-j', '--jobs', type=int, default=-1,
                help='# of jobs for knn (-1 uses all available cores)')
args = vars(ap.parse_args())

# pull the list of images that we'll be analyzing
print('[INFO] loading images...')
imagePaths = list(paths.list_images(args['dataset']))

# initialize the image preprocessor, load dataset from disk,
# and reshape the data matrix
dp = DataPreprocessing(32, 32)
dsl = DatasetLoader(preprocessors=[sp])
(data, labels) = dsl.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], 3072))

# show some information on memory consumption of the images
print('[INFO] features matrix: {:.1f}MB'.format(
    data.nbytes/(1024*1000.0)))

# encode labels as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

# train test split by 75 / 25 ratio
(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.25, random_state=42)

# train and evaluate a knn classifier on raw pixel intensities
print('[INFO] evaluating knn classifier...')
model = KNeighborsClassifier(n_neighbors=args['neighbors'],
                             n_jobs=args['jobs'])
model.fit(trainX, trainY)
print(classification_report(testY, model.predict(testX),
                            target_names=le.classes_))

import numpy as np

'''We have images of 32x32 in RGB format
labels are a list of 10000 with integers in 0-9
where i-th index value gives the index of i-th image'''

CIFAR_DIR = 'cifar-10-batches-py/'
dirs = ['batches.meta', 'data_batch_1', 'data_batch_2',
        'data_batch_3', 'data_batch_4', 'data_batch_5',
        'test_batch']


def unpickle(file):
    import pickle
    with open(file, 'rb') as opened_file:
        cifar_dict = pickle.load(opened_file, encoding='bytes')
    return cifar_dict


def one_hot_encode(vec, vals=10):
    '''Used to encode a vector of image labels into their one hot coresp'''
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out


class CifarHelper():
    def __init__(self, data):
        self.batch_number = 0

        self.all_train_batches = data[1:len(data) - 1]
        self.test_batch = data[-1]
        self.training_images = None
        self.training_labels = None

        self.test_images = None
        self.test_labels = None

    def setup_train_images(self):
        print("Setting up training dataset")
        # Vertically stack the images
        self.training_images = np.vstack([batch[b"data"] for batch in self.all_train_batches])
        train_len = len(self.training_images)

        # Reshape and normalize the training images
        self.training_images = self.training_images.reshape(train_len, 3, 32, 32).transpose(0, 2, 3, 1)
        array = np.array(self.training_images)
        self.training_images = (array - array.mean(axis=(0,1, 2),keepdims=True)) / array.std(axis=(0,1, 2),keepdims=True)

        self.training_labels = one_hot_encode(np.hstack([batch[b"labels"] for batch in self.all_train_batches]))

    def setup_test_images(self):
        print("Setting up test dataset")
        # Vertically stack the images
        self.test_images = np.vstack(self.test_batch[b"data"])
        test_len = len(self.test_images)

        # Reshape and normalize the test images
        self.test_images = self.test_images.reshape(test_len, 3, 32, 32).transpose(0, 2, 3, 1)
        array = np.array(self.test_images)
        self.test_images = (array - array.mean(axis=(0,1, 2),keepdims=True)) / array.std(axis=(0,1, 2),keepdims=True)

        # One hot encode
        self.test_labels = one_hot_encode(np.hstack(self.test_batch[b"labels"]))
        # print(np.array(self.test_batch[b"labels"]).shape)
        # print(np.hstack(self.test_batch[b"labels"]))
        # print(self.test_labels)

    def next_batch(self, batch_size=100):
        xx = self.training_images[self.batch_number:self.batch_number + batch_size].reshape(batch_size, 32,
                                                                                            32, 3)
        # array = np.array(xx)
        # xx = (array - array.mean(axis=(1,2))) / array.std(axis=(1,2))

        yy = self.training_labels[self.batch_number:self.batch_number + batch_size]
        self.batch_number = (self.batch_number + batch_size) % len(self.training_images)
        return xx, yy

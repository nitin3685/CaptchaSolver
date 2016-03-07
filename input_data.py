from PIL import Image
import numpy
import re
import glob

def single_sparse_label(raw_label, digit_index):
    sparse_indices = numpy.zeros(shape=(9))
    raw_digit = list(raw_label)[digit_index]
    sparse_index = int(raw_digit) - 1
    sparse_indices[sparse_index] = 1
    return sparse_indices

def full_sparse_label(raw_label):
    sparse_indices = numpy.zeros(shape=(54))
    for digit_index in range(0, 5):
        raw_digit = list(raw_label)[digit_index]
        sparse_index = (int(raw_digit) - 1) + (digit_index * 9)
        sparse_indices[sparse_index] = 1
    return sparse_indices

def extract_image(filename, digit_index):
    # print('Extracting image:', filename)
    image = Image.open(filename)
    image = numpy.array(image.getdata()).reshape(image.size[0], image.size[1], 3)
    image = image[15:155]
    # image = image[90:130]
    image = 1. - numpy.sum(image, axis=-1) / 765
    return image

def extract_label(filename, digit_index):
    # print('Extracting label:', filename)
    m = re.search('(\d{6})', filename)
    label = None
    if digit_index:
        label = single_sparse_label(m.group(0), digit_index)
    else:
        label = full_sparse_label(m.group(0))
    return label

def extract_images(filenames, digit_index):
    images = []
    counter = 100
    for f in filenames:
        if counter % 500 == 0:
            print(counter)
        counter += 1
        images.append(extract_image(f, digit_index))
    return numpy.array(images)

def extract_labels(filenames, digit_index):
    labels = []
    for f in filenames:
        labels.append(extract_label(f, digit_index))
    return numpy.array(labels)

class DataSet(object):
  def __init__(self, images, labels):
    assert images.shape[0] == labels.shape[0], (
        "images.shape: %s labels.shape: %s" % (images.shape,
                                               labels.shape))
    self._num_examples = images.shape[0]
    # Convert shape from [num examples, rows, columns, depth]
    # to [num examples, rows*columns] (assuming depth == 1)
    images = images.reshape(images.shape[0],
                            images.shape[1] * images.shape[2])
    # Convert from [0, 255] -> [0.0, 1.0].
    # images = images.astype(numpy.float32)
    # images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0
  @property
  def images(self):
    return self._images
  @property
  def labels(self):
    return self._labels
  @property
  def num_examples(self):
    return self._num_examples
  @property
  def epochs_completed(self):
    return self._epochs_completed
  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]

def read_data_sets(digit_index=False):
    class DataSets(object):
        pass
    data_sets = DataSets()

    print("Extract train images")
    train_filenames = numpy.array(glob.glob("./data_train/*.jpg"))
    numpy.random.shuffle(train_filenames)
    train_images = extract_images(train_filenames, digit_index)
    train_labels = extract_labels(train_filenames, digit_index)

    print("Extract test images")
    test_filenames = numpy.array(glob.glob("./data_test/*.jpg"))
    numpy.random.shuffle(test_filenames)
    test_images = extract_images(test_filenames, digit_index)
    test_labels = extract_labels(test_filenames, digit_index)

    data_sets.train = DataSet(train_images, train_labels)
    data_sets.test = DataSet(test_images, test_labels)
    return data_sets

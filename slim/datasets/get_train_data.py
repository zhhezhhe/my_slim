
from datasets import dataset_utils
import tensorflow as tf

slim = tf.contrib.slim

_NUM_CLASSES = 16
_SPLITS_TO_SIZES = {'train': 30000, 'validation': 2000} 

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A [224 x 224 x 3]  image.',
    'label': 'A single integer between 0 and 8',
}


def get_dataset(name, split_name, dataset_dir):

    reader = tf.TFRecordReader
    keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
      'image/class/label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(shape=[224, 224, 3]),
        'label': slim.tfexample_decoder.Tensor('image/class/label', shape=[]),
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)
    labels_to_names = None
    dataset_dir = dataset_dir+'/'+name
        
    return slim.dataset.Dataset(
        data_sources=dataset_dir,
        reader=reader,
        decoder=decoder,
        num_samples=_SPLITS_TO_SIZES[split_name],
        num_classes=_NUM_CLASSES,
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
        labels_to_names=labels_to_names)

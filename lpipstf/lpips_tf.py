import os
import sys
import tensorflow as tf
from six.moves import urllib

_URL = 'http://rail.eecs.berkeley.edu/models/lpips'


def _download(url, output_dir):
    """Downloads the `url` file into `output_dir`."""
    filename = url.split('/')[-1]
    filepath = os.path.join(output_dir, filename)

    def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (
            filename, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(url, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')


def lpips(input0, input1, model='net-lin', net='alex', version=0.1):
    """
    Learned Perceptual Image Patch Similarity (LPIPS) metric for TensorFlow 2.x.
    """
    tf.compat.v1.disable_eager_execution()

    # Ensure input tensors are reshaped appropriately for processing
    batch_size = tf.shape(input0)[0]  # Extract batch size from input
    input0 = tf.reshape(input0, [batch_size, 256, 256, 3])
    input1 = tf.reshape(input1, [batch_size, 256, 256, 3])

    # Transform NHWC to NCHW
    input0 = tf.transpose(input0, [0, 3, 1, 2])
    input1 = tf.transpose(input1, [0, 3, 1, 2])

    input0 = input0 * 2.0 - 1.0
    input1 = input1 * 2.0 - 1.0

    cache_dir = os.path.expanduser('~/.lpips')
    os.makedirs(cache_dir, exist_ok=True)

    pb_fnames = [
        '%s_%s_v%s_%d.pb' % (model, net, version, tf.compat.v1.get_default_graph().graph_def_versions.producer),
        '%s_%s_v%s.pb' % (model, net, version),
    ]

    for pb_fname in pb_fnames:
        if not os.path.isfile(os.path.join(cache_dir, pb_fname)):
            try:
                _download(os.path.join(_URL, pb_fname), cache_dir)
            except urllib.error.HTTPError:
                pass
        if os.path.isfile(os.path.join(cache_dir, pb_fname)):
            break

    with open(os.path.join(cache_dir, pb_fname), 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

        with tf.compat.v1.Session() as sess:
            # Load the graph without using restricted tensors directly
            tf.compat.v1.import_graph_def(graph_def, name='')

            # Instead of fetching directly, operate within the same graph scope
            distance = sess.graph.get_operations()[-1].outputs[0]

    # Ensure distance tensor is reshaped correctly based on the batch size
    shape = tf.shape(distance)
    flat_shape = tf.reduce_prod(shape[1:])  # Compute number of elements per batch

    # Reshape `distance` to the correct shape
    distance = tf.reshape(distance, [batch_size, flat_shape])

    return distance


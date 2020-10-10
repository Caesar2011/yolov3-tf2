import tensorflow as tf
from absl.flags import FLAGS
import numpy as np

@tf.function
def transform_targets_for_output(y_true, grid_size, anchor_idxs):
    # y_true: (N, boxes, (x1, y1, x2, y2, class, best_anchor))
    N = tf.shape(y_true)[0]

    # y_true_out: (N, grid_y, grid_x, anchors, [x, y, w, h, obj, class])
    y_true_out = tf.zeros(
        (N, grid_size[0], grid_size[1], tf.shape(anchor_idxs)[0], 6))

    anchor_idxs = tf.cast(anchor_idxs, tf.int32)

    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
    idx = 0
    for i in tf.range(N):
        for j in tf.range(tf.shape(y_true)[1]):
            if tf.equal(y_true[i][j][2], 0):
                continue
            anchor_eq = tf.equal(
                anchor_idxs, tf.cast(y_true[i][j][5], tf.int32))

            if tf.reduce_any(anchor_eq):
                box = y_true[i][j][0:4]
                box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2

                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                grid_xy = tf.cast(box_xy * (grid_size[1], grid_size[0]), tf.int32)
                grid_xy = tf.minimum(tf.maximum((0, 0), grid_xy), (grid_size[1]-1, grid_size[0]-1))

                # grid[y][x][anchor] = (tx, ty, bw, bh, obj, class)
                indexes = indexes.write(
                    idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]])
                updates = updates.write(
                    idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]])
                idx += 1

    # tf.print(indexes.stack())
    # tf.print(updates.stack())

    return tf.tensor_scatter_nd_update(
        y_true_out, indexes.stack(), updates.stack())


def transform_targets(y_train, anchors, anchor_masks, size):
    # size is a tuple (height, width)
    y_outs = []
    grid_size = (size[0] // 32, size[1] // 32)

    # calculate anchor index for true boxes
    anchors = tf.cast(anchors, tf.float32)
    anchor_area = anchors[..., 0] * anchors[..., 1]
    box_wh = y_train[..., 2:4] - y_train[..., 0:2]
    box_wh = tf.tile(tf.expand_dims(box_wh, -2),
                     (1, 1, tf.shape(anchors)[0], 1))
    box_area = box_wh[..., 0] * box_wh[..., 1]
    intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * \
        tf.minimum(box_wh[..., 1], anchors[..., 1])
    iou = intersection / (box_area + anchor_area - intersection)
    anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
    anchor_idx = tf.expand_dims(anchor_idx, axis=-1)

    y_train = tf.concat([y_train, anchor_idx], axis=-1)

    for anchor_idxs in anchor_masks:
        y_outs.append(transform_targets_for_output(
            y_train, grid_size, anchor_idxs))
        grid_size = (grid_size[0]*2, grid_size[1]*2)

    return tuple(y_outs)


def transform_images(x_train, size):
    # size is a tuple (height, width)
    x_train = tf.image.resize(x_train, size)
    x_train = x_train / 255
    return x_train

def get_recurrect_inputs(x, y, anchors_list, anchor_masks, classes):
    y2 = []

    @tf.function
    def inverse_sigmoid(x):
        return -tf.math.log(1. / x - 1.)

    @tf.function
    def transform_xy(true_xy, y_elem_shape):
        # 3a. inverting the pred box equations
        grid_size = y_elem_shape[1:3]
        grid = tf.meshgrid(tf.range(grid_size[1]), tf.range(grid_size[0]))
        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        true_xy = true_xy * tf.cast((grid_size[1], grid_size[0]), tf.float32) - \
            tf.cast(grid, tf.float32)
        return true_xy

    anchors_list = np.take(anchors_list, anchor_masks, axis=0)
    idx = -1
    for y_elem, anchors in zip(y, anchors_list):
        idx += 1
        anchors = tf.constant(anchors)
        # 2. transform all true outputs
        # y_elem: (batch_size, grid_y, grid_x, anchors, (x1, y1, x2, y2, obj, cls))
        true_box, true_obj, true_class_idx = tf.split(
            y_elem, (4, 1, 1), axis=-1)
        condition = tf.math.greater(true_obj, 0)

        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2
        true_wh = true_box[..., 2:4] - true_box[..., 0:2]

        # modify data to model the previous step
        # xy more centered / wh smaller / condition dropout
        shape = tf.shape(true_xy)
        shape = tf.concat((shape[0:1], tf.ones((tf.size(shape)-1, ), dtype="int32")), axis=0)
        movement = tf.reshape(tf.random.normal(shape=(shape[0], ), mean=0.98, stddev=0.02), shape)

        true_xy = (true_xy-0.5)*movement + 0.5
        true_wh = true_wh * movement
        condition = tf.math.logical_xor(condition, tf.random.uniform(tf.shape(condition)) > 0.95)

        # 3b. inverting the pred box equations
        true_wh = tf.math.log(true_wh / anchors)
        true_wh = tf.where(tf.math.is_inf(true_wh),
                           tf.ones_like(true_wh) * -20, true_wh)

        shape = tf.shape(true_wh[..., 0])
        normal_w = tf.random.normal(shape=shape, mean=-1.0, stddev=0.4)
        normal_h = tf.random.normal(shape=shape, mean=6.55533592e-02, stddev=2.29207946e-01)
        true_wh = tf.where(condition, true_wh, tf.stack((normal_w, normal_h), axis=-1))

        # Reverse sigmoid of yolo_boxes to be equal with pred_xy
        normal_xy = tf.random.normal(shape=tf.shape(true_xy), mean=0.0, stddev=1.0)
        true_xy = tf.where(condition, transform_xy(true_xy, tf.shape(y_elem)), normal_xy)
        true_obj = tf.where(condition, tf.random.normal(shape=tf.shape(true_obj), mean=20.0, stddev=3.0), tf.random.normal(shape=tf.shape(true_obj), mean=-20.0, stddev=3.0))

        classes_one_hot = tf.one_hot(tf.cast(true_class_idx, dtype=tf.int32), classes)
        classes_one_hot = tf.squeeze(classes_one_hot, axis=-2)
        classes_one_hot = classes_one_hot * 0.9

        # y_pred: (batch_size, grid_y, grid_x, anchors, (x, y, w, h, obj, ...cls))
        y2.append(tf.concat((
            true_xy,
            true_wh,
            true_obj,
            classes_one_hot
        ), axis=-1))

    return (x, ) + tuple(y2)


# https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md#conversion-script-outline-conversion-script-outline
# Commented out fields are not required in our project
IMAGE_FEATURE_MAP = {
    # 'image/width': tf.io.FixedLenFeature([], tf.int64),
    # 'image/height': tf.io.FixedLenFeature([], tf.int64),
    # 'image/filename': tf.io.FixedLenFeature([], tf.string),
    # 'image/source_id': tf.io.FixedLenFeature([], tf.string),
    # 'image/key/sha256': tf.io.FixedLenFeature([], tf.string),
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    # 'image/format': tf.io.FixedLenFeature([], tf.string),
    'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
    'image/object/class/text': tf.io.VarLenFeature(tf.string),
    # 'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    # 'image/object/difficult': tf.io.VarLenFeature(tf.int64),
    # 'image/object/truncated': tf.io.VarLenFeature(tf.int64),
    'image/object/view': tf.io.VarLenFeature(tf.string),
}


def parse_tfrecord(tfrecord, class_table, size=None):
    # size is a tuple (height, width)
    if size is None:
        size = (416, 416)
    x = tf.io.parse_single_example(tfrecord, IMAGE_FEATURE_MAP)
    x_train = tf.image.decode_jpeg(x['image/encoded'], channels=3)
    x_train = tf.image.resize(x_train, size)

    class_text = tf.sparse.to_dense(
        x['image/object/class/text'], default_value='')
    labels = tf.cast(class_table.lookup(class_text), tf.float32)
    y_train = tf.stack([tf.sparse.to_dense(x['image/object/bbox/xmin']),
                        tf.sparse.to_dense(x['image/object/bbox/ymin']),
                        tf.sparse.to_dense(x['image/object/bbox/xmax']),
                        tf.sparse.to_dense(x['image/object/bbox/ymax']),
                        labels], axis=1)

    poses = tf.sparse.to_dense(
        x['image/object/view'], default_value='')
    specs = tf.stack([poses], axis=1)

    y_train = tf.boolean_mask(y_train,
        tf.equal(specs[:, 0], "Frontal")
    )

    paddings = [[0, FLAGS.yolo_max_boxes - tf.shape(y_train)[0]], [0, 0]]
    y_train = tf.pad(y_train, paddings)

    return x_train, y_train


def load_tfrecord_dataset(file_pattern, class_file, size=None):
    # size is a tuple (height, width)
    LINE_NUMBER = -1  # TODO: use tf.lookup.TextFileIndex.LINE_NUMBER
    class_table = tf.lookup.StaticHashTable(tf.lookup.TextFileInitializer(
        class_file, tf.string, 0, tf.int64, LINE_NUMBER, delimiter="\n"), -1)

    files = tf.data.Dataset.list_files(file_pattern)
    dataset = files.flat_map(tf.data.TFRecordDataset)
    return dataset.map(lambda x: parse_tfrecord(x, class_table, size))


def load_fake_dataset():
    x_train = tf.image.decode_jpeg(
        open('./data/girl.png', 'rb').read(), channels=3)
    x_train = tf.expand_dims(x_train, axis=0)

    labels = [
        [0.18494931, 0.03049111, 0.9435849,  0.96302897, 0],
        [0.01586703, 0.35938117, 0.17582396, 0.6069674, 56],
        [0.09158827, 0.48252046, 0.26967454, 0.6403017, 67]
    ] + [[0, 0, 0, 0, 0]] * 5
    y_train = tf.convert_to_tensor(labels, tf.float32)
    y_train = tf.expand_dims(y_train, axis=0)

    return tf.data.Dataset.from_tensor_slices((x_train, y_train))

import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_integer('height', None, 'resize image height to (overrides size)')
flags.DEFINE_integer('width', None, 'resize image width to (overrides size)')
flags.DEFINE_string('image', './data/girl.png', 'path to input image')
flags.DEFINE_string('tfrecord', None, 'tfrecord instead of image')
flags.DEFINE_integer('count', 1, 'if tfrecord is set, amount of images')
flags.DEFINE_string('output', './output.jpg', 'path to output image')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')


def main(_argv):
    # Change flag values
    if FLAGS.height is None:
        FLAGS.height = FLAGS.size
    if FLAGS.width is None:
        FLAGS.width = FLAGS.size
    size = (FLAGS.width, FLAGS.height)

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights).expect_partial()
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    if FLAGS.tfrecord:
        dataset = load_tfrecord_dataset(
            FLAGS.tfrecord, FLAGS.classes, size)
        dataset = dataset.shuffle(512)
        img_iter = next(iter(dataset.take(FLAGS.count)))
    else:
        img_raw = tf.image.decode_image(
            open(FLAGS.image, 'rb').read(), channels=3)
        img_iter = iter([(img_raw, None)])

    for idx, (img_raw, _label) in enumerate(img_iter):
        img = tf.expand_dims(img_raw, 0)
        img = transform_images(img, size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo(img)
        t2 = time.time()
        logging.info('time: {}'.format(t2 - t1))

        logging.info('detections:')
        for i in range(nums[0]):
            logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                               np.array(scores[0][i]),
                                               np.array(boxes[0][i])))

        img = cv2.cvtColor(img_raw.numpy(), cv2.COLOR_RGB2BGR)
        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        output = FLAGS.output
        if idx != 0:
            dot = output.rfind(".")
            suffix = "-" + str(idx).zfill(3)
            if dot != -1:
                output = output[:dot] + suffix + output[dot:]
            else:
                output = output + suffix
        cv2.imwrite(output, img)
        logging.info('output saved to: {}'.format(output))


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs


flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_boolean('recurrent', False, 'recurrent or not')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_integer('height', None, 'resize image height to (overrides size)')
flags.DEFINE_integer('width', None, 'resize image width to (overrides size)')
flags.DEFINE_string('video', './data/video.mp4',
                    'path to video file or number for webcam)')
flags.DEFINE_string('output', None, 'path to output video')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')


def main(_argv):
    # Change flag values
    if FLAGS.height is None:
        FLAGS.height = FLAGS.size
    if FLAGS.width is None:
        FLAGS.width = FLAGS.size
    size = (FLAGS.height, FLAGS.width)

    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, True)

    if FLAGS.tiny:
        yolo = YoloV3Tiny(classes=FLAGS.num_classes, recurrent=FLAGS.recurrent)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes, recurrent=FLAGS.recurrent)

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    times = []

    try:
        vid = cv2.VideoCapture(int(FLAGS.video))
    except:
        vid = cv2.VideoCapture(FLAGS.video)

    if FLAGS.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
        out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))

    if FLAGS.recurrent:
        output_0 = tf.zeros((height // 32, width // 32, 3, 6))
        output_1 = tf.zeros((height // 16, width // 16, 3, 6))
        if not FLAGS.tiny:
            output_2 = tf.zeros((height // 8, width // 8, 3, 6))

    while True:
        _, img = vid.read()

        if img is None:
            logging.warning("Empty Frame")
            time.sleep(0.1)
            if not FLAGS.output:
                continue
            else:
                break

        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, size)

        t1 = time.time()
        if FLAGS.recurrent:
            if FLAGS.tiny:
                ((boxes, scores, classes, nums), output_0, output_1) = yolo.predict(img_in, output_0, output_1)
            else:
                ((boxes, scores, classes, nums), output_0, output_1, output_2) = yolo.predict(img_in, output_0, output_1, output_2)
        else:
            boxes, scores, classes, nums = yolo.predict(img_in)
        t2 = time.time()
        times.append(t2-t1)
        times = times[-20:]

        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        ms = sum(times)/len(times)*1000
        img = cv2.putText(img, "Time: {:.2f}ms ({:d})".format(ms, int(1000/ms)), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        if FLAGS.output:
            out.write(img)
        else:
            cv2.imshow('output', img)
            if cv2.waitKey(1) == ord('q'):
                break

    if not FLAGS.output:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

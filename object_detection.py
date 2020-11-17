import numpy as np
import tensorflow.compat.v1 as tf
import sys, os, time, argparse, cv2

from collections import defaultdict
from io import StringIO
from PIL import Image
from datetime import datetime

sys.path.append("..")
from utils import ops as utils_ops

#import utils from object detection module
from utils import label_map_util
from utils import visualization_utils as vis_util

# Helper code
def load_image_into_numpy_array(image):
  VID_WIDTH = image.shape[1]
  VID_HEIGHT = image.shape[0]
  return np.array(image).reshape(
      (VID_HEIGHT, VID_WIDTH, 3)).astype(np.uint8)
	  
def run_inference_for_single_image(image, graph, currConfig):
#  with tf.device('gpu'):
    with graph.as_default():
      with tf.Session(config=currConfig) as sess:
        # Get handles to input and output tensors
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
        ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                    tensor_name)
        if 'detection_masks' in tensor_dict:
            # The following processing is only for single image
            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
            detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
            # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
            real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
            detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
            detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, image.shape[0], image.shape[1])
            detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
            # Follow the convention by adding back the batch dimension
            tensor_dict['detection_masks'] = tf.expand_dims(
                detection_masks_reframed, 0)
        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

        # Run inference
        output_dict = sess.run(tensor_dict,
                                feed_dict={image_tensor: np.expand_dims(image, 0)})

        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict[
            'detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

# Verify if the TensorFlow is using CPU / GPU
#tf.debugging.set_log_device_placement(True)

parser = argparse.ArgumentParser(description='Simple object detection inference for videos.')
parser.add_argument('-v','--video', help='The video file path.')
parser.add_argument('-o','--out_video', default='output.mp4', help='The output video file.')
parser.add_argument('-m','--model', default='ssd_mobilenet_v2_coco', help='The model name. Use only coco trained models. Download from: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md')
args = parser.parse_args()

# Path to frozen detection graph. This is the actual model that is used for the object detection.
# If you wish to import a model from the path different from below, you may have to change the code below
PATH_TO_FROZEN_GRAPH = os.path.join('/home', 'ubuntu', 'models', 'tf1', args.model, 'frozen_inference_graph.pb')

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), 'data', 'mscoco_label_map.pbtxt')

# I/O file path
VIDEO_PATH = args.video
VIDEO_OUT_PATH = args.out_video

# (Optional) Enable GPU acceleration
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0000:00:1e.0"

config = tf.ConfigProto(device_count={"GPU": 0})
config.gpu_options.allow_growth = True

# Disable the eager execution
tf.disable_eager_execution()

# Load a (frozen) Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.io.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')
	
# Load a label map that maps label indexes to category names
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# Process video
cap = cv2.VideoCapture(VIDEO_PATH)
num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
FRAME_RATE = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
# Modify the next line if you want to process different codecs
out = cv2.VideoWriter(
    filename=VIDEO_OUT_PATH,
    fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
    fps=float(FRAME_RATE), 
    frameSize=(int(cap.get(3)),int(cap.get(4))),
    isColor=True,
)

dt_string = datetime.now().strftime("%H:%M:%S, %d-%m-%Y")
print("Inference started; Total number of frames = {}; started at {}".format(
    num_frames,
    dt_string,
    ))

start_time = time.time()
curr_frame = 0

while(cap.isOpened()):
  start_time_curr_frame = time.time()
  ret, frame = cap.read()
  
  if ret == True:
    image_np = load_image_into_numpy_array(frame)
    image_np_expanded = np.expand_dims(image_np, axis=0)
    output_dict = run_inference_for_single_image(image_np, detection_graph, config)
    vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      line_thickness=2)
    #output_image = Image.fromarray(image_np)
    out.write(image_np)

    print("Frame #{} processed; elapsed time = {:.2f}, remainder = {}".format(
        curr_frame,
        time.time() - start_time_curr_frame,
        int(num_frames - curr_frame - 1),
    ))
    curr_frame = curr_frame + 1

  else:
    break

print("Inference done. Elapsed time = {:.2f}".format(
    time.time() - start_time,
    ))
	  
# Release everything if job is finished
print("Releasing resources' handle...")
cap.release()
out.release()
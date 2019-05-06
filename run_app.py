
import grpc
import tensorflow as tf

#import run_multilabels_classifier as classifiers


from run_multilabels_classifier import MultiLabelTextProcessor 
from run_multilabels_classifier import convert_single_example 
from run_multilabels_classifier import create_int_feature


import tokenization
import collections

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import types_pb2

from flask import Flask
from flask import request

import logging

import random

app = Flask(__name__)


@app.route("/", methods = ['GET'])
def hello():
  return "Hello BERT predicting Toxic Comments! Try posting a string to this url"


@app.route("/", methods = ['POST'])
def predict():
  # MODEL PARAMS
  max_seq_length = 128

  channel = grpc.insecure_channel("bert-toxic:8500")
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

  # Parse Description
  tokenizer = tokenization.FullTokenizer(
    vocab_file="asset/vocab.txt", do_lower_case=True)
  processor = MultiLabelTextProcessor()
  label_list = [0,0,0,0,0,0]




  # logging.basicConfig(level=logging.DEBUG,format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
  # logger = logging.getLogger(__name__)
  # logger.info('getting json')
  content = request.get_json()
  #logger.info('JSON: {}'.format(content))
  request_id = str(random.randint(1, 9223372036854775807))


  inputExample = processor.serving_create_example([request_id, content['description']], 'test')
  feature = convert_single_example(0, inputExample, label_list, max_seq_length, tokenizer)
  
  features = collections.OrderedDict()
  features["input_ids"] = create_int_feature(feature.input_ids)
  features["input_mask"] = create_int_feature(feature.input_mask)
  features["segment_ids"] = create_int_feature(feature.segment_ids)
  features["is_real_example"] = create_int_feature([int(feature.is_real_example)])
  if isinstance(feature.label_id, list):
    label_ids = feature.label_id
  else:
    label_ids = [feature.label_id]
  features["label_ids"] = create_int_feature(label_ids)

  tf_example = tf.train.Example(features=tf.train.Features(feature=features))
  
  
  model_input = tf_example.SerializeToString()

  # Send request
  # See prediction_service.proto for gRPC request/response details.
  model_request = predict_pb2.PredictRequest()
  model_request.model_spec.name = 'bert'
  model_request.model_spec.signature_name = 'serving_default'
  dims = [tensor_shape_pb2.TensorShapeProto.Dim(size=1)]
  tensor_shape_proto = tensor_shape_pb2.TensorShapeProto(dim=dims)
  tensor_proto = tensor_pb2.TensorProto(
    dtype=types_pb2.DT_STRING,
    tensor_shape=tensor_shape_proto,
    string_val=[model_input])

  model_request.inputs['examples'].CopyFrom(tensor_proto)
  result = stub.Predict(model_request, 10.0)  # 10 secs timeout
  result = tf.make_ndarray(result.outputs["probabilities"])
  pretty_result = "Predicted Label: " + label_list[result[0]]
  app.logger.info("Predicted Label: %s", label_list[result[0]])
  return pretty_result


if __name__ == '__main__':
  app.run(debug=True, host='0.0.0.0')

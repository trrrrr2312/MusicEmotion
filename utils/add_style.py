import tensorflow_hub as hub
import cv2
from PIL import Image
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import IPython.display as display
import os
import tensorflow as tf
from torchvision import transforms
import json
from utils.models import get_pretrained_mobile_net
from utils.config import config
import torch

# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

hub_model = hub.load(
    'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# define a function to load pictures, limited to 512 pixels
def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor) > 3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return Image.fromarray(tensor)

def style_transfer(output_path, style_images, content_image):
    i=0
    sentiments = []
    for style in style_images:
        stylized_image = hub_model(tf.constant(
            content_image), tf.constant(style))[0]
        output = tensor_to_image(stylized_image)
        output = np.array(output)
        cv2.imwrite(output_path + '_stylized'+ str(i)+ '.jpeg', output)
        img = Image.open(output_path + '_stylized'+ str(i)+ '.jpeg')
        img = img.convert('RGB')

        convert_tensor = transforms.ToTensor()
        img = convert_tensor(img)

        sentiment = style_eval([img], output_path)
        sentiments.append(sentiment[0])
        # style_eval(output, output_path)
        i=i+1
    _write_json_metadata(sentiments, output_path)

# evaluate valence and arousal value of stylized image
def style_eval(styled_output, output_path):
    image_sentiment_model = "./models/image_sentiment.model"
    sentiment_model = get_pretrained_mobile_net()
    sentiment_model.to(config['device'])
    sentiment_model.load_state_dict(torch.load(image_sentiment_model))
    

    # Feed-forward image sentiment analysis
    eval_images = torch.stack(styled_output).to(config['device'])
    sentiment = sentiment_model(eval_images)
    return sentiment

def _write_json_metadata(predictions, output_path):
        metadata = []
        # print(len(predictions))
        for i in range(len(predictions)):
                metadata.append({
                                 'image_arousal': float(predictions[i][0]),
                                 'image_valence': float(predictions[i][1]),
                                 })

        with open(output_path + '_stylized.json', "w") as f:
            json.dump(metadata, f)
import tensorflow_hub as hub
from os.path import join
import cv2
import csv
from PIL import Image
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from torchvision import transforms
from utils.models import get_pretrained_mobile_net
from utils.config import config
import torch
import pandas as pd

# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

hub_model = hub.load(
    'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
cur_transforms = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(), normalize,])

style_path0 = 'painting_styles/Fauvismwoman-with-hat-1905.jpg'
style_path1 = 'painting_styles/Fauvism-Self-portrait in studio-André_Derain.jpeg'
style_path2 = 'painting_styles/Pointilism-Paul Signac - The Pine Tree at St Tropez 1909  - (MeisterDrucke-71791).jpeg'
style_path3 = 'painting_styles/ink and wash-approaching-yanbian.jpeg'
style_path4 = 'painting_styles/Digital art-freefall-19.png!Large.jpeg'
style_path5 = 'painting_styles/Sketch-head-of-an-angel-1506.jpg'
style_path6 = 'painting_styles/Sketch-study.jpg'
style_path7 = 'painting_styles/Barloque-assumption-of-the-virgin-mary-1601.jpg!Large.jpg'
style_path8 = 'painting_styles/Impression-VanGogh-starry_night.jpeg'
style_path9 = 'painting_styles/Impression-Claude_Monet, soleil_levant.jpeg'
style_path10 = 'painting_styles/Gongbi-saying-farewell-at-xunyang-detail.jpeg'



filepaths = []
labels = []
names = []
eval_name = []
valence = []
val_change = []
arousal = []   
aro_change = []
   
def load_img(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img


def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor) > 3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return Image.fromarray(tensor)


def imshow(image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  if title:
    plt.title(title)
def stylize_img(output_path, style_images, content_image, img_origin, music_sent, idx):
    i = 0
    sentiments = []
    for style in style_images:
        stylized_image = hub_model(tf.constant(
            content_image), tf.constant(style))[0]
        output = tensor_to_image(stylized_image)
        output = np.array(output)
        eval_name.append('styled_' + str(i) + '.jpeg')
        cv2.imwrite(output_path + '_styled_' + str(i) + '.jpeg', output)
        img = Image.open(output_path + '_styled_' + str(i) + '.jpeg')
        img = img.convert('RGB')
        convert_tensor = transforms.ToTensor()
        img = convert_tensor(img)

        sentiment = style_eval(img)
        sentiments.append(sentiment[0])
        # style_eval(output, output_path)
        i = i+1
    _write_json_metadata(sentiments, output_path, img_origin, music_sent, idx)

# evaluate valence and arousal value of stylized image
def style_eval(img):
    image_sentiment_model = "./models/image_sentiment.model"
    sentiment_model = get_pretrained_mobile_net()
    sentiment_model.to(config['device'])
    sentiment_model.load_state_dict(torch.load(image_sentiment_model))

    # img = cur_transforms(img)
    # Feed-forward image sentiment analysis
    eval_images = torch.stack([img]).to(config['device'])
    sentiment = sentiment_model(eval_images)
    return sentiment


def _write_json_metadata(predictions, output_path, img_origin, music_sent, idx):
    # metadata = []
    # original val and aro
    aro = float(img_origin[0]) 
    val = float(img_origin[1])
    music_idx = [idx for i in range(len(eval_name))]

    for i in range(len(predictions)):
        valence.append(float(predictions[i][1]))
        val_change.append(float(predictions[i][1])-val)
        arousal.append(float(predictions[i][0]))
        aro_change.append(float(predictions[i][0])-aro)


    
    dataframe = pd.DataFrame({'music_idx':music_idx, 'img_name':eval_name, 'music_valence':music_sent[1], 'original_valence':val, 'valence':valence, 'val_change':val_change, 'music_arousal': music_sent[1], 'original_arousal':aro, 'arousal':arousal, 'aro_change':aro_change})
    dataframe.to_csv(output_path + "_total.csv",index=False,sep=',')


def stylize_and_eval(output_path, music_sent, idx):
      # image to be transformed
      content_image = load_img(output_path + '_0.jpeg')

      img = Image.open(output_path + '_0.jpeg')
      img = img.convert('RGB')
      convert_tensor = transforms.ToTensor()
      img = convert_tensor(img)
      pred = style_eval(img)
      print(pred[0])
      img_origin = pred[0].tolist()
      # content_image1 = load_img(content_path1)
      # image that provides the style
      style_images = []
      style_images.append(load_img(style_path0))
      style_images.append(load_img(style_path1))
      style_images.append(load_img(style_path2))
      style_images.append(load_img(style_path3))
      style_images.append(load_img(style_path4))
      style_images.append(load_img(style_path5))
      style_images.append(load_img(style_path6))
      style_images.append(load_img(style_path7))
      style_images.append(load_img(style_path8))
      style_images.append(load_img(style_path9))
      style_images.append(load_img(style_path10))
      stylize_img(output_path, style_images, content_image, img_origin, music_sent, idx)

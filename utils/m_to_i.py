from pytorch_pretrained_biggan import BigGAN
from utils.gan_utils import feedforward_gan
from time import time
import cv2
import json
import numpy as np
from utils.config import config
from os.path import join

class ImageGenerator:

    def __init__(self, sound_model, translator_model):
        # Model for translating sound into sentiment
        self.sound_model = sound_model

        # Model for translating sentiment into the gan space
        self.translator_model = translator_model

        # Load GAN
        self.gan_model = BigGAN.from_pretrained('biggan-deep-512')
        self.gan_model.to(config['device'])

        # print("deepsing hyper-space synapses loaded!")

    def draw_song(self, audio_path, output_path, subsample=10, smoothing_window=0, debug=True, noise=0.2,
                   batch_size=1, normalize=True, use_transition=False, skip_frames=1):

        start_time = time()

        # Step 1: Extract sentiment from audio file
        sentiment_predictions, power_features = self.sound_model.extract_sentiment(audio_path, smoothing_window)
        # if normalize:
        #     sentiment_predictions = (sentiment_predictions - np.mean(sentiment_predictions, 0))/np.std(sentiment_predictions, 0)
        fe_time = time()
        # print(sentiment_predictions.shape)
        # print(sentiment_predictions)
        # Step 2: Translate the sentiment into the gan space and generate the images
        if use_transition:
            images, class_vectors = self._generate_gan_images(sentiment_predictions, debug=debug, noise=noise, batch_size=batch_size,
                                           subsample=subsample, output_path=output_path, power_features=power_features, skip_frames=skip_frames)
        else:
            images, class_vectors = self._generate_gan_images(sentiment_predictions, debug=debug, noise=noise,
                                           batch_size=batch_size, subsample=subsample, output_path=output_path,
                                                              power_features=None)

        # Save 5 shots
        print(len(images))
        for i in range(len(images)):
            cv2.imwrite(output_path + '_' + str(i) + '.jpeg', images[i])

        cg_time = time()

        print("Total time %3.2f s, feature extraction: %3.2f s, content generation: %3.2f s"
              % (cg_time - start_time, fe_time - start_time, cg_time - fe_time))

        return sentiment_predictions, class_vectors

    def _generate_gan_images(self, sentiment_predictions, noise=0.01, batch_size=4, debug=True, subsample=1,
                                output_path=None, power_features=None, skip_frames=1):

        # Step 1: Translate the sentiment space into the GAN space
        # 将sentiment value输入模型， 生成class和noice特征
        class_vectors, noise_vectors, song_words = self.translator_model.translate_sentiment(sentiment_predictions,
                                                                                             noise=noise)
      
        if power_features is None:
            pass
        else:
            new_class_vectors = []
            new_noise_vectors = []
            new_song_words = []
            new_sentiment_predictions = []

            # 下列循环中sentiment对图像生成没有影响（只是为了存档）
            for i in range(len(class_vectors)-1):
                factor = 0
                normalized_power = power_features[i*subsample:(i+1)*subsample] + 1e-6
                # normalized_power = np.exp(np.float64(normalized_power*2))
                normalized_power = normalized_power / np.sum(normalized_power)

                print(normalized_power)

                for j in range(0, subsample, skip_frames):
                    new_class_vectors.append(class_vectors[i])
                    # new_class_vectors.append((1-factor)*class_vectors[i] + factor*class_vectors[i+1])
                    new_noise_vectors.append((1-factor)*noise_vectors[i] + factor*noise_vectors[i+1])
                    new_sentiment_predictions.append(sentiment_predictions[i])
                    new_song_words.append(song_words[i])
                    factor +=normalized_power[j]

            for j in range(0, subsample+1, skip_frames):
                new_class_vectors.append(class_vectors[-1])
                new_noise_vectors.append(noise_vectors[-1])
                new_song_words.append(song_words[-1])
                new_sentiment_predictions.append(sentiment_predictions[-1])

            class_vectors = np.float32(new_class_vectors)
            noise_vectors = np.float32(new_noise_vectors)
            sentiment_predictions = np.float32(new_sentiment_predictions)
            song_words = new_song_words

        # Step 2: Employ GAN to generate the images
        images = feedforward_gan(self.gan_model, class_vectors, noise_vectors, batch_size, noise)

        # Step 6: Write metadata
        if power_features is not None:
            # self._write_json_metadata(sentiment_predictions, song_words, 1, output_path)
            self._write_json_metadata(sentiment_predictions, song_words, 1, output_path)
        else:
            # self._write_json_metadata(sentiment_predictions, song_words, subsample, output_path)
            self._write_json_metadata(sentiment_predictions, song_words, subsample, output_path)
        return images, class_vectors


    def _write_json_metadata(self, predictions, song_words, subsample, output_path):

        metadata = []
        for i in range(len(predictions)):
                metadata.append({'time': i * subsample * self.sound_model.period,
                                 'sound_arousal': float(predictions[i][0]),
                                 'sound_valence': float(predictions[i][1]),
                                 'gan_class': song_words[i],
                                 })

        with open(output_path + '.json', "w") as f:
            json.dump(metadata, f)
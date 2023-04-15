from utils.sound_sentiment import SoundSentimentExtractor
import sys
from os.path import join
from utils.m_to_i import ImageGenerator
from utils.textual_translator import TextualSentimentTranslator
from utils.neural_translator import NeuralSentimentTranslator
from utils.config import config
import numpy as np
import argparse
import warnings
warnings.filterwarnings('ignore')
sys.path.append('.')

from style_eval import stylize_and_eval


parser = argparse.ArgumentParser(description='deepsing: music-to-image translator')

parser.add_argument('input', type=str, help='song to translate')
parser.add_argument('--id', type=str, help='song id')
parser.add_argument('output', type=str, help='path to save the song (please do not provide any suffix)')
parser.add_argument('--translator', type=str, help='translator to use',
                    default="neural")
parser.add_argument('--path', type=str, help='path to translator models',
                    default=join(config['basedir'], 'models/neural_translator_'))
parser.add_argument('--nid', type=int, help='id of the translator model, if -1 a random model is chosen',
                    default=-1)
parser.add_argument('--dictionary', type=str,
                    help='json dictionary with imagenet classes (only used with textual translators)',
                    default=join(config['basedir'], 'models/imagenet_clean.txt'))
parser.add_argument('--raw', help='perform song-basis normalization', action='store_true')

args = parser.parse_args()

if args.translator == 'neural':

    if args.nid < 0:
        id = np.random.randint(0, 23, 1)[0]
        print("Translator selected: ", id)
    else:
        id = args.nid

    args.path = args.path + str(id) + '.model'
    translator = NeuralSentimentTranslator(translator_path=args.path)
elif args.translator == 'textual':
    translator = TextualSentimentTranslator(k_neighbors=5, class_dictionary_path=args.dictionary)
else:
    print("Unknown translator! Supported translators: 'neural', 'textual'")

sound_sentiment = SoundSentimentExtractor()


generator = ImageGenerator(
    sound_sentiment, translator)
music_sentiment, _ =  generator.draw_song(args.input, args.output, smoothing_window=1, noise=0.1, subsample=args.subsample,
                    normalize=not args.raw, debug=not args.noinfo, use_transition=args.dynamic)

# print("--------------------start stylization--------------------")

final = stylize_and_eval(args.output, music_sentiment[0], args.id)


# print("End stylization.")


# -------------------------style tranfer-------------------------

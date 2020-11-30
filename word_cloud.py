
from visualizations import *
from fastcore.script import *
from text import read_text
import os.path

DEBUG = True

@call_parse
def wc(language:Param("language code (i.e, 'en' for 'english').", str),
	   mask_path:Param("path for image to use as mask. ", str)=None,
	   save_path:Param("save image to path.", str)=None, 
	   max_font_size:Param("maximum font size for word cloud contents.", int)=50, 
	   use_mask:Param("use mask if specified.", store_true)=False,
	   max_words:Param("max words in the cloud.", int)=1000, 
	   background_color:Param("the color of image background.", str)="white", 
	   interpolation:Param("interpolation for image show.", str)='bilinear',
	   mask_type:Param("mask type: color or shape", str)="shape", 
       contour_width:Param("contour width, ignored when using color as mask_type.", int)=3, 
       contour_color:Param("contour color, ignored when using color as mask_type.", str)='firebrick'):

	"create word cloud from text."

		# usage: word_cloud.py [-h] [--mask_path MASK_PATH] [--save_path SAVE_PATH]
		#                      [--max_font_size MAX_FONT_SIZE] [--use_mask]
		#                      [--max_words MAX_WORDS]
		#                      [--background_color BACKGROUND_COLOR]
		#                      [--interpolation INTERPOLATION] [--pdb] [--xtra XTRA]
		#                      language

		# create word cloud from text.

		# positional arguments:
		#   language              language code (i.e, 'en' for 'english').

		# optional arguments:
		#   -h, --help            show this help message and exit
		#   --mask_path MASK_PATH
		#   --save_path SAVE_PATH
		#   --max_font_size MAX_FONT_SIZE
		#                         (default: 50)
		#   --use_mask            (default: False)
		#   --max_words MAX_WORDS
		#                         (default: 100)
		#   --background_color BACKGROUND_COLOR
		#                         (default: white)
		#   --interpolation INTERPOLATION
		#                         (default: bilinear)
		#   --pdb                 Run in pdb debugger (default: False)
		#   --xtra XTRA           Parse for additional args (default: '')
	if not language in labels and language != "all": exit(-1)

	if language == "all":
		for l in labels:
			if use_mask and mask_path is None: mask_path = f"./img/flags/{l}.png"
			text = read_text(l)
			language = languages[labels.index(l)]
			save_path = f'./img/wordclouds/{language}.png'
			word_cloud(text, language, mask_path, save_path, 
			   		   max_font_size, max_words, background_color, 
			   		   interpolation, mask_type, contour_width, 
			   		   contour_color)
			mask_path=None
			if DEBUG: print(f"word cloud for {language} created, and saved to {save_path}.")
	else:
		if use_mask and mask_path is None: mask_path = f"./img/flags/{language}.png"
		text = read_text(language)
		language = languages[labels.index(language)]
		if save_path is None: save_path = f'./img/wordclouds/{language}.png'
		word_cloud(text, language, mask_path, save_path, 
				   max_font_size, max_words, background_color, 
				   interpolation, mask_type, contour_width, 
			   	   contour_color)
		if DEBUG: print(f"word cloud for {language} created, and saved to {save_path}.")


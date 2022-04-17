#%%
import os
import math
import numpy as np
# %%
import cv2

# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter("./video.MP4", fourcc, 30.0, (1080, 1920))
# %%
import face_recognition
from PIL import Image, ImageDraw

# %%
def JudgeBrightness(image):
	sum = 0
	count = 0
	for i in image:
		for j in i:
			sum = sum + j
			count = count + 1
	return sum / count

# %%
def CalGamma(average):
	return math.log(0.5, average / 255)

# %%
def GammaTransformation(image, gamma):
	image_cp = np.copy(image)
	output_imgae = 255 * np.power(image_cp.astype(int) / 255, gamma)
	return output_imgae

# %%
def FaceRecognize(know_im, imagefirst):
    # know_im = face_recognition.load_image_file("./image/image_true/doggy_true.jpg")
    know_encodings = face_recognition.face_encodings(know_im)
    first_encodings = face_recognition.face_encodings(imagefirst)
    global numofall
    numofall = numofall + 1
    if len(know_encodings) == 0:
        global numoftrue
        numoftrue = numoftrue + 1
        return 'Error, no person in know_encoding'
    elif len(first_encodings) == 0:
        global numofimage
        numofimage = numofimage + 1
        return 'Error, no person in this image'
    else:
        return face_recognition.compare_faces([know_encodings[0]], first_encodings[0])[0]

# %%
try:
    know_im = face_recognition.load_image_file('./image/photo.png')
    imagefirst = face_recognition.load_image_file('./image/request_photo.jpeg')
except Exception as e:
    print(e)
else:
    gray = cv2.cvtColor(imagefirst, cv2.COLOR_BGR2GRAY)
    gamma = CalGamma(JudgeBrightness(gray))
    img_gamma = GammaTransformation(imagefirst, gamma)
    print(str(FaceRecognize(know_im, np.uint8(img_gamma))))

# %%
try:
    frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
except Exception as e:
	print(e)

# %%
try:
    pil_image = Image.fromarray(frame)
    img = cv2.cvtColor(np.asarray(pil_image),cv2.COLOR_RGB2BGR)
    # pil_image = Image.fromarray(img)
    # p
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gamma = CalGamma(JudgeBrightness(gray))
    frame = GammaTransformat
    ImageShow(np.uint8(frame))
    print(FaceRecognize(know_im, np.uint8(frame)))
except Exception as e:
    print(e)
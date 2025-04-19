
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import cv2
import random
import numpy as np
from PIL import Image
import math
import sys
from glob import glob
from tqdm import tqdm
import imageio.v2 as imageio

from model import model
from argparse import ArgumentParser

conf = tf.compat.v1.ConfigProto()
conf.gpu_options.allow_growth = True
tf.compat.v1.reset_default_graph()

# Input image
inputa = tf.placeholder(tf.float32, name='inputa')
labela = tf.placeholder(tf.float32, name='labela')

mask_model = model.MaskNet()
inputa_resize = tf.compat.v1.image.resize_bicubic(inputa, [tf.shape(labela)[1], tf.shape(labela)[2]], align_corners=False, name="resize_input")
mask_model.forward(tf.concat([inputa_resize, labela], axis=3))
W = mask_model.output

# Parameter variables
PARAM = model.Weights(scope='MODEL')
weights = PARAM.weights
MODEL = model.MODEL(name='MODEL')

# Graph build
MODEL.forward(inputa, weights)
output = MODEL.output

loss = tf.reduce_mean(W * tf.abs(labela - output))

var_list_sr = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='MODEL')
loader_sr = tf.train.Saver(var_list=var_list_sr)

var_list_mask = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Mask')
loader_mask = tf.train.Saver(var_list=var_list_mask)

opt = tf.train.GradientDescentOptimizer(0.01)
grad = opt.compute_gradients(loss, var_list_sr)
metatrain_op = opt.apply_gradients(grad)

init = tf.global_variables_initializer()

parser = ArgumentParser()
parser.add_argument('--input_dir', type=str, dest='input_dir', default='./datasets/RealFaces200/LQ')
parser.add_argument('--face_dir', type=str, dest='face_dir', default='./datasets/RealFaces200/')
parser.add_argument('--output_dir', type=str, dest='output_dir', default='./datasets/RealFaces200/Res')
parser.add_argument('--model_dir', type=str, dest='model_dir', default='./weights/Model1/model-1200')
parser.add_argument('--patch_size', type=int, dest='patch_size', default=64)
parser.add_argument('--patch_num_per_img', type=int, dest='patch_num_per_img', default=16)
parser.add_argument('--fine_tune_num', type=int, dest='fine_tune_num', default=1)
args = parser.parse_args()

with tf.Session(config=conf) as sess:
    sess.run(init)

    for img_path in tqdm(sorted(glob(args.input_dir + '/' + '*'))):
        img_name = img_path.split('/')[-1][:-4]

        # Debugging: Check if image is found
        print(f"Processing image: {img_name}")

        # Restore model checkpoints
        loader_sr.restore(sess, args.model_dir)
        loader_mask.restore(sess, args.model_dir)

        label_a = []
        input_a = []
        input_b = imageio.imread(img_path)  # BGR
        
        for img_lq_path, img_hq_path in zip(sorted(glob(args.face_dir + '/Face_LQ/' + img_name + '*')), 
                                            sorted(glob(args.face_dir + '/Face_HQ/' + img_name + '*'))):
            print(f"Low-quality image: {img_lq_path}, High-quality image: {img_hq_path}")

            whole_input_a = imageio.imread(img_lq_path)
            whole_label_a = imageio.imread(img_hq_path)

            h_crop, w_crop = whole_input_a.shape[:2]
            print(f"Image dimensions: {h_crop}x{w_crop}")

            # Check image size for patch extraction
            if h_crop <= args.patch_size // 4 or w_crop <= args.patch_size // 4:
                if len(input_a) == 0:
                    input_a.append(whole_input_a)
                    label_a.append(whole_label_a)
                    break
                else:
                    print(f"Skipping small image: {img_lq_path}")
                    continue

            for j in range(args.patch_num_per_img):
                face_crop_h = random.randrange(0, h_crop - args.patch_size // 4)
                face_crop_w = random.randrange(0, w_crop - args.patch_size // 4)
                face_patch_hr = whole_label_a[face_crop_h*4:face_crop_h*4 + args.patch_size, face_crop_w*4:face_crop_w*4 + args.patch_size, :]
                face_patch_lr = whole_input_a[face_crop_h:face_crop_h + args.patch_size // 4, face_crop_w:face_crop_w + args.patch_size // 4, :]

                # Check if the extracted patches have the correct shape
                if face_patch_lr.shape == (args.patch_size // 4, args.patch_size // 4, 3) and face_patch_hr.shape == (args.patch_size, args.patch_size, 3):
                    input_a.append(face_patch_lr)
                    label_a.append(face_patch_hr)
                else:
                    print(f"Skipping patch with mismatched shape: LR {face_patch_lr.shape}, HR {face_patch_hr.shape}")

        # Debugging: Check how many patches were added
        print(f"Number of patches for {img_name}: {len(input_a)}")

        # If no patches were extracted, skip this image
        if len(input_a) == 0:
            print(f"No valid patches found for {img_name}, skipping...")
            continue

        input_a = np.stack(input_a, axis=0) / 255.  # BGR
        label_a = np.stack(label_a, axis=0) / 255.  # BGR
        feed_dict = {inputa: input_a, labela: label_a}

        input_b = input_b[None, :, :, :] / 255.
        feed_dict_b = {inputa: input_b, labela: None}

        for i in range(args.fine_tune_num):
            sess.run(metatrain_op, feed_dict=feed_dict)

        output_ft = sess.run(output, feed_dict=feed_dict_b)
        output_image_path = os.path.join(args.output_dir, img_name + '.png')
        imageio.imsave(output_image_path, np.round(np.clip(np.squeeze(output_ft), 0., 1.) * 255).astype(np.uint8))

        print(f"Output saved to: {output_image_path}")


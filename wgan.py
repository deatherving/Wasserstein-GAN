# -*- coding: utf-8 -*-
# author: K

import numpy as np
import tensorflow as tf
from argparse import ArgumentParser
from PIL import Image
from os import walk
from os.path import join


'''

# The Generator and Discrminator Using Batch Norm with or without tanh

class Generator:
	def __init__(self, depths = [1024, 512, 256, 128], f_size = 8):
		self.reuse = False
		self.f_size = f_size
		# add 3 for RGB channel
		self.depths = depths + [3]

	def build_graph(self, inputs):

		input_depths = self.depths[0:4]
		output_depths = self.depths[1:5]

		out = []
		with tf.variable_scope("generator", reuse = self.reuse):
			inputs = tf.convert_to_tensor(inputs)
			with tf.variable_scope('full_connect'):
				w0 = tf.get_variable(
					'w',
					[inputs.get_shape()[-1], input_depths[0] * self.f_size * self.f_size],
					tf.float32,
					tf.truncated_normal_initializer(stddev = 0.02)

				)

				b0 = tf.get_variable(
					'b',
					[input_depths[0] * self.f_size * self.f_size],
					tf.float32,
					tf.zeros_initializer())
				

				fc = tf.matmul(inputs, w0) + b0

				#fc = tf.matmul(inputs, w0)

				reshaped = tf.reshape(fc, [-1, self.f_size, self.f_size, input_depths[0]])

				beta = tf.get_variable(
					'beta',
					[input_depths[0]],
					tf.float32,
					tf.zeros_initializer())

				# set axes to 0 ,1 ,2 for global normalization
				mean, variance = tf.nn.moments(reshaped, [0, 1, 2])

				bn = tf.nn.batch_normalization(reshaped, mean, variance, beta, None, 1e-5)

				outputs = tf.nn.relu(bn)

				out.append(outputs)
	
			# 4 is the designed network structure of paper, need to be modified
			for i in range(4):
				with tf.variable_scope('trans_conv%d' % (i + 1)):
					w = tf.get_variable(
						'w',
						[5, 5, output_depths[i], input_depths[i]],
						tf.float32,
						tf.truncated_normal_initializer(stddev = 0.02)
					)
					b = tf.get_variable(
						'b',
						[output_depths[i]],
						tf.float32,
						tf.zeros_initializer())
					
					beta = tf.get_variable(
						'beta',
						[output_depths[i]],
						tf.float32,
						tf.zeros_initializer())
					
					trans_conv = tf.nn.conv2d_transpose(
						outputs,
						w,
						[int(outputs.get_shape()[0]), self.f_size * 2 ** (i + 1), self.f_size * 2 ** (i + 1), output_depths[i]],
						[1, 2, 2, 1]
					)
					
					# need to be modified for scalability
					if i < 3:
						mean, variance = tf.nn.moments(trans_conv, [0, 1, 2])
						bn = tf.nn.batch_normalization(trans_conv, mean, variance, beta, None, 1e-5)
						outputs = tf.nn.relu(bn)
					else:
						#outputs = tf.nn.tanh(tf.nn.bias_add(trans_conv, b))
						outputs = trans_conv
					out.append(outputs)


		self.reuse = True
		self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'generator')

		return out

	def __call__(self, inputs):
		return self.build_graph(inputs)		


class Discriminator:
	def __init__(self, depths = [64, 128, 256, 512]):
		self.reuse = False
		# the input tensor of images must be 3 dimensional
		self.depths = [3] + depths
	def build_graph(self, inputs):
		input_depth = self.depths[0:4]
		output_depth = self.depths[1:5]

		out = []

		with tf.variable_scope('discriminator', reuse = self.reuse):
			outputs = inputs
			for i in range(4):
				with tf.variable_scope("conv%d" % (i + 1)):
					w = tf.get_variable(
						'w',
						[5, 5, input_depth[i], output_depth[i]],
						tf.float32,
						tf.truncated_normal_initializer(stddev = 0.02)
					)

					beta = tf.get_variable(
						'beta',
						[output_depth[i]],
						tf.float32,
						tf.zeros_initializer())

					conv = tf.nn.conv2d(outputs, w, [1 ,2 ,2 ,1], 'SAME')
					mean, variance = tf.nn.moments(conv, [0, 1, 2])
					bn = tf.nn.batch_normalization(conv, mean, variance, beta, None, 1e-5)
					outputs = self.leaky_relu(bn)

					out.append(outputs)

			with tf.variable_scope("classify"):
				tensor_shape = outputs.get_shape().as_list()

				flatten_dim = tensor_shape[1] * tensor_shape[2] * tensor_shape[3]

				w = tf.get_variable(
					'w',
					[flatten_dim, 1],
					tf.float32,
					tf.truncated_normal_initializer(stddev = 0.02)
				)

				b = tf.get_variable(
					'b',
					[1],
					tf.float32,
					tf.zeros_initializer()
				)

				print outputs.get_shape()

				outputs = tf.reshape(outputs, [-1, flatten_dim])

				print outputs.get_shape()

				h_outputs = tf.nn.bias_add(tf.matmul(outputs, w), b)

				print h_outputs.get_shape()
	
				out.append(h_outputs)				

		self.reuse = True
		self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'discriminator')
		return out



	def leaky_relu(self, x, alpha = 0.2):
		return tf.maximum(x, x * alpha)

	def __call__(self, inputs):
		return self.build_graph(inputs)

'''

class Generator:
	def __init__(self, depths = [1024, 512, 256, 128], f_size = 8):
		self.reuse = False
		self.f_size = f_size
		# add 3 for RGB channel
		self.depths = depths + [3]

	def build_graph(self, inputs):

		input_depths = self.depths[0:4]
		output_depths = self.depths[1:5]

		out = []
		with tf.variable_scope("generator", reuse = self.reuse):
			inputs = tf.convert_to_tensor(inputs)
			with tf.variable_scope('full_connect'):
				w0 = tf.get_variable(
					'w',
					[inputs.get_shape()[-1], input_depths[0] * self.f_size * self.f_size],
					tf.float32,
					tf.truncated_normal_initializer(stddev = 0.02)

				)

				b0 = tf.get_variable(
					'b',
					[input_depths[0] * self.f_size * self.f_size],
					tf.float32,
					tf.zeros_initializer())
				

				fc = tf.matmul(inputs, w0) + b0

				#fc = tf.matmul(inputs, w0)

				reshaped = tf.reshape(fc, [-1, self.f_size, self.f_size, input_depths[0]])

				outputs = tf.nn.relu(reshaped)

				out.append(outputs)
	
			# 4 is the designed network structure of paper, need to be modified
			for i in range(4):
				with tf.variable_scope('trans_conv%d' % (i + 1)):
					w = tf.get_variable(
						'w',
						[5, 5, output_depths[i], input_depths[i]],
						tf.float32,
						tf.truncated_normal_initializer(stddev = 0.02)
					)


					b = tf.get_variable(
						'b',
						[output_depths[i]],
						tf.float32,
						tf.zeros_initializer()
					)
					
					trans_conv = tf.nn.conv2d_transpose(
						outputs,
						w,
						[int(outputs.get_shape()[0]), self.f_size * 2 ** (i + 1), self.f_size * 2 ** (i + 1), output_depths[i]],
						[1, 2, 2, 1]
					)

					
					# need to be modified for scalability
					if i < 3:
						outputs = tf.nn.relu(trans_conv)
					else:
						#outputs = tf.nn.tanh(tf.nn.bias_add(trans_conv, b))
						outputs = trans_conv

					out.append(outputs)

		self.reuse = True
		self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'generator')

		return out

	def __call__(self, inputs):
		return self.build_graph(inputs)		


class Discriminator:
	def __init__(self, depths = [64, 128, 256, 512]):
		self.reuse = False
		# the input tensor of images must be 3 dimensional
		self.depths = [3] + depths
	def build_graph(self, inputs):
		input_depth = self.depths[0:4]
		output_depth = self.depths[1:5]

		out = []

		with tf.variable_scope('discriminator', reuse = self.reuse):
			outputs = inputs
			for i in range(4):
				with tf.variable_scope("conv%d" % (i + 1)):
					w = tf.get_variable(
						'w',
						[5, 5, input_depth[i], output_depth[i]],
						tf.float32,
						tf.truncated_normal_initializer(stddev = 0.02)
					)
				
					conv = tf.nn.conv2d(outputs, w, [1 ,2 ,2 ,1], 'SAME')

					outputs = self.leaky_relu(conv)

					out.append(outputs)

			with tf.variable_scope("classify"):
				tensor_shape = outputs.get_shape().as_list()

				flatten_dim = tensor_shape[1] * tensor_shape[2] * tensor_shape[3]

				print tensor_shape
				

				w = tf.get_variable(
					'w',
					[flatten_dim, 1],
					tf.float32,
					tf.truncated_normal_initializer(stddev = 0.02)
				)

				b = tf.get_variable(
					'b',
					[1],
					tf.float32,
					tf.zeros_initializer()
				)

				#print outputs.get_shape()

				outputs = tf.reshape(outputs, [-1, flatten_dim])

				#print outputs.get_shape()

				h_outputs = tf.nn.bias_add(tf.matmul(outputs, w), b)

				#print h_outputs.get_shape()
	
				out.append(h_outputs)				

		self.reuse = True
		self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'discriminator')
		return out



	def leaky_relu(self, x, alpha = 0.2):
		return tf.maximum(x, x * alpha)

	def __call__(self, inputs):
		return self.build_graph(inputs)

		
class WGAN:
	def __init__(self, batch_size, width = 64, lr = 0.0002):
		self.generator = Generator()
		self.discriminator = Discriminator()
		self.batch_size = batch_size
		self.lr = lr
		self.real_images = tf.placeholder(tf.float32, shape = [self.batch_size, width, width, 3])
	def noice(self, dim):
		return tf.random_uniform([self.batch_size, dim], minval = -1.0, maxval = 1.0)
	def sample_image(self):
		noice = self.noice(100)
		
		gen_image = self.generator(noice)[-1]

		im = tf.cast(tf.add(tf.multiply(gen_image, 127.5), 127.5), tf.uint8)

		return im

	def build_graph(self):
		
		real_images = self.real_images

		noice = self.noice(100)

		g_logits = self.generator(noice)[-1]

		d_logits_fake = self.discriminator(g_logits)[-1]

		d_logits_real = self.discriminator(real_images)[-1]

		d_loss = tf.reduce_mean(d_logits_fake) - tf.reduce_mean(d_logits_real)
			
		wgan_loss = - tf.reduce_mean(d_logits_fake)

		d_optim = tf.train.RMSPropOptimizer(learning_rate = self.lr).minimize(d_loss, var_list = self.discriminator.variables)

		wgan_optim = tf.train.RMSPropOptimizer(learning_rate = self.lr).minimize(wgan_loss, var_list = self.generator.variables)

		weight_clip = [tf.assign(var, tf.clip_by_value(var, -0.01, 0.01)) for var in self.discriminator.variables]

		return d_optim, wgan_optim, weight_clip, d_loss, wgan_loss


def get_args():
	parser = ArgumentParser()
	parser.add_argument('--lr', help = "learning rate")
	args = parser.parse_args()
	return args


def load_img(path, grayscale = False, target_size = None):
	img = Image.open(path)

	if grayscale:
		if img.mode != 'L':
			img = img.convert('L')
	else:
		if img.mode != 'RGB':
			img = img.convert('RGB')

	if target_size:
		wh_tuple = (target_size[1], target_size[0])
		if img.size != wh_tuple:
			img = img.resize(wh_tuple)
	return img

def img_to_array(img):
	x = np.asarray(img, dtype = np.float32)
	return x
	

def read_data(directory, target_size):
	filenames = []

	for root, sub, files in walk(directory):
		for f in files:
			filenames.append(join(root, f))

	imgs = []

	for f in filenames:
		imgs.append(img_to_array(load_img(f, target_size = target_size)))

	imgs = (np.asarray(imgs) - 127.5) / 127.5

	return imgs
	
	
	
def train(input_dir, save_dir,  batch_size = 32, lr = 5e-5, nb_epoch = 300):

	imgs = read_data(input_dir, target_size = [128, 128])

	nb_batches = int(imgs.shape[0] / batch_size)

	nb_samples = imgs.shape[0]

	wgan = WGAN(batch_size = batch_size, width = 128, lr = lr)

	d_optim, wgan_optim, weight_clip, d_loss, wgan_loss = wgan.build_graph()

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(0, nb_epoch + 1):
			print "Training epoch %d" % epoch
			for i in range(nb_batches):
				
				# we could apply several d_optim and then wgan_optim
				
				for j in range(5):

					real_images_batch = imgs[np.random.randint(nb_samples, size = batch_size)]
			
					sess.run(weight_clip)

					sess.run(d_optim, feed_dict = {wgan.real_images: real_images_batch})

				sess.run(wgan_optim)

			if epoch % 5 == 0 and epoch != 0:

				sampled_imgs = sess.run(wgan.sample_image())
		
				idx = 0

				for  each_im in sampled_imgs:

					img = sess.run(tf.image.encode_jpeg(each_im))

					with open(save_dir + "/sample%d_%d.jpeg" % (epoch, idx), "w") as f:
						f.write(img)

					idx += 1

				

if __name__ == '__main__':
	args = get_args()
	train("myself", "samples")

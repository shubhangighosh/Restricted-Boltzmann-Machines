import numpy as np 
import pandas as pd
import os, sys, argparse
import matplotlib.pyplot as plt 
from sklearn.manifold import TSNE
import matplotlib.gridspec as gridspec

#SET RANDOM SEED
np.random.seed(1234)


######################################### HYPERPARAMETERS ##############################################
m = 784 #Number of visible units
n = 100 #Number of hidden units

epochs = 6
batch_size = 1

method = "cd"
k_cd = 2
r = 10

learning_rate = 0.01
momentum = 0.0
weight_decay_factor = 0.0000
persistent = False
########################################################################################################



######################################### LOAD DATA ####################################################
#Load train data
train = pd.read_csv('train.csv')
X_train_original = (train.ix[:,1:-1].values).astype('float32')
labels_train = train.ix[:,-1].values.astype('int32')
X_train = np.where(X_train_original >= 127, 1, 0)

#Load test data
test = pd.read_csv("test.csv")
X_test = (test.ix[:,1:].values).astype('float32')
labels_test = pd.read_csv('test-sol.csv').ix[:,-1].values.astype('int32')
X_test = np.where(X_test >= 127, 1, 0)

#Comment out for complete training
# X_train = X_train[:200]
# labels_train = labels_train[:200]
# X_test = X_test[:10]
# labels_test = labels_test[:10]
#######################################################################################################



##################################### TWO DIMENSIONAL REPRESENTATION ###################################
def t_SNE_plot(d_dim_array, labels = None):

	#Use t-SNE to onvert to 2D
	tsne = TSNE(n_components=2, random_state=1234)
	#numpy.set_printoptions(suppress=True)
	two_dim_array = tsne.fit_transform(d_dim_array)
	print(d_dim_array.shape, two_dim_array.shape)
	
	x_coords = two_dim_array[:, 0]
	y_coords = two_dim_array[:, 1]
	#Display scatter plot
	plt.scatter(x_coords, y_coords, c = labels)
	# plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
	# plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
	plt.show()
########################################################################################################



##################################### PLOT SAMPLED IMAGES ##############################################
#RECONSTRUCT IMAGE
def reconstruct(image):
	prob_h_given_v = sigmoid(image, W, c)
	h_given_v = np.where(np.random.rand(n, 1) < prob_h_given_v, 1.0, 0.0)
	rec_image = sigmoid(h_given_v, W.T, b)

	return rec_image

#PLOT
def plot_images(images):
	gs = gridspec.GridSpec(8, 8)
	gs.update(wspace=0.0025, hspace=0.05)
	
	for k in range(64):

		image = images[k].reshape(28, 28)
		ax = plt.subplot(gs[k/8, k%8])
		imgplot = ax.imshow(image, cmap="binary")
		ax.set_xticks([])
		ax.set_yticks([])

	plt.show()
########################################################################################################



##################################### RBM ##############################################################
#Initialize weights and biases
W = 0.1 * np.random.randn(m, n)
b = np.zeros((m, 1))
c = -4.0 * np.ones((n, 1))

#Empty variables
W_update_prev = np.zeros((m, n))
b_update_prev = np.zeros((m, 1))
c_update_prev = np.zeros((n, 1))

def sigmoid_util(a):
	return np.where(a > 0, 1. / (1. + np.exp(-a)), np.exp(a) / 1+(np.exp(a)))

#SIGMOID FUNCTION
def sigmoid(x, weights, bias):
	pre = np.dot(weights.T, x) + bias
	return sigmoid_util(pre)

v_loop = None
first_it = True

def update_step_cd(v_input, e):
	global W, b, c, W_update_prev, b_update_prev, c_update_prev, v_loop, first_it

	if persistent == False or first_it == True:
		v_loop = v_input
	first_it = False
	

	prob_h = sigmoid(v_input, W, c)
	prob_loop = prob_h

	for k in range(k_cd):

		#Sample h from v
		prob_loop = sigmoid(v_loop, W, c)
		h = np.where(np.random.rand(n, batch_size) < prob_loop, 1, 0)
		
		#Sample v from h
		prob_v = sigmoid(h, W.T, b)
		# v_sampled = np.where(np.random.rand(m, batch_size) < prob_v, 1, 0)
		v_sampled = prob_v
		
		v_loop = v_sampled

	prob_h_final = sigmoid(v_sampled, W, c)

	#Products
	prob_h__v_input = np.dot(prob_h, v_input.T)
	prob_h_final__v = np.dot(prob_h_final, v_sampled.T)
	
	#Updates
	W_update = prob_h__v_input - prob_h_final__v
	b_update = v_input - v_sampled
	c_update = prob_h - prob_h_final

	#momentum = 0.5 if e >= epochs/2 else 0.9

	#Update Weights and Biases
	W += momentum * W_update_prev + learning_rate * W_update.T + weight_decay_factor *  W
	b_update_mean = np.mean(b_update, axis = 1).reshape(-1, 1)
	#print(b_update.shape, b.shape, b_update_mean.shape)
	b += momentum * b_update_prev + learning_rate * b_update_mean
	c_update_mean = np.mean(c_update, axis = 1).reshape(-1, 1)
	#print(c_update.shape, c.shape, c_update_mean.shape)
	c += momentum * c_update_prev + learning_rate * c_update_mean
	
	#For Momentum
	W_update_prev = W_update.T
	b_update_prev = b_update_mean
	c_update_prev = c_update_mean

	return np.sum(np.mean((v_input - v_sampled) ** 2, axis = 1))



def update_step_gibbs(v_input, inner_step):
	gibbs_vs = []
	global W, b, c, W_update_prev, b_update_prev, c_update_prev

	v_loop = np.random.rand(m, batch_size)

	prob_h = sigmoid(v_input, W, c)
	prob_loop = prob_h

	v_sum = np.zeros((m, batch_size))
	prob_h_sum = np.zeros((n, batch_size))
	t = 0
	for k in range(k_cd):

		#Sample h from v
		prob_loop = sigmoid(v_loop, W, c)
		h = np.where(np.random.rand(n, batch_size) < prob_loop, 1.0, 0.0)
		
		#Sample v from h
		prob_v = sigmoid(h, W.T, b)
		# v_sampled = np.where(np.random.rand(m, batch_size) < prob_v, 1.0, 0.0)
		v_sampled = prob_v

		t += 1
		if k_cd - t <= r:
			v_sum += v_sampled
			prob_h_sum += prob_loop
		
		v_loop = v_sampled

		if inner_step == 1 and k % (k_cd/64) == 0:
			gibbs_vs.append(v_sampled[:,0].T)
	
	v_sampled = v_sum/float(r)
	prob_h_final = prob_h_sum/float(r)

	if inner_step == 1:
		plot_images(gibbs_vs)

	#Products
	prob_h__v_input = np.dot(prob_h, v_input.T)
	prob_h_final__v = np.dot(prob_h_final, v_sampled.T)
	
	#Updates
	W_update = prob_h__v_input - prob_h_final__v
	b_update = v_input - v_sampled
	c_update = prob_h - prob_h_final


	#Update Weights and Biases
	W += momentum * W_update_prev + learning_rate * W_update.T
	b_update_mean = np.mean(b_update, axis = 1).reshape(-1, 1)
	b += momentum * b_update_prev + learning_rate * b_update_mean
	c_update_mean = np.mean(c_update, axis = 1).reshape(-1, 1)
	c += momentum * c_update_prev + learning_rate * c_update_mean

	#For Momentum
	W_update_prev = W_update.T
	b_update_prev = b_update_mean
	c_update_prev = c_update_mean

	return np.sum(np.mean((v_input - v_sampled) ** 2, axis = 1))
#########################################################################################################


######################################### PARSE ARGUMENTS ##############################################
parser = argparse.ArgumentParser(description='RBM')
parser.add_argument('--lr', action="store", dest = 'lr', type=float)
parser.add_argument('--batch_size', action="store", dest="batch_size", type=int)
parser.add_argument('--method', action="store", dest="method")
parser.add_argument('--k', action="store", dest="k_cd")
parser.add_argument('--r', action="store", dest="r")
parser.add_argument('--n', action="store", dest="n")
parser.add_argument('--p', action="store", dest="p")
args = parser.parse_args()
print(args)
if(args.lr):
	learning_rate = args.lr
if(args.batch_size):
	if(not(batch_size == 1 or batch_size%5 == 0)):
		raise ValueError('Valid values for batch_size are 1 and multiples of 5 only')
	else:
		batch_size = args.batch_size	
if(args.method):
	method = args.method
if(args.k_cd):
	k_cd = args.k_cd
if(args.r):
	r = args.r
if(args.n):
	n = args.n
if(args.p):
	p = args.p
#########################################################################################################



##################################### TRAIN AND TEST ###################################################
#TRAIN
all_vs = []
step = 0
loss_file = open("losses", 'w')

for e in range(epochs):
	total_loss = 0
	print 'Epoch ', e+1

	randomize = np.arange(X_train.shape[0])
	np.random.shuffle(randomize)
	X_train = X_train[randomize]

	inner_step = 0
	for i in range(0, X_train.shape[0], batch_size):
		X_train_batch = X_train[i:i + batch_size]

		step += 1
		inner_step += 1
		#print(step)
		if(method == "cd"):
			loss = update_step_cd(X_train_batch.T, e)
		elif(method == "gibbs"):
			loss = update_step_gibbs(X_train_batch.T, inner_step)

		if(step % ((epochs * X_train.shape[0]/batch_size)/64) == 0):
			all_vs.append(reconstruct(X_test[9:10].T))

		#print(loss)
		total_loss += loss

	#Training loss per input
	avg_loss = total_loss/(X_train.shape[0]/batch_size)
	#Test loss per input
	# test_image = X_test[3:4].T
	# rec_image = reconstruct(test_image)
	# test_loss = np.sum(np.mean((test_image - rec_image) ** 2, axis = 1))
	#Print and write losses to file
	print 'Training Loss :', avg_loss
	# print 'Test Loss :', test_loss
	loss_file.write(str(e+1) + " " + str(avg_loss) + "\n")
	# loss_file.write(str(e+1) + " " + str(avg_loss) + " " + str(test_loss) + "\n")

plot_images(all_vs)
loss_file.close()


#PLOT TRAINING REPRESENTATIONS AND PLOT
# Get X_train_rep from X_train
X_train_rep = sigmoid(X_train_original[:200].T, W, c)
X_train_rep = np.where(np.random.rand(n, 1) < X_train_rep, 1, 0).T
t_SNE_plot(X_train_rep, labels_train[:200])


#GET TEST REPRESENTATIONS AND PLOT
#Get X_test_rep from X_test
X_test_rep = sigmoid(X_test[:200].T, W, c)
X_test_rep = np.where(np.random.rand(n, 1) < X_test_rep, 1, 0).T
t_SNE_plot(X_test_rep, labels_test[:200])
#########################################################################################################

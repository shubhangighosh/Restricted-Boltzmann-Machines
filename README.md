# Restricted-Boltzmann-Machines
RBMs were trained on a dataset of 60000 images to obtain hidden representations.  
RBMs were trained using:  
1. Constrastive Divergence  
2. Gibbs Sampling  
T-SNE plots of hidden representations were generated for various experiments.
The results of the conducted experiments can be found [here](https://github.com/shubhangighosh/Restricted-Boltzmann-Machines/blob/master/Report.pdf).
  
To test, run :   

python train.py -lr 		(learning rate)   
&nbsp;&nbsp;				-batch_size (batch size)  
&nbsp;&nbsp;				-method 	(cd or gibbs)  
&nbsp;&nbsp;				-k			(k for cd)  
&nbsp;&nbsp;				-r 	 		(r for cd)  
&nbsp;&nbsp;				-n			(num hidden units)  
&nbsp;&nbsp;				-p 			(momentum)  
  



FILES :   

train.py    	- 	Training and testing code  
losses 			- 	Text files containing training and test losses vs epochs  
plot_loss.py 	- 	Plots training and test loss vs epochs  
images.png		- 	Visualization of the sampled images, as training progresses  
losses.png 		- 	Visualization of training loss versus epochs  
tsne_train.png 	- 	Visualization of 2D representations of training images  
tsne_test.png 	- 	Visualization of 2D representations of test images  

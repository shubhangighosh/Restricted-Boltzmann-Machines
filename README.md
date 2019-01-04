# Restricted-Boltzmann-Machines
RBMs were trained on a dataset of 60000 images to obtain hidden representations.  
  
To test, run :   

python train.py -lr 		<learning rate>   
				-batch_size <batch size>  
				-method 	<cd or gibbs>  
				-k			<k for cd>  
				-r 	 		<r for cd>  
				-n			<num hidden units>  
				-p 			<momentum>  
  



FILES :   

train.py    	- 	Training and testing code  
losses 			- 	Text files containing training and test losses vs epochs  
plot_loss.py 	- 	Plots training and test loss vs epochs  
images.png		- 	Visualization of the sampled images, as training progresses  
losses.png 		- 	Visualization of training loss versus epochs  
tsne_train.png 	- 	Visualization of 2D representations of training images  
tsne_test.png 	- 	Visualization of 2D representations of test images  

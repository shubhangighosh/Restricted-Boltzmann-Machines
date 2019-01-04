import matplotlib.pyplot as plt

f = open('losses', 'r')

epochs = []
losses_train = []
losses_val = []
for line in f.readlines():
	epoch, train = line.strip().split()
	epochs.append(int(epoch))
	losses_train.append(float(train))
	# losses_val.append(float(val))


plt.plot(epochs,losses_train, linewidth=2, markersize=12, label="Train Loss")
# plt.plot(epochs,losses_val, linewidth=2, markersize=12, label="Test Loss")
plt.legend()
plt.title("Loss vs Epochs")
plt.show()






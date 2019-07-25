import numpy as np
import matplotlib.pyplot as plt
import pickle

# Distance for moving average
avg_dist = 5

# Epoch offset
epoch_offset = (1 + avg_dist) / 2.0

# Calculate moving average
def moving_average(x, dist):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    distsum = cumsum[dist:] - cumsum[:-dist]
    return distsum / float(dist)

# ------------
# Graph SGD
# ------------
path = './train_sgd/'
suffixes = ['b{}'.format(i) for i in [2,4,8,16,32,64,128]]

plt.rc('legend',**{'fontsize':12})
plt.rc('xtick', labelsize=12)    
plt.rc('ytick', labelsize=12)
fig = plt.figure(figsize=(15,15))

for suffix in suffixes:
    train_accs = pickle.load(open(path + 'train_accs_' + suffix + '.pkl', "rb"))
    train_losses = pickle.load(open(path + 'train_losses_' + suffix + '.pkl', "rb"))
    valid_accs = pickle.load(open(path + 'valid_accs_' + suffix + '.pkl', "rb"))
    valid_losses = pickle.load(open(path + 'valid_losses_' + suffix + '.pkl', "rb"))

    train_accs = moving_average(train_accs, avg_dist)
    train_losses = moving_average(train_losses, avg_dist)
    valid_accs = moving_average(valid_accs, avg_dist)
    valid_losses = moving_average(valid_losses, avg_dist)
    
    x = np.arange(epoch_offset, len(train_accs) + epoch_offset, 1.0)
    
    plt.subplot(221)
    plt.title("Training Accuracy (SGD)", fontsize=20)
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Accuracy", fontsize=16)
    plt.plot(x, train_accs, label='Batch Size: {}'.format(suffix[1:]))
    plt.legend(loc='upper left')
    plt.grid()
    
    plt.subplot(222)
    plt.title("Validation Accuracy (SGD)", fontsize=20)
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Accuracy", fontsize=16)
    plt.plot(x, valid_accs, label='Batch Size: {}'.format(suffix[1:]))
    plt.legend(loc='upper left')
    plt.grid()
    
    plt.subplot(223)
    plt.title("Training Loss (SGD)", fontsize=20)
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Accuracy", fontsize=16)
    plt.ylim(0.65, 0.7)
    plt.plot(x, train_losses, label='Batch Size: {}'.format(suffix[1:]))
    plt.legend(loc='lower left')
    plt.grid()
    
    plt.subplot(224)
    plt.title("Validation Loss (SGD)", fontsize=20)
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Accuracy", fontsize=16)
    plt.ylim(0.65, 0.7)
    plt.plot(x, valid_losses, label='Batch Size: {}'.format(suffix[1:]))
    plt.legend(loc='lower left')
    plt.grid()

plt.savefig("sgd_mva_{}.png".format(avg_dist), bbox_inches='tight', pad_inches = 0)

# ------------
# Graph MSGD
# ------------
path = './train_msgd/'
suffixes = ['b{}'.format(i) for i in [2,4,8,16,32,64,128]]

plt.rc('legend',**{'fontsize':12})
plt.rc('xtick', labelsize=12)    
plt.rc('ytick', labelsize=12)
plt.figure(figsize=(15,15))
for suffix in suffixes:
    train_accs = pickle.load(open(path + 'train_accs_' + suffix + '.pkl', "rb"))
    train_losses = pickle.load(open(path + 'train_losses_' + suffix + '.pkl', "rb"))
    valid_accs = pickle.load(open(path + 'valid_accs_' + suffix + '.pkl', "rb"))
    valid_losses = pickle.load(open(path + 'valid_losses_' + suffix + '.pkl', "rb"))

    train_accs = moving_average(train_accs, avg_dist)
    train_losses = moving_average(train_losses, avg_dist)
    valid_accs = moving_average(valid_accs, avg_dist)
    valid_losses = moving_average(valid_losses, avg_dist)
    
    x = np.arange(epoch_offset, len(train_accs) + epoch_offset, 1.0)
    
    plt.subplot(221)
    plt.title("Training Accuracy (MSGD)", fontsize=20)
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Accuracy", fontsize=16)
    plt.plot(x, train_accs, label='Batch Size: {}'.format(suffix[1:]))
    plt.legend(loc='lower right')
    plt.grid()
    
    plt.subplot(222)
    plt.title("Validation Accuracy (MSGD)", fontsize=20)
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Accuracy", fontsize=16)
    plt.plot(x, valid_accs, label='Batch Size: {}'.format(suffix[1:]))
    plt.legend(loc='lower right')
    plt.grid()
    
    plt.subplot(223)
    plt.title("Training Loss (MSGD)", fontsize=20)
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Accuracy", fontsize=16)
    plt.ylim(0.6, 0.7)
    plt.plot(x, train_losses, label='Batch Size: {}'.format(suffix[1:]))
    plt.legend(loc='lower left')
    plt.grid()
    
    plt.subplot(224)
    plt.title("Validation Loss (MSGD)", fontsize=20)
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Accuracy", fontsize=16)
    plt.ylim(0.6, 0.7)
    plt.plot(x, valid_losses, label='Batch Size: {}'.format(suffix[1:]))
    plt.legend(loc='lower left')
    plt.grid()
    
plt.savefig("msgd_mva_{}.png".format(avg_dist), bbox_inches='tight', pad_inches = 0)

# ------------
# Graph Adam
# ------------
path = './train_adam/'
suffixes = ['b{}'.format(i) for i in [2,4,8,16,32,64,128]]

plt.rc('legend',**{'fontsize':12})
plt.rc('xtick', labelsize=12)    
plt.rc('ytick', labelsize=12)
plt.figure(figsize=(15,15))
for suffix in suffixes:
    train_accs = pickle.load(open(path + 'train_accs_' + suffix + '.pkl', "rb"))
    train_losses = pickle.load(open(path + 'train_losses_' + suffix + '.pkl', "rb"))
    valid_accs = pickle.load(open(path + 'valid_accs_' + suffix + '.pkl', "rb"))
    valid_losses = pickle.load(open(path + 'valid_losses_' + suffix + '.pkl', "rb"))

    train_accs = moving_average(train_accs, avg_dist)
    train_losses = moving_average(train_losses, avg_dist)
    valid_accs = moving_average(valid_accs, avg_dist)
    valid_losses = moving_average(valid_losses, avg_dist)
    
    x = np.arange(epoch_offset, len(train_accs) + epoch_offset, 1.0)
    
    plt.subplot(221)
    plt.title("Training Accuracy (Adam)", fontsize=20)
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Accuracy", fontsize=16)
    plt.plot(x, train_accs, label='Batch Size: {}'.format(suffix[1:]))
    plt.legend(loc='upper left')
    plt.grid()
    
    plt.subplot(222)
    plt.title("Validation Accuracy (Adam)", fontsize=20)
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Accuracy", fontsize=16)
    plt.plot(x, valid_accs, label='Batch Size: {}'.format(suffix[1:]))
    plt.legend(loc='upper left')
    plt.grid()
    
    plt.subplot(223)
    plt.title("Training Loss (Adam)", fontsize=20)
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Accuracy", fontsize=16)
    plt.ylim(0.64, 0.7)
    plt.plot(x, train_losses, label='Batch Size: {}'.format(suffix[1:]))
    plt.legend(loc='lower left')
    plt.grid()
    
    plt.subplot(224)
    plt.title("Validation Loss (Adam)", fontsize=20)
    plt.xlabel("Epoch", fontsize=16)
    plt.ylabel("Accuracy", fontsize=16)
    plt.ylim(0.64, 0.7)
    plt.plot(x, valid_losses, label='Batch Size: {}'.format(suffix[1:]))
    plt.legend(loc='lower left')
    plt.grid()
    
plt.savefig("adam_mva_{}.png".format(avg_dist), bbox_inches='tight', pad_inches = 0)
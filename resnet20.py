#!/usr/bin/env python3

# main ressource:
# 2016, He et al., Deep Residual Learning for Image Recognition
# additional ressources:
# https://www.cs.toronto.edu/~kriz/cifar.html (CIFAR-10 dataset)
# https://towardsdatascience.com/resnets-for-cifar-10-e63e900524e0 (visualizations of the layers [different shortcut connection though])
# https://pytorch-tutorial.readthedocs.io/en/latest/tutorial/chapter03_intermediate/3_2_2_cnn_resnet_cifar10/ (other example implementation)
# https://blog.paperspace.com/writing-resnet-from-scratch-in-pytorch/  (other example implementation)
# https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py (other example implementation)
# https://colab.research.google.com/drive/1AQX35TJdH0u_mWoEEAkK_CN00xA6TyTY#scrollTo=2rGfAsMsMvfe (other example implementation, this one is quite useful)
# https://catalog.ngc.nvidia.com/orgs/nvidia/resources/resnet_50_v1_5_for_pytorch (explains ResNet v1.5)

import copy
import numpy as np
import torch
#
import torch.nn as nn
import torch.optim as optim
import torchvision
#
import cifar_io

if torch.cuda.is_available():  
    dev = 'cuda:0'
else:  
    dev = 'cpu'

bool_bias_conv_before_bn = False # using bias after batch normalization (bn) is redundant, since bn already includes a bias term
bool_bias_conv = True
bool_bias_linear = True
    
# He et al.:
#
# We adopt the second nonlinearity after the addition (i.e., σ(y), see Fig. 2).
#
# We adopt batch normalization (BN) [16] right after each convolution and before activation, following [16].
#
# (A) The shortcut still performs identity mapping, with extra zero entries padded for increasing dimensions.
# In the first comparison, we use identity mapping for all shortcuts and zero-padding for increasing dimensions (option A).
class building_block(nn.Module): 
    def __init__(self, 
                 dim_in, dim_between, dim_out,
                 kernel_size=3, stride=1, bool_bias=True):
        super(building_block, self).__init__()
        self.conv1 = nn.Conv2d(dim_in, dim_between, kernel_size,
                               stride=(stride, stride), padding=(1, 1), dilation=1, groups=1, bias=bool_bias_conv_before_bn, padding_mode='zeros',
                               device=dev, dtype=torch.float32)
        self.conv2 = nn.Conv2d(dim_between, dim_out, kernel_size,
                               stride=(1, 1), padding=(1, 1), dilation=1, groups=1, bias=bool_bias_conv_before_bn, padding_mode='zeros',
                               device=dev, dtype=torch.float32)
        self.batchnorm1 = nn.BatchNorm2d(dim_between,
                                         eps=1e-05, momentum=0.1, affine=True, track_running_stats=True,
                                         device=dev, dtype=torch.float32)
        self.batchnorm2 = nn.BatchNorm2d(dim_out,
                                         eps=1e-05, momentum=0.1, affine=True, track_running_stats=True,
                                         device=dev, dtype=torch.float32)
        self.sigma = nn.ReLU(inplace=True)
        self.shortcut = self.get_shortcut(dim_in, dim_out, stride)
    
    def get_shortcut(self, dim_in, dim_out, stride):
        if (stride == 1):
            fcn = nn.Identity()
        else:
            fcn = lambda x: torch.nn.functional.pad(x[:, :, ::stride, ::stride],
                                                    (0, 0, 0, 0, int((dim_out-dim_in)/2), int((dim_out-dim_in)/2)),
                                                    mode="constant", value=0)
        return fcn

    def forward(self, x):
        y = self.batchnorm1(self.conv1(x))
        y = self.sigma(y)
        y = self.batchnorm2(self.conv2(y))
        y = y + self.shortcut(x)
        y = self.sigma(y)
        return y

# most shallow ResNet with n=3 resutling in a total of 6n+2=20 layers
#
# He et al.:
#
# The plain/residual architectures follow the form in Fig. 3 (middle/right).
# The network inputs are 32×32 images, with the per-pixel mean subtracted.
# The first layer is 3×3 convolutions.
#
# Then we use a stack of 6n layers with 3×3 convolutions on the feature maps of sizes {32, 16, 8} respectively, with 2n layers for each feature map size.
# The numbers of filters are {16, 32, 64} respectively.
# The subsampling is performed by convolutions with a stride of 2.
#
# The network ends with a global average pooling, a 10-way fully-connected layer, and softmax.
# There are totally 6n+2 stacked weighted layers.
#
# When shortcut connections are used, they are connected to the pairs of 3×3 layers (totally 3n shortcuts).
# On this dataset we use identity shortcuts in all cases (i.e., option A), so our residual models have exactly the same depth, width, and number of parameters as the plain counterparts.
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        dim_in = 3 # 3 color channels
        kernel_size = 3
        stride = 1
        #
        self.conv0 = nn.Conv2d(dim_in, 16, kernel_size,
                               stride=(1, 1), padding=(1, 1), dilation=1, groups=1, bias=bool_bias_conv, padding_mode='zeros',
                               device=dev, dtype=torch.float32)
        self.batchnorm0 = nn.BatchNorm2d(16,
                                     eps=1e-05, momentum=0.1, affine=True, track_running_stats=True,
                                     device=dev, dtype=torch.float32)
        self.sigma = nn.ReLU(inplace=True)
        #
        self.block16_1 = building_block(16, 16, 16, kernel_size, stride)
        self.block16_2 = building_block(16, 16, 16, kernel_size, stride)
        self.block16_3 = building_block(16, 16, 16, kernel_size, stride)
        #
        self.block32_1 = building_block(16, 32, 32, kernel_size, 2)
        self.block32_2 = building_block(32, 32, 32, kernel_size, stride)
        self.block32_3 = building_block(32, 32, 32, kernel_size, stride)
        #
        self.block64_1 = building_block(32, 64, 64, kernel_size, 2)
        self.block64_2 = building_block(64, 64, 64, kernel_size, stride)
        self.block64_3 = building_block(64, 64, 64, kernel_size, stride)
        #
        self.avgpool = nn.AvgPool2d(8, # dimensions at this point: 64x8x8
                                    stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)
        self.mlp = nn.Linear(int(8**2), cifar_io.nClasses, # CIFAR-10 has 10 categories -> "10-way fully-connected layer"
                             bias=bool_bias_linear, dtype=torch.float32, device=dev)
#         self.softmax = nn.Softmax(dim=1) # do not use softmax since it will be computed by the loss function
        
    def forward(self, x):
        # block 0
        y = self.conv0(x)
        y = self.batchnorm0(y)
        y = self.sigma(y)
        # block 1
        y = self.block16_1(y)
        y = self.block16_2(y)
        y = self.block16_3(y)
        # block 2
        y = self.block32_1(y)
        y = self.block32_2(y)
        y = self.block32_3(y)
        # block 3
        y = self.block64_1(y)
        y = self.block64_2(y)
        y = self.block64_3(y)
        # fully-connected layer
        y = self.avgpool(y).squeeze()
        y = self.mlp(y)
#         y = self.softmax(y) # do not use softmax since it will be computed by the loss function
        return y

if __name__ == '__main__':
    # load training and test data
    imgs_train, labels_train, label_names = cifar_io.load_images_labels(cifar_io.list_files_train)
    imgs_test, labels_test, _ = cifar_io.load_images_labels(cifar_io.list_files_test)
    
    # get validation data (45k/5k split like He et al.)
    nTest = cifar_io.nImgs_test
    nValid = int(5e3)
    nTrain = cifar_io.nImgs_train - nValid
    split_ratio = nValid / nTrain
    #
    # split into train and validation data sets and make sure all labels is represented equally in both of them 
    nValid_class = int(nValid / cifar_io.nClasses)
    ids_valid = list()
    ids_train = list()
    ids_all = np.arange(cifar_io.nImgs_train, dtype=np.int32)
    for i in range(cifar_io.nClasses):
        mask_class = (labels_train == i)
        ids_class = ids_all[mask_class]
        ids_class_perm = np.random.permutation(ids_class)
        ids_valid = ids_valid + list(ids_class_perm[:nValid_class])
        ids_train = ids_train + list(ids_class_perm[nValid_class:])
    ids_valid = np.array(sorted(ids_valid), dtype=np.int32)
    ids_train = np.array(sorted(ids_train), dtype=np.int32)
    imgs_valid = imgs_train[ids_valid]
    labels_valid = labels_train[ids_valid]
    imgs_train = imgs_train[ids_train]
    labels_train = labels_train[ids_train]
        
    # cast to float32
    imgs_train = imgs_train.astype(np.float32)
    imgs_valid = imgs_valid.astype(np.float32)
    imgs_test = imgs_test.astype(np.float32)
    labels_train = labels_train.astype(np.float32)
    labels_valid = labels_valid.astype(np.float32)
    labels_test = labels_test.astype(np.float32)
    
    # normalize pixel to be within [0, 1]
    imgs_train = imgs_train / 255.0
    imgs_valid = imgs_valid / 255.0
    imgs_test = imgs_test / 255.0
    # substract per-pixel mean (preprocessing done by He et al.)
    mean_imgs_train = np.mean(imgs_train, (0, 2, 3))[None, :, None, None] # dim: 3
    std_imgs_train = np.std(imgs_train, (0, 2, 3))[None, :, None, None] # dim: 3
    imgs_train = (imgs_train - mean_imgs_train) / std_imgs_train
    imgs_valid = (imgs_valid - mean_imgs_train) / std_imgs_train
    imgs_test = (imgs_test - mean_imgs_train) / std_imgs_train
    
#     # plot single image
#     import matplotlib.pyplot as plt
#     i_img = 0
#     imgs_plot1 = imgs_train[i_img]    
#     imgs_plot1 = np.swapaxes(imgs_plot1, 0, 2)
#     imgs_plot1 = np.swapaxes(imgs_plot1, 0, 1)
#     #
#     fig = plt.figure(1, figsize=(8, 8))
#     fig.clear()
#     ax1 = fig.add_subplot(1, 1, 1)
#     ax1.clear()
#     ax1.axis('off')
#     ax1.set_title(label_names[labels_train[i_img]])
#     im1 = ax1.imshow(imgs_plot1 / 255.0,
#                      vmin=0.0, vmax=1.0)
#     fig.canvas.draw()
#     plt.show(block=True)
    
    # put on device
    imgs_train = torch.from_numpy(imgs_train).to(dev)
    imgs_valid = torch.from_numpy(imgs_valid).to(dev)
    imgs_test = torch.from_numpy(imgs_test).to(dev)
    labels_train = torch.from_numpy(labels_train).to(dev)
    labels_valid = torch.from_numpy(labels_valid).to(dev)
    labels_test = torch.from_numpy(labels_test).to(dev)

    # define network and get parameters
    net = ResNet().to(dev)
    
    # set up optimizer
    optimizer = optim.SGD(net.parameters(),
                          lr=1e-1, # used by He et al.
                          momentum=0.9, # used by He et al.
                          dampening=0,
                          weight_decay=0.0001, # used by He et al.
                          nesterov=False,)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[int(32e3), int(48e3)],
                                                     gamma=1e-1,
                                                     last_epoch=-1,
                                                     verbose=False)

    # set up training
    nIter = int(64e3)
    size_batch = 128
    # image ids
    ratio = int(np.ceil((nIter*size_batch) / nTrain))
    # in this torch version mask idices need to be int64
    permuted_img_ids = torch.zeros(ratio*nTrain, dtype=torch.int64) 
    for i in range(ratio):
        permuted_img_ids[i*nTrain:(i+1)*nTrain] = \
            torch.randperm(nTrain,
                           generator=None,
                           out=None,
                           dtype=torch.int32,
                           layout=torch.strided,
                           device=None)
    img_ids = permuted_img_ids[:nIter*size_batch].reshape(nIter, size_batch)
    # for data augmentation
    flip = torchvision.transforms.RandomHorizontalFlip(p=0.5)
    crop = torchvision.transforms.RandomCrop(32, 4)
    # for validation inside loop
    x_valid = torch.clone(imgs_valid)
    # cast to int64 since it does not work as a mask otherwise
    y0_valid_pre = torch.clone(labels_valid).type(torch.int64) 
    y0_valid = torch.zeros([nValid, cifar_io.nClasses], dtype=torch.float32).to(dev)
    y0_valid.scatter_(1, y0_valid_pre.unsqueeze(1), 1.0)  
    accuracy_valid0 = 0.0
    # delete redundant variable to minimize memory requirements
    del(imgs_valid)
    
    # put on device
    img_ids = img_ids.to(dev)
    #
    x_pre = torch.zeros([size_batch, cifar_io.img_size+8, cifar_io.img_size+8], dtype=torch.float32).to(dev)
    x = torch.zeros([size_batch, cifar_io.img_size, cifar_io.img_size], dtype=torch.float32).to(dev)
    y0_pre = torch.zeros([size_batch], dtype=torch.int32).to(dev)
    y0 = torch.zeros([size_batch, cifar_io.nClasses], dtype=torch.float32).to(dev)
    #
    x_valid = x_valid.to(dev)
    y0_valid = y0_valid.to(dev)
    labels_valid = labels_valid.to(dev)
    
    # start training
    print('Training ResNet20 on CIFAR-10:')
    print()
    #
    # print network architecture and parameters
    net_params_list = list(net.parameters())
#     print('Network architecture:')
#     print(net)
#     print('Number of parameters:')
#     print('\t{:01d}'.format(len(net_params_list)))
#     print('Parameter dimensions:')
#     for i in range(len(net_params_list)):
#         print('\t{:02d}:\t'.format(i+1)+'x'.join([str(j) for j in net_params_list[i].size()]))
    print('Number of trainable parameters:')
    print('\t{:01d}'.format(sum(p.numel() for p in net.parameters() if p.requires_grad)))
    print()
    #
    for i in torch.arange(nIter+1, dtype=torch.int32).to(dev): # +1 to print and save weights of last validation
        x_pre = torch.clone(imgs_train[img_ids[i]])

        # FLIP & CROP
        x = crop(flip(x_pre))
        
#         # plot single image
#         import matplotlib.pyplot as plt
#         i_img = 0
#         imgs_plot1 = imgs_train[img_ids[i]][i_img]    
#         imgs_plot1 = np.swapaxes(imgs_plot1, 0, 2)
#         imgs_plot1 = np.swapaxes(imgs_plot1, 0, 1)
#         imgs_plot2 = x[i_img]
#         imgs_plot2 = np.swapaxes(imgs_plot2, 0, 2)
#         imgs_plot2 = np.swapaxes(imgs_plot2, 0, 1)
#         #
#         fig = plt.figure(1, figsize=(8, 8))
#         fig.clear()
#         ax1 = fig.add_subplot(1, 1, 1)
#         ax1.clear()
#         ax1.axis('off')
#         ax1.set_title(label_names[int(labels_train[img_ids[i]][i_img].item())])
#         im1 = ax1.imshow((imgs_plot1 + 255.0) / (255.0*2+1),
#                          vmin=0.0, vmax=1.0)
#         fig.canvas.draw()
#         plt.show(block=False)

        # TRAINING DATA
        y0_pre = torch.clone(labels_train[img_ids[i]]).type(torch.int64) # cast to int64 since it does not work as a mask otherwise
        y0.zero_()
        y0.scatter_(1, y0_pre.unsqueeze(1), 1.0)        
        
        # TRAINING
        y1 = net(x)        
        # using cross entropy loss for classification task
        loss = nn.functional.cross_entropy(y1, y0,
                                           weight=None,
                                           size_average=None,
                                           ignore_index=-100,
                                           reduce=None,
                                           reduction='mean',
                                           label_smoothing=0.0)
        loss.backward() # do backpropagation to calculate gradients
        optimizer.step() # walk in direction of negative gradient
        optimizer.zero_grad() # zero gradients for next iteration
        scheduler.step() # adjust learning rate

        # VALIDATE AND PRINT
        if ((i % 1e3) == 0):
            with torch.no_grad():
                y1_valid = net(x_valid)
                loss_valid1 = nn.functional.cross_entropy(y1_valid, y0_valid,
                                                          weight=None,
                                                          size_average=None,
                                                          ignore_index=-100,
                                                          reduce=None,
                                                          reduction='mean',
                                                          label_smoothing=0.0)
                is_correct = torch.sum(torch.argmax(y1_valid, 1) == labels_valid)
            accuracy_valid1 = float(is_correct) / float(nValid) * 100
            if (accuracy_valid1 > accuracy_valid0): # FIXME: check if this is correct now
                weights = copy.deepcopy(net.state_dict())
                accuracy_valid0 = torch.clone(accuracy_valid1)
                iteration_valid = torch.clone(i)
            #
            print('iteration:\t{:07d} / {:07d} '.format(i, nIter))
            print('learn. rate:\t{:0.2e}'.format(scheduler.get_last_lr()[0]))
#             print('train size:\t{:07d}'.format(nTrain))
#             print('batch size:\t{:07d}'.format(size_batch))
            print('train loss:\t{:0.2e}'.format(float(loss.data)))
            print('val. loss:\t{:0.2e}'.format(float(loss_valid1.data)))
            print('val. accuracy:\t{:0.2f}%'.format(accuracy_valid1))
            print()

    # SAVE
    torch.save(weights, 'weights.pt')
    
    # TEST
#     weights = torch.load('weights.pt', map_location=dev) # load weights
    net.load_state_dict(weights)
    # use for loop to minimze memory requirements
    n = 10
    N = int(cifar_io.nImgs_test / n)
    is_correct = 0
    with torch.no_grad():
        for i in range(n):
            x_test = imgs_test[i*N:(i+1)*N]
            y1 = net(x_test)
            is_correct += torch.sum(torch.argmax(y1, 1) == labels_test[i*N:(i+1)*N])
    accuracy = float(is_correct) / float(cifar_io.nImgs_test) * 100
    print('Finished training ResNet20 on CIFAR-10')
    print('Using weights from iteration {:01d} to compute classification accuracy'.format(iteration_valid.data))
    print('Resutling classification accuracy on test data:\t{:0.2f}%'.format(accuracy))

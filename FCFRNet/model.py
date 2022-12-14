import paddle
import paddle.nn as nn
from paddle.vision.models import  resnet
from paddle import  ParamAttr
import paddle.nn.functional as F
import os 
import argparse
from dataloader.kitti_loader import load_calib, oheight, owidth, input_options, KittiDepth
import os
from metric import Result,AverageMeter
from inverse_warp import Intrinsics, homography_from
import cv2 

def init_weights(m):
    init = nn.initializer.Normal(mean=0, std=1e-3)
    zeros = nn.initializer.Constant(0.)
    ones = nn.initializer.Constant(1.)
    if isinstance(m, nn.Conv2D) or isinstance(m, nn.Linear):
        # m.weight.data.normal_(0, 1e-3)
        init(m.weight)
        if m.bias is not None:
            # m.bias.data.zero_()
            zeros(m.bias)
    elif isinstance(m, nn.Conv2DTranspose):
        # m.weight.data.normal_(0, 1e-3)
        init(m.weight)
        if m.bias is not None:
            # m.bias.data.zero_()
            zeros(m.bias)
    elif isinstance(m, nn.BatchNorm2D):
        # m.weight.data.fill_(1) torch
        # m.bias.data.zero_() torch
        ones(m.weight)
        zeros(m.weight)

def my_conv(tensor_image):
    kernel = [[1, 1, 1,1,1],
              [1, 1, 1,1,1],
              [1, 1, 1,1,1],
              [1, 1, 1,1,1],
              [1, 1, 1,1,1]]
    # kernel = torch.cuda.FloatTensor(kernel).expand(1, 1, 5, 5)
    kernel=paddle.to_tensor(kernel,dtype="float32").expand((1,1,5,5))
    #weight = ParamAttr(data=kernel, requires_grad=False)
    # 
    param=paddle.create_parameter(shape=kernel.shape,dtype=str(kernel.numpy().dtype))####之前默认初始化当成设置处置
    param.set_value(kernel) ###############################new added #################
    param.stop_gradient=True #等价于requires_grad=False 不进行求导
    
    return F.conv2d(tensor_image, param, padding=2)

def feature_fuse(d1, d2):
    # set M=N=3
    #print(d1.shape)
    b,c,h,w = d1.shape
    d1_2 = d1.multiply (d1)
    d2_2 = d2.multiply (d2)
    E1 = my_conv(d1_2.reshape((-1,1,h,w)))
    E2 = my_conv(d2_2.reshape((-1,1,h,w)))
    E1 = E1.reshape((b,c,h,w))
    E2 = E2.reshape((b,c,h,w))

    mask1 = paddle.ones_like(d1)
    mask1=mask1.clone()
    mask1[E1<E2] = 0
    F11 = d1*mask1

    mask2 = paddle.ones_like(d1)
    mask2 = mask2.clone()
    mask2[E1 >= E2] = 0
    F12 = d2 * mask2
    feature1 = F11+F12

    return feature1 * 2

def conv_bn_relu(in_channels,out_channels,kernel_size,\
                 stride=1,padding=0,bn=True,relu=True):
    bias_attr=not bn
    layers=[]
    layers.append(
        nn.Conv2D(in_channels,out_channels,kernel_size,stride,padding,bias_attr=bias_attr))
    if bn:
        layers.append(nn.BatchNorm2D(out_channels))
    if relu:
        layers.append(nn.LeakyReLU(0.2))
    layers=nn.Sequential(*layers)
    for m in layers.sublayers():
        init_weights(m)

    return  layers

def convt_bn_relu(in_channels,out_channels,kernel_size,\
                 stride=1,padding=0,output_padding=0,bn=True,relu=True):
    bias_attr=not bn
    layers=[]
    layers.append(
        nn.Conv2DTranspose(in_channels,
                           out_channels,
                           kernel_size,
                           stride,
                           padding,
                           output_padding,bias_attr=bias_attr))
    if bn:
        layers.append(nn.BatchNorm2D(out_channels))
    if relu:
        layers.append(nn.LeakyReLU(0.2))
    layers=nn.Sequential(*layers)
    for m in layers.sublayers():
        init_weights(m)

    return  layers

class DepthComletionNet(nn.Layer):
    def __init__(self,args):
        assert args.layers in [18,34,50,101,152], 'Only layers 18, 34, 50, 101, and 152 are defined, but got {}'.format(args.layers)
        super(DepthComletionNet, self).__init__()
        self.modality=args.input # input type
        # print(len(self.modality))
        self.layers=args.layers
        if 'd' in self.modality:
            channels = 64 // len(self.modality)
         #   self.conv1_d = conv_bn_relu(1,channels,kernel_size=3,stride=1,padding=1)
            self.conv1_d_2 = conv_bn_relu(1,64,kernel_size=3,stride=1,padding=1)
        if 'rgb' in self.modality:
            channels = 64 * 3 // len(self.modality)
            self.conv1_img = conv_bn_relu(3,64,kernel_size=3,stride=1,padding=1)
        elif 'g' in self.modality:
            channels = 64 // len(self.modality)
            # self.conv1_img = conv_bn_relu(1,channels,kernel_size=3,stride=1,padding=1)
            self.conv1_img = conv_bn_relu(1,64,kernel_size=3,stride=1,padding=1)

        pretrained_model = resnet.__dict__['resnet{}'.format(args.layers)](pretrained=args.pretrained)
        # print(pretrained_model)
        if not args.pretrained:
            pretrained_model.apply(init_weights)
        #self.maxpool = pretrained_model._modules['maxpool']
        self.conv2 = pretrained_model.layer1
        self.conv3 = pretrained_model.layer2
        self.conv4 = pretrained_model.layer3
        self.conv5 = pretrained_model.layer4
        del pretrained_model  # clear memory

        # define number of intermediate channels
        if args.layers <= 34:
            num_channels = 512
        elif args.layers >= 50:
            num_channels = 2048
        self.conv6 = conv_bn_relu(num_channels, 512, kernel_size=3, stride=2, padding=1)


        # decoding layers
        kernel_size = 3
        stride = 2
        self.convt5 = convt_bn_relu(in_channels=(512),out_channels=256,kernel_size=kernel_size,
                                    stride=stride,padding=1,output_padding=1)
        self.convt4 = convt_bn_relu(in_channels=(768+512),out_channels=128,kernel_size=kernel_size,
                                    stride=stride,padding=1,output_padding=1)
        self.convt3 = convt_bn_relu(in_channels=(256 + 128+256),out_channels=64,kernel_size=kernel_size,
                                    stride=stride,padding=1,output_padding=1)
        self.convt2 = convt_bn_relu(in_channels=(128 + 64+128),out_channels=64,kernel_size=kernel_size,
                                    stride=stride,padding=1,output_padding=1)
        self.convt1 = convt_bn_relu(in_channels=(128+64),out_channels=64,kernel_size=kernel_size,stride=1,padding=1)
        self.convtf = conv_bn_relu(in_channels=(128+64),out_channels=1,kernel_size=1,stride=1,bn=False,relu=False)

        ##############
        ## second path
        pretrained_model2 = resnet.__dict__['resnet{}'.format(args.layers)](pretrained=args.pretrained)
        if not args.pretrained:
            pretrained_model2.apply(init_weights)
        # self.maxpool = pretrained_model._modules['maxpool']
        self.conv2_2 = pretrained_model2.layer1
        self.conv3_2 = pretrained_model2.layer2
        self.conv4_2 = pretrained_model2.layer3
        self.conv5_2 = pretrained_model2.layer4
        del pretrained_model2  # clear memory

        # define number of intermediate channels
        if args.layers <= 34:
            num_channels = 512
        elif args.layers >= 50:
            num_channels = 2048
        self.conv6_2 = conv_bn_relu(num_channels, 512, kernel_size=3, stride=2, padding=1)

    def forward(self,x):
        if "d" in self.modality:
            conv1_2=self.conv1_d_2(x['d'])  # fd0
        if 'rgb' in self.modality:
            conv1=self.conv1_img(x['rgb']) # fcolor0
        elif 'g' in self.modality: 
            conv1=self.conv1_img(x['g']) 
        
        conv2=self.conv2(conv1)  #Rcolor1
        #print(f"conv2.shape:{conv2.shape}")
        conv2_2=self.conv2_2(conv1_2) # Rd1
        #print(f"conv2_2.shape:{conv2_2.shape}")
        b,c,h,w=conv2.shape 
        cat_all=paddle.concat((conv2,conv2_2),1) ###?这是对应的哪个地方？############>>>>>??????>>>>????
        #print(f"cat_all.shape:{cat_all.shape}")        # [1, 128, 1216, 352]====>b 2 c h w ??
        cat_reshape=paddle.reshape(cat_all,(b,2,c,h,w)) ### 为何#######################???>>>???>>>???
        #print(f"cat_reshape.shape:{cat_reshape.shape}")
        cat_transpose=paddle.transpose(cat_reshape,(0,2,1,3,4)) # b c 2 h w 这个2是啥
        cat_view=paddle.reshape(cat_transpose,[b,-1,h,w]) #### b c 2 h w ====> b c h w >
        #print(cat_view.shape)
        #=========================> channel fuse  <======================
        conv2=cat_view[:,:c,:,:]  
        conv2_2=cat_view[:,c:,:,:] # 不是叠起来再取出来？
        
        conv3=self.conv3(conv2) ### 调用conv2
        conv3_2=self.conv3_2(conv2_2)
        b,c,h,w=conv3.shape
        cat_all=paddle.concat((conv3,conv3_2),1)   
        cat_reshape=paddle.reshape(cat_all,[b,2,c,h,w])
        cat_transpose=paddle.transpose(cat_reshape,[0,2,1,3,4])
        cat_view=paddle.reshape(cat_transpose,[b,-1,h,w])   #### ?
        conv3 = cat_view[:, :c, :, :]
        conv3_2 = cat_view[:, c:, :, :]

        conv4=self.conv4(conv3)  
        conv4_2=self.conv4_2(conv3)
        b,c,h,w=conv4.shape 
        cat_all=paddle.concat((conv4,conv4_2),1) 
        cat_reshape=paddle.reshape(cat_all,[b,2,c,h,w])  
        cat_transpose=paddle.transpose(cat_reshape,[0,2,1,3,4])  
        cat_view=paddle.reshape(cat_transpose,[b,-1,h,w]) 
        conv4=cat_view[:,:c,:,:]
        conv4_2=cat_view[:,c:,:,:] 

        conv5=self.conv5(conv4)  
        conv5_2=self.conv5_2(conv4)
        b,c,h,w=conv5.shape 
        cat_all=paddle.concat((conv5,conv5_2),1) 
        cat_reshape=paddle.reshape(cat_all,[b,2,c,h,w])  
        cat_transpose=paddle.transpose(cat_reshape,[0,2,1,3,4])  
        cat_view=paddle.reshape(cat_transpose,[b,-1,h,w]) 
        conv5=cat_view[:,:c,:,:]
        conv5_2=cat_view[:,c:,:,:] 

        conv6=self.conv6(conv5)  
        conv6_2=self.conv6_2(conv5)
        b,c,h,w=conv6.shape 
        cat_all=paddle.concat((conv6,conv6_2),1) 
        cat_reshape=paddle.reshape(cat_all,[b,2,c,h,w])  
        cat_transpose=paddle.transpose(cat_reshape,[0,2,1,3,4])  
        cat_view=paddle.reshape(cat_transpose,[b,-1,h,w]) 
        conv6=cat_view[:,:c,:,:]
        conv6_2=cat_view[:,c:,:,:] 

        conv6_all = feature_fuse(conv6, conv6_2)
        # decoder
        convt5 = self.convt5(conv6_all)
        y=paddle.concat((convt5,conv5,conv5_2),1)
        #print(f"y.shape:{y.shape}")
        convt4 = self.convt4(y)
        # y = torch.cat((convt4, conv4, conv4_2), 1)
        y=paddle.concat((convt4,conv4,conv4_2),1)
        convt3 = self.convt3(y)
        # y = torch.cat((convt3, conv3, conv3_2), 1)
        y=paddle.concat((convt3,conv3,conv3_2),1)  
        convt2 = self.convt2(y)
        # y = torch.cat((convt2, conv2, conv2_2), 1)
        y=paddle.concat((convt2,conv2,conv2_2),1) 
        convt1 = self.convt1(y)
        # y = torch.cat((convt1, conv1, conv1_2), 1)
        y=paddle.concat((convt1,conv1,conv1_2),1)  
        y = self.convtf(y)

        depth_pred = y +x['d']

        if self.training:
            depth_pred = 85 * depth_pred
        else:
            min_distance = 0.9
            depth_pred = F.relu(85 * depth_pred - min_distance) + min_distance

        return depth_pred
    

if __name__ == '__main__':
    print("=====>start test model")
    parser = argparse.ArgumentParser(description='Sparse-to-Dense')


    parser.add_argument('--data-folder',
                    default='/home/aistudio/KITTI/',
                    type=str,
                    metavar='PATH',
                    help='data folder (default: none)')
    parser.add_argument('-i',
                    '--input',
                    type=str,
                    default='gd',
                    choices=input_options,
                    help='input: | '.join(input_options))
    parser.add_argument('-l',
                    '--layers',
                    type=int,
                    default=34,
                    help='use 16 for sparse_conv; use 18 or 34 for resnet')
    parser.add_argument('--pretrained',
                    action="store_true",
                    help='use ImageNet pre-trained weights')
    parser.add_argument(
    '--rank-metric',
    type=str,
    default='rmse',
    choices=[m for m in dir(Result()) if not m.startswith('_')],
    help='metrics for which best result is sbatch_datacted')
    parser.add_argument(
    '-m',
    '--train-mode',
    type=str,
    default="dense",
    choices=["dense", "sparse", "photo", "sparse+photo", "dense+photo"],
    help='dense | sparse | photo | sparse+photo | dense+photo')

    args = parser.parse_args()
    args.use_pose = ("photo" in args.train_mode)
# args.pretrained = not args.no_pretrained
    args.result = os.path.join('..', 'results')
    args.use_rgb = ('rgb' in args.input) or args.use_pose
    args.use_d = 'd' in args.input
    args.use_g = 'g' in args.input
    if args.use_pose:
        args.w1, args.w2 = 0.1, 0.1
    else:
        args.w1, args.w2 = 0, 0
    print(args)

    model=DepthComletionNet(args)



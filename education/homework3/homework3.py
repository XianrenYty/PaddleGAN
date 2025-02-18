#!/usr/bin/env python
# coding: utf-8

# # Day 3 作业--Pixel2Pixel：人像卡通化
# 
# 经过今天的学习，相信大家对图像翻译、风格迁移有了一定的了解啦，是不是也想自己动手来实现下呢？
# 
# 那么，为了满足大家动手实践的愿望，同时为了巩固大家学到的知识，我们Day 3的作业便是带大家完成一遍课程讲解过的应用--**Pixel2Pixel：人像卡通化**
# 
# 在本次作业中，大家需要做的是：**补齐代码，跑通训练，提交一张卡通化的成品图，动手完成自己的第一个人像卡通化的应用~**
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/6e3af14bf9f847ab92215753fb3b8f61a66186b538f44da78ca56627c35717b8)

# ## 准备工作：引入依赖 & 数据准备

# In[1]:


import paddle
import paddle.nn as nn
from paddle.io import Dataset, DataLoader

import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# ### 数据准备：
# 
# - 真人数据来自[seeprettyface](http://www.seeprettyface.com/mydataset.html)。
# - 数据预处理（详情见[photo2cartoon](https://github.com/minivision-ai/photo2cartoon)项目）。
# <div>
#   <img src='https://ai-studio-static-online.cdn.bcebos.com/c56c889827534363a8b6909d7737a1da64635ad33e1e44cb822f4c1cf1dfc689' height='1000px' width='1000px'>
# </div>
# 
# - 使用[photo2cartoon](https://github.com/minivision-ai/photo2cartoon)项目生成真人数据对应的卡通数据。

# In[ ]:


# 解压数据
get_ipython().system('unzip -oq data/data79149/cartoon_A2B.zip -d data/')


# ### 数据可视化

# In[2]:


# 训练数据统计
train_names = os.listdir('data/cartoon_A2B/train')
print(f'训练集数据量: {len(train_names)}')

# 测试数据统计
test_names = os.listdir('data/cartoon_A2B/test')
print(f'测试集数据量: {len(test_names)}')

# 训练数据可视化
imgs = []
for img_name in np.random.choice(train_names, 3, replace=False):
    img = cv2.imread('data/cartoon_A2B/train/'+img_name)
    imgs.append(img)
    print(img.shape)

img_show = np.vstack(imgs)[:,:,::-1]
plt.figure(figsize=(10, 10))
plt.imshow(img_show)
plt.show()


# In[3]:


class PairedData(Dataset):
    def __init__(self, phase):
        super(PairedData, self).__init__() 
        self.img_path_list = self.load_A2B_data(phase)    # 获取数据列表
        self.num_samples = len(self.img_path_list)        # 数据量

    def __getitem__(self, idx):
        img_A2B = cv2.imread(self.img_path_list[idx])     # 读取数据
        img_A2B = img_A2B.astype('float32') / 127.5 - 1.  # 归一化
        img_A2B = img_A2B.transpose(2, 0, 1)              # HWC -> CHW
        img_A = img_A2B[..., :256]                        # 真人照
        img_B = img_A2B[..., 256:]                        # 卡通图
        return img_A, img_B

    def __len__(self):
        return self.num_samples

    @staticmethod
    def load_A2B_data(phase):
        assert phase in ['train', 'test'], "phase should be set within ['train', 'test']"
        # 读取数据集，数据中每张图像包含照片和对应的卡通画。
        data_path = 'data/cartoon_A2B/'+phase
        return [os.path.join(data_path, x) for x in os.listdir(data_path)]


# In[4]:


paired_dataset_train = PairedData('train')
paired_dataset_test = PairedData('test')


# ## 第一步：搭建生成器
# 
# ### 请大家补齐空白处的代码，‘#’ 后是提示。

# 原始代码输出模型尺寸打印：
# 
# ![](https://ai-studio-static-online.cdn.bcebos.com/ca5008e8d9644ab8afe38fe9ba6d493b120e679151a1477384b4a779c027d98c)
# 

# In[5]:


class UnetGenerator(nn.Layer):
    def __init__(self, input_nc=3, output_nc=3, ngf=64):
        super(UnetGenerator, self).__init__()

        self.down1 = nn.Conv2D(input_nc, ngf, kernel_size=4, stride=2, padding=1)
        self.down2 = Downsample(ngf, ngf*2)
        self.down3 = Downsample(ngf*2, ngf*4)
        self.down4 = Downsample(ngf*4, ngf*8)
        self.down5 = Downsample(ngf*8, ngf*8)
        self.down6 = Downsample(ngf*8, ngf*8)
        self.down7 = Downsample(ngf*8, ngf*8)

        self.center = Downsample(ngf*8, ngf*8)

        self.up7 = Upsample(ngf*8, ngf*8, use_dropout=True)
        self.up6 = Upsample(ngf*8*2, ngf*8, use_dropout=True)
        self.up5 = Upsample(ngf*8*2, ngf*8, use_dropout=True)
        self.up4 = Upsample(ngf*8*2, ngf*8)
        self.up3 = Upsample(ngf*8*2, ngf*4)
        self.up2 = Upsample(ngf*4*2, ngf*2)
        self.up1 = Upsample(ngf*2*2, ngf)

        self.output_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2DTranspose(ngf*2, output_nc, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        # print(d1.shape)
        d2 = self.down2(d1)
        # print(d2.shape)
        d3 = self.down3(d2)
        # print(d3.shape)
        d4 = self.down4(d3)
        # print(d4.shape)
        d5 = self.down5(d4)
        # print(d5.shape)
        d6 = self.down6(d5)
        # print(d6.shape)
        d7 = self.down7(d6)
        # print(d7.shape)
        
        c = self.center(d7)
        
        x = self.up7(c, d7)
        # print(x.shape)
        x = self.up6(x, d6)
        # print(x.shape)
        x = self.up5(x, d5)
        # print(x.shape)
        x = self.up4(x, d4)
        # print(x.shape)
        x = self.up3(x, d3)
        # print(x.shape)
        x = self.up2(x, d2)
        # print(x.shape)
        x = self.up1(x, d1)
        # print(x.shape)

        x = self.output_block(x)
        return x


class Downsample(nn.Layer):
    # LeakyReLU => conv => batch norm
    def __init__(self, in_dim, out_dim, kernel_size=4, stride=2, padding=1):
        super(Downsample, self).__init__()

        self.layers = nn.Sequential(
            # LeakyReLU, leaky=0.2
            # Conv2D
            # BatchNorm2D
            nn.LeakyReLU(0.2),
            nn.Conv2D(in_dim, out_dim, kernel_size, stride, padding, bias_attr=False),
            nn.BatchNorm2D(out_dim)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class Upsample(nn.Layer):
    # ReLU => deconv => batch norm => dropout
    def __init__(self, in_dim, out_dim, kernel_size=4, stride=2, padding=1, use_dropout=False):
        super(Upsample, self).__init__()

        sequence = [
            # ReLU
            # Conv2DTranspose
            # nn.BatchNorm2D
            nn.ReLU(),
            # (o - k + 2 * p) / s + 1 = i -> o = (i-1) * s + k - 2 * p  
            # -> o = (i-1) * 2 + 4 - 2 * 1 = 2*i
            nn.Conv2DTranspose(in_dim, out_dim, kernel_size, stride, padding), 
            nn.BatchNorm2D(out_dim)
        ]

        if use_dropout:
            sequence.append(nn.Dropout(p=0.5))

        self.layers = nn.Sequential(*sequence)

    def forward(self, x, skip):
        x = self.layers(x)
        x = paddle.concat([x, skip], axis=1)
        return x


# ## 第二步：鉴别器的搭建
# 
# ### 请大家补齐空白处的代码，‘#’ 后是提示。

# In[6]:


class NLayerDiscriminator(nn.Layer):
    def __init__(self, input_nc=6, ndf=64):
        super(NLayerDiscriminator, self).__init__()

        self.layers = nn.Sequential(
            # (B, 6, 256, 256)
            nn.Conv2D(input_nc, ndf, kernel_size=4, stride=2, padding=1), 
            # (B, 64, 128, 128)
            # add BN
            nn.BatchNorm2D(ndf),
            nn.LeakyReLU(0.2),
            
            ConvBlock(ndf, ndf*2),
            # (B, 128, 64, 64)
            ConvBlock(ndf*2, ndf*4),
            # (B, 256, 32, 32)
            # change kernel_size
            ConvBlock(ndf*4, ndf*8, kernel_size=3, stride=1),
            # (B, 512, 32, 32)

            nn.Conv2D(ndf*8, 1, kernel_size=3, stride=1, padding=1),
            # (B, 1, 32, 32)
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.layers(input)


class ConvBlock(nn.Layer):
    # conv => batch norm => LeakyReLU
    def __init__(self, in_dim, out_dim, kernel_size=4, stride=2, padding=1):
        super(ConvBlock, self).__init__()

        self.layers = nn.Sequential(
            # Conv2D
            # BatchNorm2D
            # LeakyReLU, leaky=0.2
            nn.Conv2D(in_dim, out_dim, kernel_size, stride, padding, bias_attr=False),  # (i - 4 + 2 * 1) / 2 + 1 = i / 2 
            nn.BatchNorm2D(out_dim),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        x = self.layers(x)
        return x


# In[7]:


generator = UnetGenerator()
discriminator = NLayerDiscriminator()


# In[8]:


out = generator(paddle.ones([1, 3, 256, 256]))
print('生成器输出尺寸：', out.shape)  # 应为[1, 3, 256, 256]

out = discriminator(paddle.ones([1, 6, 256, 256]))
print('鉴别器输出尺寸：', out.shape)  # 应为[1, 1, 30, 30]  改为 [1, 1, 32, 32]


# In[9]:


# 超参数
LR = 3e-4  # 1e-4
BATCH_SIZE = 64
EPOCHS = 100

# 优化器
optimizerG = paddle.optimizer.Adam(
    learning_rate=LR,
    parameters=generator.parameters(),
    beta1=0.9, # 0.5
    beta2=0.999)

optimizerD = paddle.optimizer.Adam(
    learning_rate=LR,
    parameters=discriminator.parameters(), 
    beta1=0.9, # 0.5
    beta2=0.999)
    
# 损失函数
bce_loss = nn.BCELoss()
# l1_loss = nn.L1Loss() 
# change to smoothL1
l1_loss = nn.SmoothL1Loss()

# dataloader
data_loader_train = DataLoader(
    paired_dataset_train,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
    )

data_loader_test = DataLoader(
    paired_dataset_test,
    batch_size=BATCH_SIZE
    )


# In[10]:


results_save_path = 'work/results'
os.makedirs(results_save_path, exist_ok=True)  # 保存每个epoch的测试结果

weights_save_path = 'work/weights'
os.makedirs(weights_save_path, exist_ok=True)  # 保存模型

for epoch in range(EPOCHS):
    for data in tqdm(data_loader_train):
        real_A, real_B = data
        
        optimizerD.clear_grad()
        # D([real_A, real_B])
        real_AB = paddle.concat((real_A, real_B), 1)  # N C H W 在C维度上拼接
        d_real_predict = discriminator(real_AB)
        d_real_loss = bce_loss(d_real_predict, paddle.ones_like(d_real_predict))

        # D([real_A, fake_B])
        fake_B = generator(real_A).detach()
        fake_AB = paddle.concat((real_A, fake_B), 1)  # N C H W 在C维度上拼接
        d_fake_predict = discriminator(fake_AB)
        d_fake_loss = bce_loss(d_fake_predict, paddle.zeros_like(d_fake_predict))
        
        # train D
        d_loss = (d_real_loss + d_fake_loss) / 2.
        d_loss.backward()
        optimizerD.step()

        optimizerG.clear_grad()
        # D([real_A, fake_B])
        fake_B = generator(real_A)
        fake_AB = paddle.concat((real_A, fake_B), 1)
        g_fake_predict = discriminator(fake_AB)
        g_bce_loss = bce_loss(g_fake_predict, paddle.ones_like(g_fake_predict))
        g_l1_loss = l1_loss(fake_B, real_B) * 100.
        g_loss = g_bce_loss + g_l1_loss
        
        # train G
        g_loss.backward()
        optimizerG.step()

    print(f'Epoch [{epoch+1}/{EPOCHS}] Loss D: {d_loss.numpy()}, Loss G: {g_loss.numpy()}')

    if (epoch+1) % 10 == 0:
        paddle.save(generator.state_dict(), os.path.join(weights_save_path, 'epoch'+str(epoch+1).zfill(3)+'.pdparams'))

        # test
        generator.eval()
        with paddle.no_grad():
            for data in data_loader_test:
                real_A, real_B = data
                break

            fake_B = generator(real_A)
            result = paddle.concat([real_A[:3], real_B[:3], fake_B[:3]], 3)

            result = result.detach().numpy().transpose(0, 2, 3, 1)
            result = np.vstack(result)
            result = (result * 127.5 + 127.5).astype(np.uint8)

        # cv2.imshow(os.path.join(results_save_path, 'epoch'+str(epoch+1).zfill(3)+'.png'), result)
        cv2.imwrite(os.path.join(results_save_path, 'epoch'+str(epoch+1).zfill(3)+'.png'), result)

        generator.train()


# ## 最后：用你补齐的代码试试卡通化的效果吧！

# In[11]:


# 为生成器加载权重
last_weights_path = os.path.join(weights_save_path, sorted(os.listdir(weights_save_path))[-1])
print('加载权重:', last_weights_path)

model_state_dict = paddle.load(last_weights_path)
generator.load_dict(model_state_dict)
generator.eval()


# In[18]:


# 读取数据
test_names = os.listdir('data/cartoon_A2B/test')
for img_name in np.random.choice(test_names, 5, replace=False):
    img_A2B = cv2.imread('data/cartoon_A2B/test/'+img_name)
    img_A = img_A2B[:, :256]                                  # 真人照
    img_B = img_A2B[:, 256:]                                  # 卡通图

    g_input = img_A.astype('float32') / 127.5 - 1             # 归一化
    g_input = g_input[np.newaxis, ...].transpose(0, 3, 1, 2)  # NHWC -> NCHW
    g_input = paddle.to_tensor(g_input)                       # numpy -> tensor

    g_output = generator(g_input)
    g_output = g_output.detach().numpy()                      # tensor -> numpy
    g_output = g_output.transpose(0, 2, 3, 1)[0]              # NCHW -> NHWC
    g_output = g_output * 127.5 + 127.5                       # 反归一化
    g_output = g_output.astype(np.uint8)

    img_show = np.hstack([img_A, g_output])[:,:,::-1]
    plt.figure(figsize=(8, 8))
    plt.imshow(img_show)
plt.show()


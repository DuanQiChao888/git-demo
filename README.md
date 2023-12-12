'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

#创建一个argumentpaeser对象，用于解析命令行参数。description参数是一个可选的描述性字符串，用于描述程序的功能。
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
#向parser中添加一个参数规格说明。--lr是一个可选参数，defalut参数指定了默认值，type参数指定了参数的类型，help参数是一个可选的帮助字符串，用于描述参数的作用。
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
#向parser中添加另一个参数规格说明。--resume和-r都是可选参数，action参数制定了参数的行为，help参数是一个可选的帮助字符串，用于描述参数的作用。
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()   #解析命令行参数，并将结果存储在args变量中。

device = 'cuda' if torch.cuda.is_available() else 'cpu'  #检查是否有可用的gpu，有则使用gpu，否则使用cpu。
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
#这里定义了对训练数据进行的一系列图像转换操作，包括随机裁剪，随机水平翻转，转换为张量，归一化操作。
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),#随机裁剪图像，裁剪后的图像大小为32*32，同时在裁剪前进行4个像素的填充。
    transforms.RandomHorizontalFlip(),#随机水平翻转图像
    transforms.ToTensor(),   #将PIL图像转换为pytorch张量
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #对图像进行标准化处理，使得每个通道的均值为0.4914,0.4822,0.4465,标准差为）。2023，0.1994，0.2010.
])

#定义了一系列的数据预处理操作（图像转换），包括转换为张量以及归一化（标准化）操作。
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

#创建一个cifar-10数据集对象，用于训练模型。root参数指定数据集类型（训练集或测试集），download参数指定了是否需要下载数据集，transform参数指定增强操作。
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
#创建一个数据加载器，用于批量加载测试数据。batch_size参数指定了每个批次的大小，shuffle参数指定了是否需要打乱数据.num_workers参数指定了数据加载器使用的线程数。
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

#定义了cifar-10数据集中10个类别。
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
net = SimpleDLA()   #创建一个SimpleDLA对象
net = net.to(device)   #将模型移动到指定的设备上，如果有可用的GPU，则用GPU，否则用CPU
#如果使用的是GPU，则使用DataParallel将模型复制到多个GPU上，以加速训练。
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

#如果需要从之前的训练中恢复模型，则加载之前保存的模型参数。
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss() #定义交叉熵损失函数
optimizer = optim.SGD(net.parameters(), lr=args.lr,            #定义随机梯度下降（SGD）优化器，其中lr参数指定了学习率，momentum参数指定了动量,
                      momentum=0.9, weight_decay=5e-4)         #weight_decay参数指定了权重衰减。
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)  #定义余弦退火学习率调度器，其中T_max参数指定了最大迭代次数。


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()   #将模型设置为训练模式，启用dropout和batch normalization等层的功能
    train_loss = 0  #初始化训练损失为0
    correct = 0    #初始化正确预测的数量为0
    total = 0     #初始化总体样本数为0
    for batch_idx, (inputs, targets) in enumerate(trainloader):  #遍历训练数据集的每个批次，batch_idx是批次的索引，inputs是输入的图像,targets是真实的标签
        inputs, targets = inputs.to(device), targets.to(device)  #将输入和标签移动到设备（device)上
        optimizer.zero_grad()   #将优化器的梯度清零，避免累积
        outputs = net(inputs)   #将输入通过模型(net)得到输出，输出是每个类别的概率
        loss = criterion(outputs, targets)  #计算输出和标签之间的损失，使用预先定义的损失函数（criterion)
        loss.backward()  #计算损失对模型参数的梯度，并反向传播
        optimizer.step()  #使用优化器（optimizer)更新模型参数，沿着梯度的反方向

        train_loss += loss.item()  #累加每个批次的损失
        _, predicted = outputs.max(1) #得到每个样本的预测类别，是输出概率最大的那个
        total += targets.size(0) #累加每个批次的样本数
        correct += predicted.eq(targets).sum().item() #累加每个批次的正确预测的数量，使用eq函数比较预测和标签是否相等

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total)) #显示进度条，包括批次索引，批次总数，平均损失，准确率，正确预测数和总样本数


def test(epoch):
    global best_acc #声明一个全局变量，表示最佳的测试准确率
    net.eval() #将模型设置为评估模式，禁用dropout和batch normalization等层的功能
    test_loss = 0 #初始化测试损失为0
    correct = 0  #初始化正确预测的数量为0
    total = 0  #初始化总样本数为0
    with torch.no_grad(): #禁用自动微分，不需要计算梯度，节省内存和时间
        for batch_idx, (inputs, targets) in enumerate(testloader):   # 遍历测试数据集的每个批次，batch_idx是批次的索引，inputs是输入的图像，targets是真实的标签
            inputs, targets = inputs.to(device), targets.to(device) # 将输入和标签移动到设备（device）上，可以是CPU或GPU
            outputs = net(inputs) # 将输入通过模型（net）得到输出，输出是每个类别的概率
            loss = criterion(outputs, targets) # 计算输出和标签之间的损失，使用预先定义的损失函数（criterion）

            test_loss += loss.item() # 累加每个批次的损失
            _, predicted = outputs.max(1) # 得到每个样本的预测类别，是输出概率最大的那个
            total += targets.size(0) # 累加每个批次的样本数
            correct += predicted.eq(targets).sum().item() # 累加每个批次的正确预测的数量，使用eq函数比较预测和标签是否相等

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))  # 显示进度条，包括批次索引，批次总数，平均损失，准确率，正确预测数和总样本数

    # Save checkpoint.
    acc = 100.*correct/total # 计算测试准确率，乘以100转换为百分比
    if acc > best_acc:  # 如果当前的准确率大于之前的最佳准确率
        print('Saving..')  # 打印保存的提示
        state = {  # 创建一个字典，包含模型的参数，准确率和轮数
            'net': net.state_dict(),  # 模型的参数，使用state_dict()方法获取
            'acc': acc,   # 准确率
            'epoch': epoch,  # 轮数
        }
        if not os.path.isdir('checkpoint'):  # 如果不存在checkpoint文件夹
            os.mkdir('checkpoint')  # 创建一个checkpoint文件夹
        torch.save(state, './checkpoint/ckpt.pth')  # 使用torch.save()方法保存字典到checkpoint文件夹下的ckpt.pth文件
        best_acc = acc    # 更新最佳准确率为当前准确率


for epoch in range(start_epoch, start_epoch+200):   # 使用for循环，从start_epoch开始，到start_epoch+200结束，每次增加1，遍历每个轮数
    train(epoch) # 调用train函数，进行训练
    test(epoch)  # 调用test函数，进行测试
    scheduler.step()  # 调用scheduler的step方法，更新学习率

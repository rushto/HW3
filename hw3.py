'''引入模块'''

# 数据操作
import math
import numpy as np                  # 支持大量的维度数组与矩阵运算，以及对数组运算提供大量数学函数库
# 读取写入数据
import csv
import pandas as pd                 # 读取 Excel 文件并对读取的数据进行处理
import os                           # 提供与操作系统进行交互的接口，比如之后用的创建目录，保存文件
# Pytorch
import torch                        # 用于导入整个 PyTorch 库。
import torch.nn as nn               # 用于导入 PyTorch 的神经网络子模块，并简化对该子模块内容的访问。
import torchvision.transforms as transforms  # 提供常见的图像变换，用于数据增强和预处理。 用于对图像应用变换，如调整大小、裁剪、归一化和转换为张量。
# 图像处理
from PIL import Image                # Python Imaging Library（PIL）及其分支Pillow的一部分，用于打开、操作和保存图像文件。  有助于图像处理任务，如读取图像文件和转换格式。
# "ConcatDataset" 和"Subset" 会在半监督学习的数据用到
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset  # ConcatDataset 和 Subset用于合并多个数据集和创建数据集的子集。在半监督学习中很有用，比如你可能想合并有标签和无标签的数据集或拆分数据集。
from torchvision.datasets import DatasetFolder, VisionDataset   # PyTorch中用于处理特定类型的数据集。DatasetFolder用于通用的数据集结构，VisionDataset用于图像数据集，提供了一些额外的功能和约定。
# 进度条
from tqdm.auto import tqdm   # 常用于长时间运行的循环中，以直观地显示处理进度。  这种方式直接将 tqdm 函数导入到当前命名空间中，这样你可以直接使用 tqdm()，而不需要加上模块前缀。否则 tqdm.tqdm() 调用.
from tqdm import tqdm
import random                # 导入 Python 的内置模块 random。random 模块提供了生成伪随机数的功能，这些随机数在编写各种程序和算法时非常有用。



'''一些操作'''

# 设置随机种子(可以当模版)
def same_seeds(seed):                           # 它的输入 seed 是一个整数，我们可以在后面中的config里设置它。
    torch.backends.cudnn.deterministic = True   # 每次使用确定性的卷积算法，即默认算法
    torch.backends.cudnn.benchmark = False      # 默认值True，True会为整个网络的每一个卷积层选择一个最优的算法，实现整个网络加速，但是每次选择算法可能不一样，不能保证实验的可重复性
    np.random.seed(seed)                        # 确保每次使用Numpy随机函数时，生成随机数序列可重复
    torch.manual_seed(seed)                     # 假如用CPU训练，为CPU设置随机种子，方便复现结果
    if torch.cuda.is_available():               # 假如用GPU训练，为GPU设置随机种子，方便复现结果
        torch.cuda.manual_seed(seed)            # 为当前 GPU 设备设置随机数生成器的种子。
        torch.cuda.manual_seed_all(seed)        # 为所有可用的 GPU 设备设置随机数生成器的种子。


# 定义对图片的变换操作（数据增强）


# 一般情况下，我们不会在验证集和测试集上做数据扩增
# 1.将图片裁剪成同样的大小 2.转换成Tensor就行，并将像素值标准化到 0-1 之间。
test_tfm = transforms.Compose([        # 将多个图像变换组合在一起，以便依次对图像进行处理。  transforms.Compose 接受一个包含多个变换操作的列表，并返回一个组合变换。
    transforms.Resize((128, 128)),     # 图片裁剪 (height = width = 128)
    
    transforms.ToTensor(),             # 将 PIL 图像或 NumPy 数组转换为 PyTorch 张量（Tensor）。    该变换会将图像像素值从 0-255 的范围缩放到 0-1 之间，并将通道顺序从 (H, W, C) 转换为 (C, H, W)，其中 H 是高度，W 是宽度，C 是通道数
])                           

# 当然，我们也可以再测试集中对数据进行扩增（对同样本的不同装换）
#  - 用训练数据的装化方法（train_tfm）去对测试集数据进行转化，产出扩增样本
#  - 对同个照片的不同样本分别进行预测
#  - 最后可以用soft vote / hard vote 等集成方法输出最后的预测
train_tfm = transforms.Compose([
    transforms.Resize((128, 128)),     # 图片裁剪 (height = width = 128)

    # 图像增强
    transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),    # 使用针对 ImageNet 数据集优化的增强策略，随机应用各种数据增强技术，提高训练数据的多样性和模型的泛化能力。

    # ToTensor() 放在所有处理的最后
    transforms.ToTensor(),
])



'''Dataset'''
# 由于该数据集图片的名字就包含了类别（0_4.jpg代表第0类第4张），所以利用__getitem__来获取label
class FoodDataset(Dataset):

    # 构造函数，读取数据，对数据进行预处理。(self, 数据集所在的路径, 用于图像预处理的变换函数,指定的文件列表(默认为 None))
    def __init__(self,path,tfm=test_tfm,files = None):
        super(FoodDataset).__init__()
        self.path = path           # 将传入的路径 path 赋值给实例变量 self.path。
        self.files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")])   # 读取所有文件的名字，并按字母顺序排序，生成完整的文件路径列表
        if files != None:          # 如果传入的 files 参数不为 None，则使用 files 指定的文件列表覆盖 self.files。
            self.files = files     # （这允许用户在创建数据集实例时，提供特定的文件列表，而不是默认的从目录中读取所有文件。）
        print(f"One {path} sample",self.files[0])  # 打印文件列表中的第一个文件路径，以确认数据加载正确。
        self.transform = tfm       # 这些变换将在后续的数据加载过程中应用于每个图像，确保图像在输入模型前经过正确的预处理。
    
    # 用于获取数据集的长度（即样本的数量）
    def __len__(self):
        return len(self.files)
    
    # 每次从数据集里取出一笔数据（这个方法用于根据索引 idx 从数据集中获取一个样本（图像和标签））
    def __getitem__(self,idx):
        fname = self.files[idx]            # 根据索引 idx 获取文件名。
        im = Image.open(fname)             # 使用 PIL.Image.open 打开图像文件并读取图像。
        im = self.transform(im)            # 对图像应用预处理变换。（self.transform 是在初始化方法中设置的变换函数，用于对图像进行标准化、调整大小等预处理操作。）
        #im = self.data[idx]
        try:
            label = int(fname.split("/")[-1].split("_")[0])   # 尝试从文件名中提取标签，并将其转换为整数。
        except:
            label = -1 # test has no label         # 如果提取标签失败（如在测试集中没有标签），则将标签设为 -1
        return im,label           # 返回图像和标签



'''Model（模版）'''

# torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
# torch.nn.MaxPool2d(kernel_size, stride, padding)
# input 維度 [3, 128, 128]

# 基本块
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, pool_kernel_size=2, pool_stride=2, pool_padding=0):
        super(BasicBlock, self).__init__()
        self.block = nn.Sequential(               # Sequential容器（全连接和激活函数）
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(pool_kernel_size, pool_stride, pool_padding)
        )

    def forward(self, x):                        # 前向传播
        x = self.block(x)                        # x传入模型
        return x


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()

        self.cnn = nn.Sequential(
            BasicBlock(3, 64),       # [64, 128, 128]     # [64, 64, 64]
            BasicBlock(64, 128),     # [128, 64, 64]      # [128, 32, 32]
            BasicBlock(128, 256),    # [256, 32, 32]      # [256, 16, 16] 
            BasicBlock(256, 512),    # [512, 16, 16]      # [512, 8, 8]
            BasicBlock(512, 512)     # [512, 8, 8]        # [512, 4, 4]
        )

        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024), # 输入特征512*4*4
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size(0), -1)  # 将卷积层的输出展平为一维向量，以便输入全连接层。out.size()[0] 代表 batch_size
        out = self.fc(out)
        return out



'''定义参数'''

device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = {
    'seed': 6666,
    'dataset_dir': "./food11",
    'n_epochs': 10,      
    'batch_size': 64, 
    'learning_rate': 0.0003,           
    'weight_decay':1e-5,
    'early_stop': 300,
    'clip_flag': True, 
    'save_path': './models/model.ckpt',
}
print(device)



'''训练(模版)'''
def trainer(train_loader, valid_loader, model, config, device):
    '''  train_loader:训练集的DataLoader，用于批量加载训练数据。
           valid_loader:验证集的DataLoader，用于批量加载验证数据。
           model:要训练的神经网络模型。
           config:包含训练配置参数的字典，如学习率、训练轮数等。
           device:指定模型在哪个设备上进行训练,通常是CPU或GPU。
    '''


    # 对于分类任务, 我们常用cross-entropy评估模型表现.  (ignore_index=-1: 指定在计算损失时忽略的标签索引)。
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    # 初始化优化器(weight_decay=config['weight_decay']: 权重衰减（weight decay），用于 L2 正则化。它帮助防止过拟合。权重衰减值也是从配置字典 config 中获取的。)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay']) 
    # 模型存储位置
    save_path =  config['save_path'] 
    
    # 如果 ./models 目录不存在，则创建该目录
    if not os.path.isdir('./models'):
        os.mkdir('./models')

    
    n_epochs = config['n_epochs']     # 需要训练的n_epoch
    best_loss = math.inf              # 初始化为正无穷 (math.inf)，用于记录在训练过程中得到的最佳验证损失。
    # step = 0                          # 初始化为 0，用于跟踪训练步数。
    early_stop_count =  0             # 初始化为 0，用于记录早停计数。


    # 训练过程
    for epoch in range(n_epochs):
        model.train()
        loss_record = []           # 用于记录每个批次的训练损失。
        train_accs = []            # 用于记录每个批次的训练准确率。
        train_pbar = tqdm(train_loader, position=0, leave=True)   # 使用 tqdm 创建一个进度条，显示训练进度。position=0 表示进度条将显示在最顶部。leave 参数控制进度条在循环完成后是否保持在终端上。如果 leave=True，进度条在循环结束后将继续显示，通常以完成状态显示。

        for x, y in train_pbar:      # 遍历训练数据加载器中的每个批次。

            # 前馈
            optimizer.zero_grad()             
            x, y = x.to(device), y.to(device)  
            pred = model(x)             
            loss = criterion(pred, y)

            #反馈
            loss.backward()                   
            # 稳定训练的技巧
            '''梯度裁剪，以确保梯度的范数不会超过某个阈值，从而防止梯度爆炸。
               max_norm 是梯度范数的最大允许值。
               clip_grad_norm_ 会将所有参数的梯度缩放到一个不超过这个最大值的范围。
            '''
            if config['clip_flag']:  # 检查是否启用梯度裁剪。（对梯度进行裁剪，防止梯度爆炸。）
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

            # 更新
            optimizer.step()    

            

            acc = (pred.argmax(dim=-1) == y.to(device)).float().mean()  # 计算当前批次的训练准确率。
            l_ = loss.detach().item()    # 获取当前批次的损失值
            loss_record.append(l_)       #  记录当前批次的损失值。
            train_accs.append(acc.detach().item())   #  记录当前批次的准确率
            train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')   # 更新进度条的描述。
            train_pbar.set_postfix({'loss': f'{l_:.5f}', 'acc': f'{acc:.5f}'})   # 更新进度条的后缀，显示当前损失和准确率。
        
        
        mean_train_acc = sum(train_accs) / len(train_accs)    # 计算训练集上的平均准确率。
        mean_train_loss = sum(loss_record)/len(loss_record)   # 计算训练集上的平均损失。

        print(f'Epoch [{epoch+1}/{n_epochs}]: Train loss: {mean_train_loss:.4f},acc: {mean_train_acc:.4f} ')
        
        model.eval()                               # 设置模型为评估模式
        loss_record = []                           # 用于记录每个批次的验证损失。
        test_accs = []                             # 用于记录每个批次的验证准确率。
        for x, y in valid_loader:                  # 遍历验证数据加载器中的每个批次
            x, y = x.to(device), y.to(device)
            with torch.no_grad():

                # 前馈
                pred = model(x)
                loss = criterion(pred, y)
                acc = (pred.argmax(dim=-1) == y.to(device)).float().mean()
            loss_record.append(loss.item())         # 记录当前批次的验证损失值。
            test_accs.append(acc.detach().item())   # 记录当前批次的验证准确率。
            
        mean_valid_acc = sum(test_accs) / len(test_accs)   # 计算验证集上的平均准确率。
        mean_valid_loss = sum(loss_record)/len(loss_record)  # 计算验证集上的平均损失。

        # 打印当前轮次的训练和验证损失及准确率。
        print(f'Epoch [{epoch+1}/{n_epochs}]: Valid loss: {mean_valid_loss:.4f},acc: {mean_valid_acc:.4f} ')
        

        if mean_valid_loss < best_loss:      # 如果当前验证损失低于之前记录的最佳损失。
            best_loss = mean_valid_loss      # 更新最佳损失值。
            torch.save(model.state_dict(), save_path) # 保存最优模型
            print('Saving model with loss {:.3f}...'.format(best_loss))  # 打印保存模型的消息。
            early_stop_count = 0  # 重置早停计数器，因为模型性能改善。
        else: 
            early_stop_count += 1  # 早停计数器加一，表示模型性能没有改善的次数。
            if early_stop_count >= config['early_stop']:  # 如果早停计数器超过了设定的早停次数阈值，
               print('\nModel is not improving, so we halt the training session.')
               break                     # 终止训练过程，即提前结束训练。



'''训练准备'''

# 随机种子
same_seeds(config['seed'])

# 读取数据
_dataset_dir = config['dataset_dir']
# 训练（这个参数指定用于加载数据的子进程数量。如果 num_workers 为 0，数据加载将会在主进程中完成。）
train_set = FoodDataset(os.path.join(_dataset_dir,"training"), tfm=train_tfm)
train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
# 验证
valid_set = FoodDataset(os.path.join(_dataset_dir,"validation"), tfm=test_tfm)
valid_loader = DataLoader(valid_set, batch_size=config['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
# 测试
# 测试集保证输出顺序一致
test_set = FoodDataset(os.path.join(_dataset_dir,"test"), tfm=test_tfm)
test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, num_workers=0, pin_memory=True)
print(test_loader)
# 测试集数据扩增,使用数据增强来测试模型的鲁棒性
test_set = FoodDataset(os.path.join(_dataset_dir,"test"), tfm=train_tfm)
test_loader_extra1 = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

test_set = FoodDataset(os.path.join(_dataset_dir,"test"), tfm=train_tfm)
test_loader_extra2 = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

test_set = FoodDataset(os.path.join(_dataset_dir,"test"), tfm=train_tfm)
test_loader_extra3 = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, num_workers=0, pin_memory=True)



'''开始训练'''

model = Classifier().to(device)
trainer(train_loader, valid_loader, model, config, device)



'''预测'''

# 做预测（模版）
def predict(model, test_loader):

    pred = []  # 创建数组保存预测结果

    model.eval()
    for batch in tqdm(test_loader):
            
        features, _ = batch    # 解包元组
        print(type(features))  # 调试：打印 features 的类型
        print(features.shape)  # 调试：打印 features 的形状
        # print("1",features,features.shape)
        # features=torch.stack(features, dim=1)
        features = features.to(device)
        with torch.no_grad():
             # 前馈
            outputs = model(features)

            test_label = np.argmax(outputs.cpu().data.numpy(), axis=1)
            pred += test_label.squeeze().tolist()
    
    return pred


def pad4(i):
    return "0"*(4-len(str(i))) + str(i)


# 这段代码的主要作用是加载已经训练好的模型参数，然后使用该模型对测试集进行预测，并将预测结果保存到CSV文件中。
def save_pred(preds, file):                                       # 定义了一个名为 save_pred 的函数，接受两个参数 preds 和 file,分别表示预测结果和要保存的文件路径。
    with open(file, 'w') as fp:                                   # 使用 with 语句打开文件，指定打开方式为写入模式（'w'），并将文件对象赋值给 fp。
        writer = csv.writer(fp)                                   # 创建一个 CSV 文件写入器 writer，用于向文件中写入 CSV 格式的数据。
        writer.writerow(['Id', 'Category'])                       # 写入 CSV 文件的第一行，即表头，包含两列，分别为 'id' 和 'tested_positive'。
        for i, p in enumerate(preds):                             # 使用 enumerate 函数遍历预测结果 preds，获取索引 i 和对应的元素 p。   
            writer.writerow([pad4(i+1), p])                               # 将当前预测结果的索引和值写入到 CSV 文件中的一行中。


# 加载模型
model = Classifier().to(device)
model.load_state_dict(torch.load(config['save_path']))
# print(config['save_path'])

# 开始预测
preds = predict(model, test_loader)
print(preds)
save_pred(preds,'prdeiction.csv')
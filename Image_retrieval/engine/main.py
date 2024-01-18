import torch
import numpy as np
from Image_retrieval.Dataset.load_and_split_dataset import split_dataset
from Image_retrieval.Dataset.dataset_loader import ImageDataSet
from Image_retrieval.Dataset.sampler import RandomIdentitySampler
from torch.utils.data import DataLoader
from Image_retrieval.Model.model import Net
from Image_retrieval.Loss.combine_loss import triplet_and_center_and_cross_smooth
from Image_retrieval.Loss.center_loss import CenterLoss
from Image_retrieval.engine.train import train
from Image_retrieval.engine.test import inference

# 参数
path = r"C:\Users\90696\Desktop\deeplearning\deeplearning\DHAN-main\data\difffeature_matrix.npy"  # data: DCT histrogram features
L = 6
DCT_feature_dim = 1800
hidden_dim = 1024
deep_dim = 512
n_class = 100
cetner_loss_weight = 0.0001
EPOCH_NUM = 50

# 加载数据
Xp_train, yp_train, Xp_test, yp_test = split_dataset(path)
# 构建数据集
dataset = ImageDataSet(Xp_train, yp_train)
train_loader = DataLoader(dataset=dataset, batch_size=200, sampler=RandomIdentitySampler(list(zip(Xp_train, yp_train.numpy())), 200, 10))

# 网络模型
net = Net(n_feature=DCT_feature_dim, n_hidden1=hidden_dim, num_block=L, deep_dim=deep_dim, num_class=n_class).cuda()
print(net)  # net architecture

# 定义优化器
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-5)
center = CenterLoss(num_classes=100, feat_dim=512)
optimizer_center = torch.optim.Adam(center.parameters(), lr=0.01)


for epoch in range(EPOCH_NUM):
    loss = train(loss_fun=triplet_and_center_and_cross_smooth, optimizer=optimizer, model=net,
          train_loader=train_loader, center=center, optimizer_center=optimizer_center, cetner_loss_weight=cetner_loss_weight)
    mAP = inference(model=net, Xp_test=Xp_test, yp_test=yp_test)
    print('Epoch: ', epoch, '| train loss: %.4f' % loss, '| test mAP: %.4f' % np.mean(mAP))

torch.save(net.state_dict(), 'model_last.pth')

# post-processing reranking
# re_ranking()
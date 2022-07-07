import torch
import torchvision 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


# class Model(torch.nn.Module): 

#     def __init__(self):
#         super(Model,self).__init__()
#         self.base = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.DEFAULT)
#         self.base.avgpool = torch.nn.AdaptiveAvgPool2d((4,4))
#         self.base.classifier = torch.nn.Sequential(
#             torch.nn.Linear(8192,8192),
#             torch.nn.ReLU(inplace=True),
#             torch.nn.Linear(8192,256),
#             torch.nn.ReLU(inplace=True),
#             torch.nn.Linear(256, 14)
#         )

#         for param in self.base.features.parameters():
#             param.requires_grad = False

#     def forward(self,inputs):

#         x = self.base(inputs)
#         return x

if __name__=='__main__':

    # check for GPU
    if torch.backends.mps.is_available():
        print('MPS available')
        device = torch.device('mps')
    else:
        print('MPS not available')
        device = torch.device('cpu')

    # load in data 










    # # check for GPU
    # if not torch.backends.mps.is_available():
    #     print('MPS not available')
    # else:
    #     print('MPS available')
    #     mps_device = torch.device('mps')


    # # define transforms 
    # transforms = torchvision.transforms.Compose([
    #     torchvision.transforms.Resize((150,150)),
    #     torchvision.transforms.ToTensor()
    # ])
    # target_transform=torchvision.transforms.Lambda(lambda y: torch.zeros(17, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))


    # # load dataset
    # ds = torchvision.datasets.ImageFolder(
    #     root='data',
    #     transform = transforms,
    # )

    # # train-test split 
    # train_length = int(len(ds) * .9)
    # test_length = len(ds) - train_length
    # ds_train, ds_test = torch.utils.data.random_split(ds,[train_length, test_length])

    # # cross validation
    # lr = 1e-3 
    # k = 5
    # num_epochs = 20
    # loss_fn = torch.nn.CrossEntropyLoss()

    # kfold = KFold(n_splits=k,shuffle=True)

    # for fold, (train_ids, val_ids) in enumerate(kfold.split(ds_train)):

    #     print(f'FOLD {fold}')
    #     print('---------------------------------\n')

    #     # train val split 
    #     train_subset = torch.utils.data.SubsetRandomSampler(train_ids)
    #     val_subset = torch.utils.data.SubsetRandomSampler(val_ids)
    #     dl_train = torch.utils.data.DataLoader(ds_train, batch_size=8,sampler=train_subset)
    #     dl_val = torch.utils.data.DataLoader(ds_train, batch_size=8,sampler=val_subset)

    #     # initialize model 
    #     model = Model().to(mps_device)
    #     optimizer = torch.optim.Adam(model.parameters(),lr=lr)


    #     # train loop 
    #     stats = {
    #         'train_loss_final': [],
    #         'val_loss_final': [],
    #         'train_acc_final': [],
    #         'val_acc_final': []
    #     }
    #     for epoch in range(0,num_epochs):

    #         print(f'Epoch {epoch + 1}\n---------------------------------')

    #         num_correct = 0
    #         acc = 0

    #         # loop through batches 
    #         for batch_idx, (x, y) in enumerate(dl_train): 
    #             x = x.to(mps_device)
    #             y = y.to(mps_device)

    #             train_size = len(dl_train.dataset)

    #             # forward pass 
    #             pred = model(x)

    #             # compute loss
    #             loss = loss_fn(pred, y)

    #             # backprop
    #             optimizer.zero_grad()
    #             loss.backward()

    #             # update weights
    #             optimizer.step()

    #             _, pred_label = torch.max(pred, dim=1)
                
    #             num_correct += (pred_label==y).type(torch.float).sum().item()
    #             loss = loss.item()
    #             current = batch_idx * len(x)

    #             if batch_idx > 0: 
    #                 acc = num_correct / current * 100

                
    #             if batch_idx % 8 == 0 and batch_idx > 0: 
    #                 # print(f'current: {current}')
    #                 # print(f'num_corrent: {num_correct}')
    #                 print(f'[{current:>5d}/{train_size:>5d}] loss: {loss:>7f} accuracy: {acc:>7f}')

    #         stats['train_acc_final'].append(acc)
    #         stats['train_loss_final'].append(loss)

    #         num_batches = len(dl_val)
    #         val_size = len(dl_val.dataset)
    #         val_loss = 0
    #         correct = 0

    #         with torch.no_grad():
    #             for x,y in dl_val:
    #                 x = x.to(mps_device)
    #                 y = y.to(mps_device)

    #                 # forward pass
    #                 pred = model(x)

    #                 # compute loss
    #                 val_loss += loss_fn(pred, y)
    #                 correct += (pred.argmax(1)==y).type(torch.float).sum().item()

    #         val_loss /= num_batches
    #         val_acc = correct / val_size * 100

    #         print(f'Val Error: \n Accuracy: {val_acc:>0.1f}%, Avg loss: {val_loss:>8f} \n')


    #         stats['val_acc_final'].append(val_acc)
    #         stats['val_loss_final'].append(val_loss)
            
                






    
    

    






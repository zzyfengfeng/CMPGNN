#ensure that PyTorch 1.2.0 and torch-geometric 1.3.2 are installed
import torch
from torch_geometric.nn.conv import MessagePassing
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool,global_mean_pool
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import add_self_loops,degree,remove_self_loops
from torch.nn import Sequential, LeakyReLU, Linear,Sigmoid,LogSoftmax,BatchNorm1d,GRUCell
import torch.optim as optim
from torch_geometric.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from torch.nn.modules import Module
from torch_scatter import scatter_mean
import numpy as np
from torch_geometric.transforms import OneHotDegree
import torch_geometric.data

class ConvBlock(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__(aggr='add')  # "add" aggregation.
        self.message_generation = Sequential(Linear(3 * in_channels, out_channels, bias=True),
                                             LeakyReLU(),
                                             Linear(out_channels, out_channels, bias=True),
                                             LeakyReLU())
        self.local_gate = Sequential(Linear(out_channels, out_channels, bias=True),
                                     Sigmoid())
        self.local_gate_loss = []
    def forward(self, x, edge_index, batch, **kwargs):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        return self.propagate(edge_index=edge_index, size=(x.size(0), x.size(0)), x=x, batch=batch,global_information=kwargs['global_information'])


    def message(self, x_i, x_j, edge_index, batch, global_information):
        gf = global_information[batch[edge_index[1]]]
        x = torch.cat([x_i, x_j, gf], dim=1)
        msg = self.message_generation(x)
        local_w = self.local_gate(msg) * msg
        w_msg = local_w * msg  #####################
        self.local_gate_loss = torch.sum(local_w ** 2, dim=-1, keepdim=True) ** 0.5  ######################
        return w_msg
    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]
        return aggr_out, self.local_gate_loss


class ReadoutBlock(Module):
    def __init__(self, in_channels, out_channels):
        super(ReadoutBlock, self).__init__()

        self.mlp = Sequential(Linear(in_channels * 2, in_channels, bias=True),
                              Sigmoid())
        self.gru = GRUCell(out_channels, out_channels)
        self.global_gate_loss = []
        self.bn = BatchNorm1d(in_channels)

    def forward(self, x, batch, global_information):
        global_w = self.mlp(torch.cat([x, global_information[batch]], dim=-1))
        gf_new = global_add_pool(global_w * x, batch)
        gf = self.gru(gf_new, global_information)
        self.global_gate_loss = torch.mean(global_mean_pool(torch.sum(global_w ** 2, dim=-1) ** 0.5, batch))
        return gf, self.global_gate_loss

class ClfBlock(Module):
    def __init__(self, in_channels, hidden_layer_channels, out_channels):
        super(ClfBlock, self).__init__()
        self.bn = BatchNorm1d(in_channels)
        self.mlp = Sequential(Linear(in_channels, hidden_layer_channels),
                              LeakyReLU(),
                              Linear(hidden_layer_channels, out_channels),
                              LogSoftmax(dim=-1))
    def forward(self, x, batch):
        x = self.bn(x)
        x = self.mlp(x)
        return x


class CMPGNN_Net(torch.nn.Module):
    def __init__(self, num_features,
                 num_conv_blocks,
                 conv_blocks_in_channels,
                 conv_blocks_out_channels,
                 clf_block_channels):
        super(CMPGNN_Net, self).__init__()
        self.conv_blocks = torch.nn.ModuleDict()
        self.gru_hidden_size = conv_blocks_out_channels[-1]
        self.fc = Linear(num_features, conv_blocks_in_channels[0])
        for i in range(num_conv_blocks):
            self.conv_blocks['block' + str(i)] = ConvBlock(conv_blocks_in_channels[i], conv_blocks_out_channels[i])
            # self.bn_blocks['block'+str(i)]=BatchNorm1d(bn_block_in_channels[i])
        self.rd_block = ReadoutBlock(conv_blocks_out_channels[-1], conv_blocks_out_channels[-1])
        self.clf_block = ClfBlock(conv_blocks_out_channels[-1], clf_block_channels[0], clf_block_channels[1])
        self.num_conv_blocks = num_conv_blocks
    def forward(self, x, edge_index, batch):
        local_gate_loss_list = []  ################3
        global_gate_loss_list = []
        global_feature_list = []
        x = self.fc(x)
        gf = torch.zeros(torch.max(batch) + 1, self.gru_hidden_size).to(x.device)
        for i in range(self.num_conv_blocks):
            gf, ggl = self.rd_block(x, batch, gf)
            global_feature_list.append(gf)
            x, lgl = self.conv_blocks['block' + str(i)](x, edge_index, batch=batch, global_information=gf)
            local_gate_loss_list.append(lgl)
            global_gate_loss_list.append(ggl)
        gf, ggl = self.rd_block(x, batch, gf)
        global_gate_loss_list.append(ggl)
        local_gate_loss_list = torch.cat(local_gate_loss_list, dim=-1)  ###################
        x = self.clf_block(gf, batch)
        return x, local_gate_loss_list, global_gate_loss_list

def cmpgnn_loss_fn(lgl_list, ggl_list, edge_index, batch, y_p, y_t,lbd):  ###################3
    nllloss = F.nll_loss(y_p, y_t)
    if torch.isnan(nllloss):
        print('nan')
    lgl = torch.sum(torch.abs(lgl_list), dim=-1).to(torch.float)
    dg = degree(edge_index[1])
    lgl = lgl / (dg[edge_index[1]])
    lgl = scatter_mean(scatter_mean(lgl, edge_index[1]), batch)
    return nllloss + lbd * torch.mean(lgl) + lbd * torch.mean(torch.tensor(ggl_list))  # 0.01

def one_epoch_train(net,optimizer,data_loader,num_samples,device,lbd):
    net.train()
    loss_epoch=0.0
    for data in data_loader:
        if data.num_graphs==1:
            ds=[]
            for i in range(2):
                ds.append(torch_geometric.data.Data(x=data.x,edge_index=data.edge_index,y=data.y))
            data=iter(DataLoader(ds, batch_size=2, shuffle=False)).next()
            optimizer.zero_grad()
            data.to(device)
            edge_index, _ = remove_self_loops(data.edge_index)
            edge_index, _ = add_self_loops(edge_index)
            output, lgl_list, ggl_list = net(data.x, edge_index, data.batch)
            loss = cmpgnn_loss_fn(lgl_list, ggl_list, edge_index, data.batch, output, data.y,lbd)
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()
            #print('single graph train')
        else:
            optimizer.zero_grad()
            data.to(device)
            edge_index, _ = remove_self_loops(data.edge_index)
            edge_index, _ = add_self_loops(edge_index)
            output, lgl_list, ggl_list = net(data.x, edge_index, data.batch)
            loss = cmpgnn_loss_fn(lgl_list, ggl_list, edge_index, data.batch, output, data.y,lbd)
            loss.backward()
            optimizer.step()
            loss_epoch += loss.item() * data.num_graphs

    return loss_epoch/num_samples

def test(net,data_loader,num_samples,device):
    net.eval()
    acc=0.0
    with torch.no_grad():
        for data in data_loader:
            if data.num_graphs == 1:
                ds = []
                for i in range(2):
                    ds.append(torch_geometric.data.Data(x=data.x, edge_index=data.edge_index, y=data.y))
                data = iter(DataLoader(ds, batch_size=2, shuffle=False)).next()
                data = data.to(device)
                edge_index, _ = remove_self_loops(data.edge_index)
                edge_index, _ = add_self_loops(edge_index)
                output, _, __ = net(data.x, edge_index, data.batch)
                _, pre = output.topk(1)
                pre = pre.squeeze(-1)
                acc += float(pre.eq(data.y).sum().item())/2
                print('single graph test')
            else:
                data = data.to(device)
                edge_index, _ = remove_self_loops(data.edge_index)
                edge_index, _ = add_self_loops(edge_index)
                output, _, __ = net(data.x, edge_index, data.batch)
                _, pre = output.topk(1)
                #print(pre)
                #print(data.y)
                pre = pre.squeeze(-1)
                acc += float(pre.eq(data.y).sum().item())

    return acc/num_samples

def main_process(seed, dataname,H_DIM,lbd,transform_flag=False,LR = 0.001,EPOCHS = 100,BATCHSIZE = 20,NUM_CONV_BLOCKS = 3):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if transform_flag==True:
        dataset = TUDataset(root='data/' + dataname, name=dataname ,transform=OneHotDegree(1000))
    else:
        dataset = TUDataset(root='data/' + dataname, name=dataname)
    conv_blocks_in_channels = [H_DIM, H_DIM, H_DIM]
    conv_blocks_out_channels = [H_DIM, H_DIM, H_DIM]
    clf_block_channels = [H_DIM, dataset.num_classes]
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    acc_rate_list = []
    k=0
    for train_index, test_index in skf.split(dataset, dataset.data.y):
        k=k+1
        print(k)

        train_set = [dataset[int(index)] for index in train_index]
        test_set = [dataset[int(index)] for index in test_index]

        train_data_loader = DataLoader(train_set, batch_size=BATCHSIZE, shuffle=True)
        test_data_loader = DataLoader(test_set, batch_size=BATCHSIZE)
        net = CMPGNN_Net(dataset.num_features, NUM_CONV_BLOCKS, conv_blocks_in_channels, conv_blocks_out_channels,
                         clf_block_channels).to(device)
        optimizer = optim.Adam(net.parameters(), lr=LR,weight_decay=0)
        lr_schedule = optim.lr_scheduler.StepLR(optimizer, 20, 0.5)
        for epoch in range(EPOCHS):
            loss_epoch = one_epoch_train(net, optimizer, train_data_loader, len(train_set),device,lbd)
            #print('epoch:' + str(epoch) + '   loss:' + str(loss_epoch))
            lr_schedule.step()
        acc_rate = test(net, test_data_loader, len(test_set),device)
        print(acc_rate)

        acc_rate_list.append(acc_rate)

    acc_records = np.array(acc_rate_list)
    print(dataname)
    print('acc records', acc_records)
    print(dataname+" acc mean:{}".format(acc_records.mean()))
    print(dataname+" acc std:{}".format(acc_records.std()))

main_process(0,'MUTAG',8,0.01,False)
#main_process(0,'IMDB-MULTI',8,0.01,True)
#main_process(0,'PROTEINS',12,0.01,False)
#main_process(0,'PTC_MR',16,0.02,False)
#main_process(0,'COLLAB',16,0.1,True)
#main_process(0,'IMDB-BINARY',6,0.5,True)

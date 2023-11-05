import torch
import torch.nn as nn
import torch.nn.functional as F
# import esig as es
import torch.autograd as autograd
# import iisignature
import signatory
import os
import pandas as pd


def test_1():
    # input (minibatch,in_channels,iW)
    # output(out_channels, in_channels/groups(default=1),  kW(kernel size))

    conv1 = nn.Conv1d(in_channels=256, out_channels=100, kernel_size=2)
    input = torch.randn(32, 35, 256)
    # batch_size x text_len x embedding_size -> batch_size x embedding_size x text_len
    input = input.permute(0, 2, 1)
    out = conv1(input)
    print(out.size())
    # 这里32为batch_size，35为句子最大长度，256为词向量


def test():
    mu = torch.Tensor([1,2,3,4,5])
    eps = mu.mul(0).normal_()
    print(eps)

def test_sum():
    t = torch.ones([2,2,3])
    print(t)
    s = torch.sum(t * t, dim=[1, 2])
    print(s.size())
    print(s)

def test_conv1():
    inputs = torch.randn(33, 16, 30)
    filters = torch.randn(20, 16, 1)
    out = F.conv1d(inputs, filters)
    print(out.size())


class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()

    def forward(self, x):
        return F.avg_pool1d(x,kernel_size=2)
        # return F.max_pool1d(x, kernel_size=x.shape[2])  # shape: (batch_size, channel, 1)


def cnn():
    cnn_layers = nn.ModuleList()
    for c, k in zip([4,5], [2,3]):
        cnn = nn.Sequential(
            nn.Conv1d(in_channels=25,
                      out_channels=c,
                      kernel_size=k),
            nn.BatchNorm1d(c),
            nn.ReLU(inplace=True),
        )
        cnn_layers.append(cnn)
        # 最大池化层
    pool = GlobalMaxPool1d()
    # 输出层
    classify = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(9, 2)
    )
    return cnn_layers, pool, classify

def adaptive_cnn():
    m = nn.AdaptiveMaxPool1d(8)
    input = torch.randn(1, 64, 5)
    output = m(input)
    print(output.size())

def test_attention():
    # [batch_size, num_post in a group, length of post, dimension]
    input = torch.ones([4,4]) * -1
    mask = torch.tensor([[0,1,1,1],
                         [1,0,1,1],
                         [1,1,0,1],
                         [1,1,1,0]])
    input_one = torch.ones_like(input)
    change_inpput = input * mask + input_one
    print(input)
    print(change_inpput)
    # print(mask.size())
    # new_input = input * \
    # mask.unsqueeze(-1).unsqueeze(-1)
    #
    # print(new_input.size(), new_input)

def normal_test():
    input = torch.ones([4,4])
    input[:, 2:] = 0.
    print(input)
    # [2,4,3]
    ii = torch.tensor([[[1,2,3],[4,5,6],[7,8,9],[10, 11, 12]],
                       [[1,0,3],[4,0,6],[7,8,0],[0, 11, 12]]])
    ii = ii.view(2,2,-1,3).view(4,-1,3)
    print(ii.size(), ii)

def test_list():
    test = [1,2,3,4,5,6,7]
    for i in enumerate(test):
        print(i)

def test_sig():
    z = torch.rand((7,20,10))
    # print(z)
    # s = es.stream2logsig(z, depth=3)
    # s = iisignature.sig(z, 3)
    # s = torch.from_numpy(s)
    # print(s.size(), 's -=-=-=-=-=-=-=')
    signature2 = signatory.LogSignature(depth=3, stream=False)
    print('之心')
    out = signature2(z)
    print(out.size())


def read_prompt():
    dir = 'data/amazon/reddit_prompt/train'
    files = os.listdir(dir)

    for file in files:
        if file == '.DS_Store':
            continue

        path = os.path.join(dir, file)

        data = pd.read_csv(path, sep='\t')

        # head = data.columns
        # if len(head) == 4:
        #     continue
        # else:

        col = data['prompt']
        for p in col:
            if type(p) == float:
                print(file, '===========')


def test_cosin():
    a = torch.rand((2,4,3))
    b = torch.rand((2,3))
    b = b.unsqueeze(1)
    b = b.repeat(1,4,1)
    print(b.size(), 's -=-=-=-=-')
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    s = cos(a,b)
    print(s.size(), s, 'sssss')

def test_reshpe():
    a = torch.tensor([[[1,2,3,4],[23,4,5,6],[2,3,4,5]],
                      [[1,2,3,4],[23,4,5,6],[2,3,4,5]]])
    print(a.size())
    bs = a.size()[0]
    dimen = a.size()[-1]
    a = a.view(bs, -1)
    print(a.size(), 'a -=--=--=-=-')
    print(a)



if __name__ == '__main__':

    test_reshpe()



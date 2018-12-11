import csv
import os
import random

import cv2
import numpy
import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam, lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vgg16_bn


class CutSeq(Dataset):

    def __init__(self, data_dir: str, index_file: str, tchw=(16, 3, 224, 224), show: bool = False):
        """
        class is a frame number in the sequence in which cut is made

        :param data_dir: 
        :param index_file: 
        :param tchw: 
        :param show: 
        """
        self.show = show
        self.tchw = tchw
        self.index_file = index_file
        self.data_dir = data_dir
        self.cls_weights = numpy.zeros(shape=(tchw[0] + 1))

        self.t = tchw[0]
        self.t_half = int(self.t / 2)

        self.data_xy = []

        with open(index_file, 'r') as _f:
            for row in csv.reader(_f):
                # charmed20,5014.png,5015.png,0
                dir, start, end, is_cut = row

                if 'landmark-breaks' == dir:
                    continue

                if show:
                    print("row", row)
                cut_frame_idx = random.randint(0, (self.t - 1))
                y = numpy.zeros(shape=self.t + 1)  # +1 is for negative class

                x = []
                cut_fn = int(end.replace(".png", ""))
                start_fn = int(start.replace(".png", ""))
                if int(is_cut) == 1:
                    # positive sample
                    y[cut_frame_idx] = 1.
                    s_frame = cut_fn - cut_frame_idx
                    for _fn in range(s_frame, cut_fn):
                        f = os.path.join(dir, "%s.png" % _fn)
                        x.append(f)

                    e_frame = cut_fn + (self.t - cut_frame_idx)
                    for _fn in range(cut_fn, e_frame):
                        f = os.path.join(dir, "%s.png" % _fn)
                        x.append(f)

                else:
                    # negative sample
                    cut_frame_idx = self.t
                    y[cut_frame_idx] = 1.
                    jitter = 0
                    if cut_fn - start_fn > self.t:
                        jitter = random.randint(0, cut_fn - start_fn - self.t)

                    _s = start_fn + jitter
                    for _fn in range(_s, _s + self.t):
                        f = os.path.join(dir, "%s.png" % _fn)
                        x.append(f)

                self.cls_weights[cut_frame_idx] += 1

                if self.show:
                    print("datum:", start, end)
                    print("cut idx:", cut_frame_idx)
                    print("x:", x)
                    print()
                assert len(x) == self.t, "actual len is %d" % len(x)

                self.data_xy.append((x, y))

            self.cls_weights = Tensor(self.cls_weights).float()

            print("Class counts", self.cls_weights)
            self.cls_weights = self.cls_weights / self.cls_weights.max()
            self.cls_weights = 1. - torch.nn.Softmax(dim=-1)(self.cls_weights)
            print("Class weights", self.cls_weights)

            random.shuffle(self.data_xy)

    def __getitem__(self, index):
        frame_files, y = self.data_xy[index]
        # channel_first!
        x = numpy.zeros(shape=self.tchw, dtype=numpy.float)

        if self.show:
            print("cut idx", numpy.argmax(y))

        for i, f in enumerate(frame_files):
            png_fn = os.path.join(self.data_dir, f)
            # print(png_fn)

            bgr = cv2.imread(png_fn, 1)

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            _w = self.tchw[-1]
            _h = self.tchw[-2]
            rgb = cv2.resize(rgb, (_w, _h))

            if self.show:
                cv2.imshow("frame %d" % index, cv2.cvtColor(rgb.astype("uint8"), cv2.COLOR_RGB2BGR))
                cv2.waitKey(25)
                # wait input on new cut
                # if y[i] == 1:
                #     cv2.waitKey(0)

            # _x = rgb / 127.5 - 1
            _x = rgb
            _x = numpy.moveaxis(_x, -1, 0)  # channel first
            x[i] = _x

        tempo2d = temporal_slice_X(x, self.show)
        return tempo2d, y

    def __len__(self):
        return len(self.data_xy)


def temporal_slice_X(x_tchw: numpy.ndarray, show=False):
    t, c, h, w = x_tchw.shape

    xy_grid_step = 1
    h_t = int(h / xy_grid_step)

    tempo_map = numpy.zeros(shape=(t * h_t, w, c))

    for _y in range(0, h, xy_grid_step):
        slice_X = x_tchw[:, :, _y, :]
        slice_X = numpy.moveaxis(slice_X, 1, 2)
        # print(_y, slice_X.max())
        tempo_map[t * _y:t * (_y + 1), :, :] = slice_X

    if show:
        cv2.imshow("tempo_map", cv2.cvtColor(tempo_map.astype("uint8"), cv2.COLOR_RGB2BGR))
        cv2.imwrite("/tmp/tempo_map.png", cv2.cvtColor(tempo_map.astype("uint8"), cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)

    # t=16 -> 4x4
    result_w = 4 * w
    result_h = 4 * h_t
    tempo_4x4 = numpy.zeros(shape=(result_h, result_w, c))

    for j in range(1, 5):
        for i in range(1, 5):
            # print(j, i)
            tempo_4x4[i * h_t - h_t:i * h_t, j * w - w:j * w, :] = tempo_map[i * j * h_t - h_t:i * j * h_t, :, :]

    tempo_4x4 = cv2.resize(tempo_4x4, (224, 224))
    if show:
        cv2.imshow("tempo_4x4", cv2.cvtColor(tempo_4x4.astype("uint8"), cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)

    return tempo_4x4


def temporal_slice_X_1x16(x_tchw: numpy.ndarray, show=False):
    t_size, c, h, w = x_tchw.shape

    xy_grid_step = 1
    h_t = int(h / xy_grid_step)
    tempo_map = numpy.zeros(shape=(t_size * h_t, w, c))

    for _y in range(0, h, xy_grid_step):
        slice_X = x_tchw[:, :, _y, :]
        slice_X = numpy.moveaxis(slice_X, 1, 2)
        # print(_y, slice_X.max())
        tempo_map[t_size * _y:t_size * (_y + 1), :, :] = slice_X

    if show:
        cv2.imshow("tempo_map", cv2.cvtColor(tempo_map.astype("uint8"), cv2.COLOR_RGB2BGR))
        cv2.imwrite("/tmp/tempo_map.png", cv2.cvtColor(tempo_map.astype("uint8"), cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
    return tempo_map


def temporal_slices(x_spacial: numpy.ndarray):
    t_size, w, h, c = x_spacial.shape
    seq_twhc = numpy.zeros(shape=(t_size, w, h, c), dtype=numpy.uint8)
    xy_grid_step = 4

    for t in range(0, t_size):
        rgb_whc = x_spacial[t]
        seq_twhc[t, :, :, :] = rgb_whc

        # cv2.imshow("space frame", seq_twhc[max(0, t), :, :, :])

        for _y in range(0, h, xy_grid_step):
            slice_X = seq_twhc[:, :, _y, :]
            # cv2.imshow("space-timeX", slice_X)
            # cv2.waitKey(0)

        for _x in range(0, w, xy_grid_step):
            slice_Y = seq_twhc[:, _x, :, :]
            # cv2.imshow("space-timeY", slice_Y)
            # cv2.waitKey(0)

    return seq_twhc


class C3D(nn.Module):
    def __init__(self):
        super(C3D, self).__init__()
        self.group1 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))
        # init.xavier_normal(self.group1.state_dict()['weight'])
        self.group2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        # init.xavier_normal(self.group2.state_dict()['weight'])
        self.group3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        # init.xavier_normal(self.group3.state_dict()['weight'])
        self.group4 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        # init.xavier_normal(self.group4.state_dict()['weight'])
        self.group5 = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        # init.xavier_normal(self.group5.state_dict()['weight'])

        ncls = 17
        self.fc1 = nn.Sequential(
            # nn.Linear(512 * 3 * 3, 32*16),  #
            nn.Linear(25088, 32 * ncls),  #
            nn.ReLU(),
            nn.Dropout(0.5))
        # init.xavier_normal(self.fc1.state_dict()['weight'])
        self.fc2 = nn.Sequential(
            nn.Linear(32 * ncls, 32 * ncls),
            nn.ReLU(),
            nn.Dropout(0.5))
        # init.xavier_normal(self.fc2.state_dict()['weight'])
        self.fc3 = nn.Sequential(
            nn.Linear(32 * ncls, ncls))  # 101

        self._features = nn.Sequential(
            self.group1,
            self.group2,
            self.group3,
            self.group4,
            self.group5
        )

        self._classifier = nn.Sequential(
            self.fc1,
            self.fc2
        )

    def forward(self, x):
        out = self._features(x)
        out = out.view(out.size(0), -1)
        out = self._classifier(out)
        return self.fc3(out)


def weights_init(m: nn.Module):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)


if __name__ == '__main__':
    num_epochs = 100
    batch_size = 1
    learning_rate = 0.003

    # train_dataset = CutSeq(data_dir="/Volumes/bstorage/datasets/scenes",
    #                       index_file="/Volumes/bstorage/datasets/scenes/train.txt",
    #                       tchw=(2, 3, 224*2, 224*2),
    #                       show=False)

    train_dataset = CutSeq(data_dir="/kote/svlk/anton/fades_rgb2",
                           index_file="/kote/svlk/anton/fades_rgb2/DAWSONS_CREEK_206.csv",
                           tchw=(16, 3, 224, 224),
                           show=False)
    # for x in range(0, 100):
    #     train_dataset[x]
    # sys.exit(1)

    data = DataLoader(train_dataset, batch_size=batch_size)

    # c3d = C3D().cpu()
    # c3d = C3D()
    # print(c3d)
    # print("total params:", sum([param.nelement() for param in c3d.parameters()]))
    # c3d.apply(weights_init)
    # m = c3d
    # print("c3d loaded")

    if torch.cuda.is_available():
        print("Using CUDA")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vgg16 = vgg16_bn(pretrained=True, )

    # Freeze training for all layers
    for param in vgg16.features.parameters():
        param.require_grad = False

    classes = 16  # temporal space

    # Newly created modules have require_grad=True by default
    num_features = vgg16.classifier[6].in_features
    features = list(vgg16.classifier.children())[:-1]  # Remove last layer
    features.extend([nn.Linear(num_features, classes + 1)])  # Add our layer with 4 outputs
    vgg16.classifier = nn.Sequential(*features)  # Replace the model classifier

    # m = models.vgg16(pretrained=True, ).to(device)
    m = vgg16.to(device)
    print(m)

    criterion = nn.CrossEntropyLoss(weight=Tensor(train_dataset.cls_weights))
    # optimizer = SGD(c3d.parameters(), lr=learning_rate, momentum=0.9)
    optimizer = Adam(m.parameters(), lr=learning_rate)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    for b in iter(data):
        y_true = Variable(b[1]).long().to(device)

        _x = b[0]
        # _x = _x.permute(0, 2, 1, 3, 4)  # B,C,T,HW
        _x = _x.permute(0, 3, 1, 2)  # B,C,H,W
        x = Variable(_x, requires_grad=True).float().to(device)

        optimizer.zero_grad()
        print("x.shape: ", x.shape)
        y = m.forward(x)
        print("outputs: ", y)
        print("   true: ", y_true)

        inp = torch.argmax(y, 1)
        target = torch.argmax(y_true, 1)
        print("target ", target)
        loss = criterion(y, target)
        loss.backward()
        optimizer.step()

        print("step loss:", loss)

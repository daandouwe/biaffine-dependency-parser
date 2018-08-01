#!/usr/bin/env python
import torch
import torch.nn as nn
from torch.autograd import Variable

from torch.nn.parallel import data_parallel

class MultiGPULossCompute:
    """A multi-gpu loss compute and train function."""
    def __init__(self, model, optimizer, devices, output_device=None):
        self.model = model
        self.optimizer = optimizer
        self.devices = devices
        self.output_device = output_device if output_device is not None else devices[0]

    def __call__(self, input):
        # data_parallel(self.model, input, device_ids=self.devices)
        replicas = nn.parallel.replicate(self.model, self.devices)
        print(replicas)
        inputs = nn.parallel.scatter(input, self.devices)
        print(inputs)
        outputs = nn.parallel.parallel_apply(replicas, inputs)
        print(outputs)
        return nn.parallel.gather(outputs, self.output_device)
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()


class Model(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        # print("Device {}.".format(torch.cuda.current_device()))
        return self.linear(x)


if __name__ == '__main__':
    model = Model(5, 10)
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters())
    devices = [0, 1, 2, 3]

    input = Variable(torch.randn(8, 5).cuda())

    par_model = nn.DataParallel(model, devices)
    par_model(input)

    train_fn = MultiGPULossCompute(model, optimizer, devices)
    # train_fn(input)

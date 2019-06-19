from torch import nn

search_space = dict(
    base=[
        nn.Conv2d(in_channels=3,out_channels=16,
            kernel_size=3,stride=2,padding=1,bias=False),
        nn.BatchNorm2d(16),
        nn.MaxPool2d(kernel_size=2,stride=2,padding=0), 
        nn.ReLU(inplace=True)
            ],
    depth=[4, 4, 4],
    out=[1,2],
    space=[
        dict(
            type="ResNetBottleneck",
            param=dict(
            _in=16,_out=16,
            kernel_size=1,padding=0,
            expansion=1)),
        dict(
            type="ResNetBottleneck",
            param=dict(
            _in=16,_out=16,
            kernel_size=3,padding=1,
            expansion=1)),
        dict(
            type="ResNetBottleneck",
            param=dict(
            _in=16,_out=16,
            kernel_size=5,padding=2,
            expansion=1)),
        dict(
            type="ResNetBottleneck",
            param=dict(
            _in=16,_out=16,
            kernel_size=7,padding=3,
            expansion=1))
    ]
    )

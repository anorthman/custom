

search_space = dict(
    base=dict(
            in_channels=3,out_channels=16,
            kernel_size=3,stride=2,padding=1,bias=False
            ),
    depth=[4, 5, 6],#, 2],
    #out_indices=[0,1,2,3],
    space=[
        dict(
            type="ResNetBasicBlock",
            param=dict(
            _in=16,_out=16,
            kernel_size=1,padding=0,
            expansion=1)),
        dict(
            type="ResNetBasicBlock",
            param=dict(
            _in=16,_out=16,
            kernel_size=3,padding=1,
            expansion=1)),
        dict(
            type="ResNetBasicBlock",
            param=dict(
            _in=16,_out=16,
            kernel_size=5,padding=2,
            expansion=1)),
        dict(
            type="ResNetBasicBlock",
            param=dict(
            _in=16,_out=16,
            kernel_size=7,padding=3,
            expansion=1))
    ]
    )

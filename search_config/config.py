

search_space = dict(
    depth=[2, 2, 2, 2],
    space=[
        dict(
            type="ResNetBottleneck",
            param=dict(
            _in=3,_out=8,
            expansion=1)),
        dict(
            type="ResNetBottleneck",
            param=dict(
            _in=3,_out=8,
            expansion=1))
    ]
    )

def foo(x: 'b,d,t'):
    y: 'b,d' = x.mean(dim=0)  # error!         [line 37]
    z: 'b,d' = x.mean(dim=1) #shape check: ok! [line 38]

def test_foo():
    import torch
    x = torch.Tensor(10, 100, 1024)
    foo(x)

def setup_named_dims():
    from tsalib import dim_vars
    #declare the named dimension variables using the tsalib api
    #e.g., 'b' stands for 'Batch' dimension with size 10
    dim_vars('Batch(b):10 Length(t):100 Hidden(d):1024')

    # initialize tsanley's dynamic shape analyzer
    from tsanley.dynamic import init_analyzer
    init_analyzer(trace_func_names=['foo'], show_updates=True) #check_tsa=True, debug=False


if __name__ == '__main__':
    setup_named_dims()
    test_foo()

from torch2trt.torch2trt import *
from torch2trt.module_test import add_module_test


@tensorrt_converter('torch.topk')
def covnert_topk(ctx):
    input = ctx.method_args[0]
    k = get_arg(ctx, 'k', pos=1, default=1)
    # Currently only values up to 3840 are supported.
    assert k <= 3840
    dim = get_arg(ctx, 'dim', pos=2, default=len(input.shape) - 1)
    if dim < 0:
        dim += len(input.shape)
    assert dim > 0, 'dim cannot be the batch dimension'
    # controls whether to return largest or smallest elements
    largest = get_arg(ctx, 'largest', pos=3, default=True)

    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    topk_op = trt.TopKOperation.MAX if largest else trt.TopKOperation.MIN
    layer = ctx.network.add_topk(input_trt, topk_op, k, torch_dim_to_trt_axes(dim))

    output_val = ctx.method_return[0]
    output_idx = ctx.method_return[1]
    output_val._trt = layer.get_output(0)
    output_idx._trt = layer.get_output(1)


@tensorrt_converter('torch.where')
def convert_where(ctx):
    cond = ctx.method_args[0]
    then_input = ctx.method_args[1]
    else_input = ctx.method_args[2]
    output = ctx.method_return

    inputs_trt = add_missing_trt_tensors(ctx.network, [cond])
    inputs_trt += add_missing_trt_tensors(ctx.network, [then_input, else_input])
    layer = ctx.network.add_select(*inputs_trt)
    output._trt = layer.get_output(0)


@tensorrt_converter('torch.index_select')
def convert_index_select(ctx):
    """trt.gather default mode
    """
    input = ctx.method_args[0]
    dim = get_arg(ctx, 'dim', pos=1, default=0) # ctx.method_args[1]
    if dim < 0:
        dim += len(input.shape)
    index = get_arg(ctx, 'index', pos=2, default=None) # ctx.method_args[2]

    input_trt = add_missing_trt_tensors(ctx.network, [input])[0]
    index_trt = add_missing_trt_tensors(ctx.network, [index])[0]
    layer = ctx.network.add_gather(input_trt, index_trt, dim)

    output = ctx.method_return
    output._trt = layer.get_output(0)


# @tensorrt_converter('torch.gather')
# def convert_gather(ctx):
#     input = ctx.method_args[0]
#     dim = ctx.method_args[1]
#     if dim < 0:
#         dim += len(input.shape)
#     index = ctx.method_args[2]

#     input_trt, index_trt = add_missing_trt_tensors(ctx.network, [input, index])
#     layer = ctx.network.add_gather(input_trt, index_trt, torch_dim_to_trt_axes(dim))

#     output = ctx.method_return
#     output._trt = layer.get_output(0)


@tensorrt_converter('torch.mm')
@tensorrt_converter('torch.matmul')
def convert_matmul(ctx):
    input0 = ctx.method_args[0]
    input1 = ctx.method_args[1]

    op0, op1 = trt.MatrixOperation.NONE, trt.MatrixOperation.NONE
    input0_trt, input1_trt = add_missing_trt_tensors(ctx.network, [input0, input1])
    layer = ctx.network.add_matrix_multiply(input0_trt, op0, input1_trt, op1)

    output = ctx.method_return
    output._trt = layer.get_output(0)

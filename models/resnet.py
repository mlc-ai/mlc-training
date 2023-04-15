from time import monotonic
from typing import List

import numpy as np
import numpy.random

import tvm
from tvm import relax
from tvm.relax import Var
from tvm.relax import training
from tvm.relax.block_builder import BlockBuilder
from tvm.relax.expr import Expr
from tvm.script.parser import ir as I
from tvm.script.parser import relax as R
from tvm.script.parser import tir as T
from tvm._ffi.registry import get_global_func
from tvm.meta_schedule.relax_integration import tune_relax
from tvm.meta_schedule.runner.runner import Runner
from tvm.meta_schedule.runner.utils import alloc_argument_common
from tvm.runtime.ndarray import NDArray
from tvm.relax.transform.tuning_api import Trace


# Constants
fdtype = "float32"
batch = 32
lr = 0.1


# TrainerContext Class
class TrainerContext:
    bb: BlockBuilder
    param_list: List[Var]
    p_default: List[np.ndarray]
    state_list: List[Var]
    updated_state_list: List[Var]
    s_default: List[np.ndarray]

    def __init__(self, bb):
        self.bb = bb
        self.param_list = []
        self.p_default = []
        self.state_list = []
        self.updated_state_list = []
        self.s_default = []

    def __getattr__(self, name):
        return getattr(self.bb, name)

    def add_param(self, var, default_val):
        self.param_list.append(var)
        self.p_default.append(default_val)

    def add_state(self, var, updated_var, default_val):
        self.state_list.append(var)
        self.updated_state_list.append(updated_var)
        self.s_default.append(default_val)

    def emit_output_list(self, exprs: List[Expr]):
        ret = []
        for e in exprs:
            ret.append(self.bb.emit_output(e))
        return ret

    def func_params(self):
        return self.param_list + self.state_list

    def mod_attrs(self):
        return {"param_num": len(self.param_list), "state_num": len(self.state_list)}


def get_np_shape(expr):
    return [int(i) for i in expr.struct_info.shape]


# Model definition using TrainerContext
def Conv2d(
    ctx: TrainerContext, input, in_channel, out_channel, kernel_size, stride, padding=0
):
    weight = relax.Var(
        "conv2d_weight",
        R.Tensor((out_channel, in_channel, kernel_size, kernel_size), fdtype),
    )

    # kaiming init
    bound = 1.0 / np.sqrt(in_channel * kernel_size * kernel_size)
    ctx.add_param(
        weight,
        numpy.random.uniform(-bound, bound, size=get_np_shape(weight)).astype(fdtype),
    )

    res = ctx.emit(R.nn.conv2d(input, weight, stride, padding))
    return res


def BatchNorm2d(ctx: TrainerContext, input, channel):
    gamma = relax.Var("bn_gamma", R.Tensor((channel,), fdtype))
    beta = relax.Var("bn_beta", R.Tensor((channel,), fdtype))
    moving_mean = relax.Var("bn_mm", R.Tensor((channel,), fdtype))
    moving_var = relax.Var("bn_mv", R.Tensor((channel,), fdtype))

    ctx.add_param(gamma, np.ones(get_np_shape(gamma)).astype(fdtype))
    ctx.add_param(beta, np.zeros(get_np_shape(beta)).astype(fdtype))

    bn = ctx.emit(R.nn.batch_norm(input, gamma, beta, moving_mean, moving_var))
    res, new_moving_mean, new_moving_var = (
        ctx.emit(bn[0]),
        ctx.emit(bn[1]),
        ctx.emit(bn[2]),
    )

    ctx.add_state(
        moving_mean, new_moving_mean, np.zeros(get_np_shape(moving_mean), fdtype)
    )
    ctx.add_state(
        moving_var, new_moving_var, np.ones(get_np_shape(moving_mean), fdtype)
    )

    return res


def Linear(ctx: TrainerContext, input, in_feature, out_feature):
    weight = relax.Var("ln_weight", R.Tensor((in_feature, out_feature), fdtype))
    bias = relax.Var("ln_bias", R.Tensor((out_feature,), fdtype))

    bound = 1.0 / np.sqrt(in_feature)
    ctx.add_param(
        weight,
        numpy.random.uniform(-bound, bound, size=get_np_shape(weight)).astype(fdtype),
    )
    ctx.add_param(
        bias,
        numpy.random.uniform(-bound, bound, size=(get_np_shape(bias))).astype(fdtype),
    )

    res = ctx.emit(R.matmul(input, weight) + bias)
    return res


def BasicBlock(ctx: TrainerContext, input, in_planes, planes, stride=1):
    expansion = 1
    conv1 = Conv2d(ctx, input, in_planes, planes, 3, stride, 1)
    bn1 = BatchNorm2d(ctx, conv1, planes)
    relu1 = ctx.emit(R.nn.relu(bn1))
    conv2 = Conv2d(ctx, relu1, planes, planes, 3, 1, 1)
    bn2 = BatchNorm2d(ctx, conv2, planes)
    shortcut = input
    if stride != 1 or in_planes != expansion * planes:
        conv3 = Conv2d(ctx, input, in_planes, expansion * planes, 1, stride)
        shortcut = BatchNorm2d(ctx, conv3, expansion * planes)
    relu2 = ctx.emit(R.nn.relu(bn2 + shortcut))
    return relu2


def get_expansion(block):
    return 1 if block is BasicBlock else 4


def ResNet_layer(
    ctx: TrainerContext, input, block, in_planes, planes, num_blocks, stride
):
    strides = [stride] + [1] * (num_blocks - 1)
    for stride in strides:
        input = block(ctx, input, in_planes, planes, stride)
        in_planes = planes * get_expansion(block)
    return input, in_planes


def ResNet(ctx: TrainerContext, input, block, num_blocks, num_classes=10):
    in_planes = 64
    conv1 = Conv2d(ctx, input, 3, 64, 3, 1, 1)
    bn1 = BatchNorm2d(ctx, conv1, 64)
    relu1 = ctx.emit(R.nn.relu(bn1))
    layer1, in_planes = ResNet_layer(ctx, relu1, block, in_planes, 64, num_blocks[0], 1)
    layer2, in_planes = ResNet_layer(
        ctx, layer1, block, in_planes, 128, num_blocks[1], 2
    )
    layer3, in_planes = ResNet_layer(
        ctx, layer2, block, in_planes, 256, num_blocks[2], 2
    )
    layer4, in_planes = ResNet_layer(
        ctx, layer3, block, in_planes, 512, num_blocks[3], 2
    )
    pool = ctx.emit(R.nn.avg_pool2d(layer4, 4, 1, 0, ceil_mode=False))
    reshape = ctx.emit(R.reshape(pool, (pool.struct_info.shape[0], -1)))
    linear = Linear(ctx, reshape, 512 * get_expansion(block), num_classes)
    return linear


def ResNet18(ctx: TrainerContext, input):
    return ResNet(ctx, input, BasicBlock, [2, 2, 2, 2])


# Wrap the model backbone into an IRModule
bb = BlockBuilder()
ctx = TrainerContext(bb)
input = relax.Var("input", R.Tensor((batch, 3, 32, 32), fdtype))
input_list = [input]

with bb.function("backbone"):
    with bb.dataflow():
        result = ResNet18(ctx, input)
        ret = ctx.emit_output_list([result] + ctx.updated_state_list)
    bb.emit_func_output(ret, input_list + ctx.func_params())

Backbone = bb.get()
Backbone = Backbone.with_attrs(ctx.mod_attrs())

# Show the backbone
# Backbone.show()


# Setup Trainer
out_sinfo = relax.TensorStructInfo((batch, 10), fdtype)
label_sinfo = relax.TensorStructInfo((batch,), "int64")

setup_trainer = training.SetupTrainer(
    training.loss.CrossEntropyLoss(),
    training.optimizer.MomentumSGD(lr, 0.9, weight_decay=5e-4),
    [out_sinfo, label_sinfo],
)

train_mod = setup_trainer(Backbone)

# Show the module after SetupTrainer
# print(train_mod.without_attr("optim_state").script())

# CPU Training
# target, dev = "llvm", tvm.cpu()

# GPU Training
# If the model is trained under GPU, the module must be tuned. CPU do not need the tuning process.
target, dev = tvm.target.Target("nvidia/geforce-rtx-3080"), tvm.cuda()


# Tune utils definition
def random_fill(data: NDArray):
    random_fill_for_measure = get_global_func(
        "tvm.contrib.random.random_fill_for_measure"
    )
    if "int" in data.dtype:
        new_data = np.zeros(data.shape, dtype=data.dtype)
        data.copyfrom(new_data)
    else:
        random_fill_for_measure(data)


def alloc_argument(device, args_info, alloc_repeat):
    return alloc_argument_common(random_fill, device, args_info, alloc_repeat)


runner = Runner.create("local", f_alloc_argument=alloc_argument)


# Tuning and ApplyDB

# with tempfile.TemporaryDirectory() as work_dir:
work_dir = "/home/yxdong/relax-mlcai/other-repos/AD-Example/tmp/tune"
with target, tvm.transform.PassContext(trace=Trace(train_mod)):
    # start_time = monotonic()
    # tune_relax(train_mod, {}, target, work_dir, 10000, runner=runner)
    # print(f"Tune time {monotonic() - start_time} seconds")

    start_time = monotonic()
    tuned_mod = relax.transform.MetaScheduleApplyDatabase(work_dir)(train_mod)
    print(f"ApplyDB time {monotonic() - start_time} seconds")


# Build the module
start_time = monotonic()
ex = relax.build(tuned_mod, target)
print(f"Build time {monotonic() - start_time} seconds")
vm = relax.VirtualMachine(ex, dev)


# Trainer setup
trainer = training.Trainer(train_mod, vm, dev)
trainer.load_params(ctx.p_default)
trainer.load_states(ctx.s_default)


# Test the predict and the update function
input = np.ones((batch, 3, 32, 32)).astype(fdtype)
label = np.zeros((batch,)).astype("int64")

start_time = monotonic()
res1 = trainer.predict(input)
print(f"Run time {monotonic() - start_time} seconds")

print(res1)

start_time = monotonic()
res2 = trainer.update([input], [label])
print(f"Run time {monotonic() - start_time} seconds")

print(res2)


# Data preparation using torch.utils.data.DataLoader
import torch.utils.data
import torchvision
import torchvision.transforms as transforms

# Data
print("==> Preparing data..")
transform_train = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

transform_test = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)

trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_train
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=batch, shuffle=True, num_workers=2, drop_last=True
)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform_test
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=batch, shuffle=False, num_workers=2, drop_last=True
)

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


# Training function
def train(epoch):
    start_time = monotonic()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        loss = trainer.update(inputs.numpy(), targets.numpy())

    print(
        f"Train: Epoch = {epoch}, run time {monotonic() - start_time} seconds, final loss={loss}"
    )


# Testing function
def test(epoch):
    total = 0
    correct = 0
    start_time = monotonic()
    for batch_idx, (inputs, targets) in enumerate(testloader):
        predicted = trainer.predict(inputs.numpy())
        total += targets.size(0)
        correct += np.sum(predicted.numpy() == targets.numpy())

    print(
        f"Evaluate: Epoch = {epoch}, run time {monotonic() - start_time} seconds, "
        f"acc: {correct/total:.2f}, ({correct}/{total}"
    )


for epoch in range(200):
    train(epoch)
    test(epoch)

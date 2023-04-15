from typing import List
import numpy as np
import tvm
from tvm.relax.expr import Function, Var
from tvm.relax.training.loss import CrossEntropyLoss
from tvm.relax.training.setup_trainer import SetupTrainer
from tvm.relax.training.trainer import Trainer
import tvm.script
from tvm import relax
from tvm.script.parser import relax as R, ir as I
from tvm.relax.training.optimizer import SGD

batch_size = 64

import torch
import torch.utils.data
import torchvision
import torchvision.transforms as Tr
import torch.nn.functional as Func

train_data = torchvision.datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=Tr.Compose([Tr.ToTensor(), Tr.Lambda(torch.flatten)]),
    target_transform=lambda x: Func.one_hot(torch.tensor(x), 10).float(),
)
test_data = torchvision.datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=Tr.Compose([Tr.ToTensor(), Tr.Lambda(torch.flatten)]),
    target_transform=lambda x: Func.one_hot(torch.tensor(x), 10).float(),
)
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True, drop_last=True
)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=batch_size, shuffle=False, drop_last=True
)
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

import matplotlib.pyplot as plt

img, label = next(iter(train_loader))
img = img[0].reshape(1, 28, 28).numpy()
plt.figure()
plt.imshow(img[0])
plt.colorbar()
plt.grid(False)
plt.show()
print("Class:", class_names[label.argmax()])


@I.ir_module
class Backbone:
    @R.function
    def predict(input, weight, bias, old_running_mean, old_running_var):
        R.func_attr({"params_num": 2, "states_num": 2})
        with R.dataflow():
            dense = input * weight + bias
            pred, new_running_mean, new_running_var = R.nn.batch_norm(
                dense, ..., old_running_mean, old_running_var
            )
            R.output(pred, new_running_mean, new_running_var)
        return pred, new_running_mean, new_running_var


pred_sinfo = relax.TensorStructInfo((batch_size, 10), "float32")
label_sinfo = relax.TensorStructInfo((batch_size, 10), "float32")


class Loss:
    ...


class CrossEntropyLoss(Loss):
    def __init__(self, reduction, ignore_index):
        ...

    def __call__(self, *param_sinfo) -> Function:
        ...


loss = CrossEntropyLoss(reduction="sum")(
    pred_sinfo,
    label_sinfo,
)


class Optimizer:
    ...

class SGD(Optimizer):
    def __init__(self, lr, weight_decay):
        ...

    def init(self, params: List[Var]) -> "SGD":
        ...

    def get_function(self) -> Function:
        ...

    @property
    def state(self) -> tvm.ir.Array:
        ...

x = relax.Var("x", R.Tensor((3, 3), "float32"))
opt = SGD(0.01).init(x)
func = opt.get_function()

@R.function
def SGD(
    params: R.Tuple(R.Tensor((3, 3), "float32")),
    gradients: R.Tuple(R.Tensor((3, 3), "float32")),
    optim_states: R.Tuple(R.Tensor((), "int64")),
):
    with R.dataflow():
        num_steps = optim_states[0]
        num_steps_new = num_steps + R.const(1, "int64")
        x = params[0]
        x_grad = gradients[0]
        lv = R.multiply(R.const(0.01, "float32"), x_grad)
        x_new = R.subtract(x, lv)
        params_new = (x_new,)
        optim_states_new: R.Tuple(R.Tensor((), "int64")) = (num_steps_new,)
        R.output(params_new, optim_states_new)
    return (params_new, optim_states_new)

pred, new_states = 1, 1


@I.ir_module
class Module:
    @R.function
    def predict(inputs, params, states):
        return pred, new_states



params_adjoints = 1
new_params, new_optim_states = 1, 1

@I.ir_module
class Module:
    @R.function
    def predict(inputs, params, states):
        return pred, new_states

    @R.function
    def predict_loss(inputs, params, states, labels):
        return loss

    @R.function
    def predict_loss_adjoint(inputs, params, states, labels):
        return loss, (params_adjoints)

    @R.function
    def optimize(params, params_adjoints, optim_states):
        return new_params, new_optim_states


    PREDICT_FUNC_NAME: str = "predict"
    LOSS_FUNC_NAME: str = "predict_loss"
    ADJOINT_FUNC_NAME: str = "predict_loss_adjoint"
    UPDATE_PARAMS_FUNC_NAME: str = "update_params"



setup_trainer = SetupTrainer(loss, opt, [out_sinfo, label_sinfo])


trainer = Trainer(Backbone, 6, setup_trainer)
# build the IRModule in the trainer
trainer.build(target="llvm", device=tvm.cpu(0))


trainer.xaiver_uniform_init_params()


epochs = 5
log_interval = 200


for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        loss = trainer.update(data.numpy(), target.numpy())

        if batch_idx % log_interval == 0 or batch_idx == len(train_loader):
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.numpy():.2f}"
            )

    total, correct = 0, 0
    for data, target in test_loader:
        predict = trainer.predict(data.numpy())  # batch_size * 10
        total += len(data)
        correct += np.sum(predict.numpy().argmax(1) == target.numpy().argmax(1))

    print(
        f"Train Epoch: {epoch} Accuracy on test dataset: {100.0 * correct / total:.2f}%"
    )

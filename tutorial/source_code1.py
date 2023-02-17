import numpy as np
import tvm
from tvm.relax.training.loss import CrossEntropyLoss
from tvm.relax.training.setup_trainer import SetupTrainer
from tvm.relax.training.trainer import Trainer
import tvm.script
from tvm import relax
from tvm.script.parser import relax as R
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
    target_transform=lambda x:Func.one_hot(torch.tensor(x), 10).float()
)
test_data = torchvision.datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=Tr.Compose([Tr.ToTensor(), Tr.Lambda(torch.flatten)]),
    target_transform=lambda x:Func.one_hot(torch.tensor(x), 10).float()
)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=True)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

import matplotlib.pyplot as plt

img, label = next(iter(train_loader))
img = img[0].reshape(1, 28, 28).numpy()
plt.figure()
plt.imshow(img[0])
plt.colorbar()
plt.grid(False)
plt.show()
print("Class:", class_names[label.argmax()])

@tvm.script.ir_module
class Backbone:
    @R.function
    def predict(
        w0: R.Tensor((784, 128), "float32"),
        b0: R.Tensor((128,), "float32"),
        w1: R.Tensor((128, 128), "float32"),
        b1: R.Tensor((128,), "float32"),
        w2: R.Tensor((128, 10), "float32"),
        b2: R.Tensor((10,), "float32"),
        x: R.Tensor((batch_size, 784), "float32"),
    ) -> R.Tensor((batch_size, 10), "float32"):
        with R.dataflow():
            lv0 = R.matmul(x, w0)
            lv1 = R.add(lv0, b0)
            lv2 = R.nn.relu(lv1)
            lv3 = R.matmul(lv2, w1)
            lv4 = R.add(lv3, b1)
            lv5 = R.nn.relu(lv4)
            lv6 = R.matmul(lv5, w2)
            out = R.add(lv6, b2)
            R.output(out)
        return out


loss = CrossEntropyLoss(reduction="sum")
opt = SGD(0.01, weight_decay=0.01)


out_sinfo = relax.TensorStructInfo((batch_size, 10), "float32")
label_sinfo = relax.TensorStructInfo((batch_size, 10), "float32")

setup_trainer = SetupTrainer(loss, opt, [out_sinfo, label_sinfo])


trainer = Trainer(Backbone, 6, setup_trainer)
# build the IRModule in the trainer
trainer.build(target="llvm", device=tvm.cpu(0))


trainer.xaiver_uniform_init_params()


epochs = 5
log_interval = 200


for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        loss = trainer.update_params(data.numpy(), target.numpy())

        if batch_idx % log_interval == 0 or batch_idx == len(train_loader):
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.numpy():.2f}")

    total, correct = 0, 0
    for data, target in test_loader:
        predict = trainer.predict(data.numpy()) # batch_size * 10
        total += len(data)
        correct += np.sum(predict.numpy().argmax(1) == target.numpy().argmax(1))

    print(f"Train Epoch: {epoch} Accuracy on test dataset: {100.0 * correct / total:.2f}%")

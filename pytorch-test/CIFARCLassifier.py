import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
import foolbox as fb
import torchvision
from torchvision.datasets import CIFAR10
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn.utils import prune
from copy import deepcopy

class CIFARCLassifier(pl.LightningModule):
    def __init__(self):
        super(CIFARCLassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 16, 5, padding=2)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 5, padding=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.do1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, 10)
        self.cw = fb.attacks.L2CarliniWagnerAttack()

    def forward(self, x):
        x = self.pool(self.bn1(F.relu(self.conv1(x))))
        x = self.pool(self.bn2(F.relu(self.conv2(x))))
        x = self.pool(self.bn3(F.relu(self.conv3(x))))
        x = self.flatten(x)
        x = self.do1(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)


    def perform_pruning(self):
        prune.random_unstructured(module=self.fc1, name='weight', amount=0.2)


    def prepare_data(self):
        #download the data, augment and normalize them
        self.cifar_train = CIFAR10(root="data/", train=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(0.5, 0.5)]))

        self.cifar_val = CIFAR10(root="data/", train=False, transform=torchvision.transforms.Compose([ torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(0.5, 0.5)]))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.cifar_train, batch_size=128, shuffle=True, num_workers=12)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.cifar_val, batch_size=128, shuffle=False, num_workers=12)

    def training_step(self, train_batch, batch_idx):
        #1 batch of training
        data, label = train_batch
        out = self.forward(data)
        loss = F.nll_loss(out, label)
        prediction = out.argmax(dim=1, keepdim=True).squeeze()

        correct =  prediction.eq(label.view_as(prediction)).sum().item()
        return {'loss' : loss, 'correct' : correct}
    
    def training_epoch_end(self, outputs):
        #end of an epoch
        #calculate the average loss of this epoch
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        accuracy = 100 * (sum([x['correct'] for x in outputs]) / float(len(self.cifar_train)))

        logs = {'train_loss': avg_loss, 'train_accuracy': accuracy}
        '''
        -----------------------WARNING------------------------------------
        at the end of the training step, set the final model as the current 
        weight set. Don't forget to zero_grad() as the attack requires using
        the graients of the parameters, we are going to set the context to be 
        torch.enable_grad() as during the evaluation, the weight tensors are so to not
        store the gradient information. However, when we enable the gradient
        the gradients with respect to the training loss function are stored in
        the tensor whereas the adversarial loss function is different, and this makes the
        the optimization completely break apart.
        -----------------------WARNING------------------------------------
        ''' 

        return {'loss' : avg_loss, 'log' : logs}


    def validation_step(self, validation_batch, batch_idx):
        if batch_idx == 0:
            torch.cuda.memory_summary()
        
        data, label = validation_batch
        #standard inference
        out = self.forward(data)
        loss = F.nll_loss(out, label)
        prediction = out.argmax(dim=1, keepdim=True).squeeze()
        correct =  prediction.eq(label.view_as(prediction)).sum().item()

        return {'loss' : loss, 'correct' : correct}


    def validation_epoch_end(self, outputs):
        #end of an epoch
        #calculate the average loss of this epoch
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        accuracy = 100. * (sum([x['correct'] for x in outputs]) / float(len(self.cifar_val)))


        a = list(self.conv1.named_parameters())

        #robust_accuracy = self.adversarial_validation()
        robust_accuracy = 123
        print("Are lists equal?")
        print(list(self.conv1.named_parameters()) == a)


        #print("Total successful_attack no:{}, Total Attacks: {},  Attack Accuracy:{}".format(succesful_attack_no, len(self.cifar_val), robust_accuracy))
        logs = {'validation_loss': avg_loss, 'validation_accuracy': accuracy, 'robust_accuracy': robust_accuracy}
        return {'loss' : avg_loss, 'log' : logs}


    def adversarial_validation(self):
        val_dataloader = self.val_dataloader()
        self.eval()
        adv_model = fb.PyTorchModel(self, bounds=(0,1))
        successful_attack_sum = 0
        
        with torch.enable_grad():
            for batch_id, (data, label) in enumerate(val_dataloader):
                data, label = data.cuda(), label.cuda()
                _, _, success = self.cw(adv_model, data, label, epsilons=[0.01])
                successful_attack_no = torch.sum(success.long())
                #print("Successful attack:{} Attack count:{} Percentage:{}".format(successful_attack_no, len(label), float(successful_attack_no) /len(label)))
                successful_attack_sum += successful_attack_no

        self.zero_grad()

        print("Successful attack:{} Attack count:{} Percentage:{}".format(successful_attack_sum, len(self.cifar_val), float(successful_attack_sum) /len(self.cifar_val)))
        robust_accuracy = 100. * (1-(float(successful_attack_sum) /len(self.cifar_val))) 
        return robust_accuracy

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)


class ThresholdPruning(prune.BasePruningMethod):
    
    PRUNING_TYPE = 'unstructured'

    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()

        torch.gt()

        return mask


model = CIFARCLassifier()
model.prepare_data()
logger = TensorBoardLogger('tb_logs', name="adversarial_cifar10_model")
trainer = pl.Trainer(max_epochs=5, logger=logger, gpus=[0], fast_dev_run=False)
trainer.fit(model)

model.adversarial_validation()


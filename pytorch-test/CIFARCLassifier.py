import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
import foolbox as fb
import torchvision
from torchvision.datasets import CIFAR10


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
        self.cw = fb.attacks.L2FastGradientAttack()
        self.adv_model = fb.PyTorchModel(self, bounds=(0,1))

    def forward(self, x):
        x = self.pool(self.bn1(F.relu(self.conv1(x))))
        x = self.pool(self.bn2(F.relu(self.conv2(x))))
        x = self.pool(self.bn3(F.relu(self.conv3(x))))
        x = self.flatten(x)
        x = self.do1(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        return F.log_softmax(x, dim=1)

    def prepare_data(self):
        #download the data, augment and normalize them
        self.cifar_train = CIFAR10(root="data/", train=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(0.5, 0.5)]))

        self.cifar_val = CIFAR10(root="data/", train=False, transform=torchvision.transforms.Compose([ torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(0.5, 0.5)]))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.cifar_train, batch_size=128, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.cifar_val, batch_size=128, shuffle=False)

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
        self.zero_grad()
        self.adv_model = fb.PyTorchModel(self, bounds=(0,1))
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
        
        with torch.enable_grad():  
            _, _, success = self.cw(self.adv_model, data, label, epsilons=[0.001])
            successful_attack_no = torch.sum(success.long())
            
        return {'loss' : loss, 'correct' : correct, 'successful_attack_no' : successful_attack_no}
        
        #return {'loss' : loss, 'correct' : correct, 'adv_correct' : 1}

    def validation_epoch_end(self, outputs):
        #end of an epoch
        #calculate the average loss of this epoch
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        accuracy = 100. * (sum([x['correct'] for x in outputs]) / float(len(self.cifar_val)))

        #adv_avg_loss = torch.stack([x['adv_loss'] for x in outputs]).mean()
        robust_accuracy = 100. * (1-((sum([x['successful_attack_no'] for x in outputs]) / float(len(self.cifar_val)))))

        logs = {'validation_loss': avg_loss, 'validation_accuracy': accuracy, 'robust_accuracy': robust_accuracy}
        return {'loss' : avg_loss, 'log' : logs}


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)




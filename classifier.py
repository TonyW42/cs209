import torch 
from tqdm import tqdm 
import numpy as np
import pandas as pd 
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

from model import *

class baseline_classifier:
    def __init__(self, model, criterion, optimizer, args):
        self.model = model 
        self.args = args 
        self.optimizer = optimizer
        self.criterion = criterion 
    
    def _train_epoch(self, trainloader, devloader, testloader):
        tbar = tqdm(trainloader, dynamic_ncols=True)
        for data in tbar:
            logits = self.model(data)
            loss = self.criterion(logits, data["label"].to(self.args.device))
            loss.backward()
            self.optimizer.step()
            tbar.set_description("train_loss - {:.4f}".format(loss))
        dev_acc=  self._eval(devloader)
        test_acc=  self._eval(testloader)
        return dev_acc, test_acc
    
    def train(self, trainloader, devloader, testloader):
        test_accs, dev_accs = [], []
        for i in range(self.args.n_epochs):
            dev_acc, test_acc = self._train_epoch(trainloader, devloader, testloader)
            dev_accs.append(dev_acc)
            test_accs.append(test_acc)
            print("=" * 20)
            print(f"dev acc: {dev_acc}")
            print(f"test acc: {test_acc}")
            print("=" * 20)
        return dev_accs , test_accs

    def _eval(self, dataloader):
        with torch.no_grad():
            tbar = tqdm(dataloader, dynamic_ncols=True)
            preds = []
            labels = []
            for data in tbar:
                logits = self.model(data)
                pred = torch.argmax(logits, dim=-1)
                label = data["label"]
                pred.extend(pred.detach().cpu().tolist())
                labels.extend(label.cpu().tolist())
            acc = np.sum(np.equal(preds, labels)) / len(preds)
            return acc 

def run_baseline_experiment(args):
    trainloader, devloader, testloader = get_data(args)
    lm = AutoModel.from_pretrained(args.model_name)
    model = baseline_model(lm, args)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    criterion = nn.CrossEntropyLoss()
    classifier = baseline_classifier(model, criterion, optimizer, args)
    dev_accs , test_accs = classifier.train(trainloader, devloader, testloader)
    test_acc_chosen = test_accs[np.argmax(dev_accs)]
    return test_acc_chosen, dev_accs , test_accs



def get_data(args):
    df = pd.read_csv("data/fake reviews dataset.csv")
    label_2_id = {"CG":0, "OR": 1}
    train_df = df.iloc[:int(len(df) * args.train_size), :]
    dev_df = df.iloc[int(len(df) * args.train_size) : int(len(df) * (args.train_size + args.dev_size)), :]
    test_df = df.iloc[int(len(df) * (args.train_size + args.dev_size)):, :]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    traindataset = classification_dataset(train_df, tokenizer, label_2_id, args)
    devdataset = classification_dataset(dev_df, tokenizer, label_2_id, args)
    testdataset = classification_dataset(test_df, tokenizer, label_2_id, args)

    trainloader = DataLoader(traindataset, batch_size = args.bs, shuffle=True)
    devloader = DataLoader(devdataset, batch_size = args.bs, shuffle=True)
    testloader = DataLoader(testdataset, batch_size = args.bs, shuffle=True)

    return trainloader, devloader, testloader






            

        
import torch 
import torch.nn as nn 
from torch.utils.data import Dataset

class baseline_model(nn.Module):
    def __init__(self, lm, args):
        self.lm = lm
        self.args = args
        feature_size =  args.emb_size
        if args.other_features : feature_size + args.feature_size
        self.head = nn.Linear(feature_size, args.num_labels)
    
    def forward(self, data):
        encoded = self.lm(
            input_ids = data["input_ids"].to(self.args.device),
            attention_mask = data["attention_mask"].to(self.args.device)
        )
        emb = encoded["pooler_output"]
        if self.args.other_features:
            emb = torch.cat(emb, data["other_features"], dim = -1)
        logits = self.head(emb)
        return logits

class classification_dataset(Dataset):
    def __init__(self, df, tokenizer, label_2_id, args):
        self.df = df
        self.args = args
        self.tokenizer = tokenizer
        self.label_2_id = label_2_id
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        tokenized = self.tokenizer(self.df["text_"][idx])
        tokenized["label"] = self.label_2_id[self.df["label"][idx]]
        if self.args.other_features:
            pass ## TODO here 
        for k, v in tokenized.items():
            tokenized[k] = torch.tensor(v)
        return tokenized 

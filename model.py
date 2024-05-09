import torch 
import torch.nn as nn 
from torch.utils.data import Dataset

class baseline_model(nn.Module):
    def __init__(self, lm, args):
        super(baseline_model, self).__init__()
        self.lm = lm
        self.args = args
        feature_size =  args.emb_size
        if args.other_features : feature_size += args.feature_size
        self.head = nn.Linear(feature_size, args.num_labels)
    
    def forward(self, data):
        encoded = self.lm(
            input_ids = data["input_ids"].to(self.args.device),
            attention_mask = data["attention_mask"].to(self.args.device)
        )
        emb = encoded["pooler_output"]
        if self.args.other_features:
            emb = torch.cat((emb, data["other_features"].type(torch.float).to(self.args.device)), dim = -1)
        logits = self.head(emb)
        return logits

class classification_dataset(Dataset):
    def __init__(self, df, tokenizer, label_2_id, args):
        self.df = df
        self.args = args
        self.tokenizer = tokenizer
        self.label_2_id = label_2_id
        self.categories = [
            'Books_5', 'Clothing_Shoes_and_Jewelry_5', 
            'Electronics_5','Home_and_Kitchen_5', 'Kindle_Store_5', 
            'Movies_and_TV_5','Pet_Supplies_5', 'Sports_and_Outdoors_5',
            'Tools_and_Home_Improvement_5', 'Toys_and_Games_5'
            ]
        self.categroy2id = {self.categories[i]: i for i in range(len(self.categories))}
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        tokenized = self.tokenizer(self.df["text_"][idx], 
                                   padding = "max_length",
                                   truncation = True )
        tokenized["label"] = self.label_2_id[self.df["label"][idx]]
        if self.args.other_features:
            category = self.df.category[idx]
            category_id = self.categroy2id[category]
            category_binary = [1 if i == category_id else 0 for i in range(len(self.categories))]
            category_binary.append(self.df.rating[idx]) ## add ratings
            tokenized["other_features"] = category_binary
        for k, v in tokenized.items():
            tokenized[k] = torch.tensor(v)
        return tokenized 

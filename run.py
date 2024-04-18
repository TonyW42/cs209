import argparse
import torch 
from classifier import *


parser = argparse.ArgumentParser()
parser.add_argument("--emb_size", type=int, default=768)
parser.add_argument("--bs", type=int, default=8)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--other_features", action='store_true')
parser.add_argument("--num_labels", type=int, default=2)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--n_epochs", type=int, default=25)  
parser.add_argument("--train_size", type=float, default=0.7)  
parser.add_argument("--dev_size", type=float, default=0.1)  
parser.add_argument("--model_name", type=str, default="roberta-base")  
parser.add_argument("--feature_size", type=int, default=11)  


args = parser.parse_args()

if torch.cuda.is_available():
    args.device = "cuda"
# elif torch.backends.mps.is_available():
#     args.device = "mps"

if __name__ == "__main__":
    print(f"Add other features:{args.other_features}")
    test_acc_chosen, dev_accs , test_accs = run_baseline_experiment(args)
    print(f"Test acc is: {test_acc_chosen}")
    print(f"All dev acc {dev_accs}")
    print(f"All test acc {test_accs}")


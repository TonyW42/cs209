# cs209
The course project page for CS209 project.  

Please see our video presentation at: https://drive.google.com/file/d/1F0B8ZB-zsXZskry-R3YBLzO7WfMavhdx/view?usp=sharing

Some explanation to the flags:
- `--model_name`: used to identify a model on Huggingface.
- `--other_feature`: raise this flag (no need to add anything after) to indicate that we are using other structured features 
- `--emb_size`: the embedding size of the model. typically set to 769 for "base" models and 1024 for "large" models (please check model specifications). Default to 768.

Some experiments that we should run on different LMs:
- `python run.py --model_name roberta-base `  (uses RoBERTa-base)
- `python run.py --model_name roberta-large --emb_size 1024`  (uses RoBERTa-large)
- `python run.py --model_name bert-base-cased` (base cased bert)
- `python run.py --model_name bert-base-uncased` (base uncased bert)
- `python run.py --model_name distilbert-base-uncased` (base uncased, smaller bert)
- `python run.py --model_name google/canine-s` (character-level language model)

Run experiment that uses other structured features (combine with different LM):
- `python run.py --other_feature`  (uses RoBERTa-base and structured features)

Note that you should adjust some hyperparameters (e.g., `--bs`, `lr`, but not others like `--num_labels`). Please run with GPU (V100 preferably, any base model should work with T-4 though). 




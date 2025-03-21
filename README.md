# Code for HierGR

## Offline Training on MSMARCO

### Dataset

MSMARCO dataset is obtained from https://github.com/liyongqi67/MINDER.

### RQ-VAE
code for offline hierarchical RQ-VAE training.

```bash
bash train_tokenizer.sh
bash tokenize.sh
```
 

 ### GR_train

 ```bash
 bash train.sh
 bash test.sh
 ```


 ## Large-scale Deployment

 ### online_deployment_version/RQ-VAE

 This is a version that utilize pytorch DDP to train hierarchical RQ-VAE. The embeddings of all items can be divide into several npy files, named "semantic_emb_*.npy".

 ### Online GR Model Training

 We directly use LLama-Factory to train our GR model based on Qwen2.5-1.5B-Instruct. LLama-Factory can be accessed from https://github.com/hiyouga/LLaMA-Factory. 
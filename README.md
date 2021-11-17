## Enhance Disentanglement of Popularity Bias for Recommendation with Contrastive Learning

### Dataset
We provide three datasets: Ciao, LastFM, and Movielens-10M

### Example to run the codes
1. Environment: I have tested this code with python3.8 Pytorch=1.7.1 CUDA=11.0
2. Run DECL
	- first start the visdom server:
```
visdom -port 33336
```
	- then run model on each dataset by running:
```
python app.py --flagfile ./config/ciao_decl.cfg
```
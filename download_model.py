from huggingface_hub import hf_hub_download
import os

token = os.environ.get('HF_TOKEN')

hf_hub_download(repo_id='ElBOOH55/simclr-stl10', filename='simclr_best.pt', local_dir='checkpoints', token=token)
hf_hub_download(repo_id='ElBOOH55/simclr-stl10', filename='faiss.index', local_dir='index', token=token)
hf_hub_download(repo_id='ElBOOH55/simclr-stl10', filename='labels.npy', local_dir='index', token=token)

print('Model downloaded successfully')
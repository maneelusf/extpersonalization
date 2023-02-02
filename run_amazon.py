import os
import subprocess
import gdown
from recbole.quick_start import run_recbole
url = "https://drive.google.com/uc?id=1De099aEeHZ-8rKcscElYnO1Mo3pCZ-oO"
gdown.download(url, quiet=True, use_cookies=False)
os.system('unzip Amazon_Toys_and_Games.zip')
os.system('rm Amazon_Toys_and_Games.zip')
parameter_dict = {
   'train_neg_sample_args': None,
    'load_col':{'inter': ['user_id', 'item_id', 'rating', 'timestamp'],
 'item': ['item_id', 'categories', 'brand']},
    'selected_features':['categories','brand'],
   'data_path':'',
    'epochs':600,'train_batch_size':2048
}
run_recbole(model='GRU4RecF', dataset='Amazon_Toys_and_Games', config_dict=parameter_dict)

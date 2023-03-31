import os
import subprocess
import gdown
from recbole.quick_start import run_recbole
url = "https://drive.google.com/file/d/1o79AJLqrDGGixzfNayczunuoSNQ7hCSm"
gdown.download(url, quiet=True, use_cookies=False)
os.system('unzip yelp2022.zip')
os.system('rm yelp2022.zip')
parameter_dict = {
   'train_neg_sample_args': None,
    'load_col':{'inter': ['user_id', 'item_id', 'rating', 'timestamp'],
 'item': ['item_id', 'categories', 'brand']},
    'selected_features':['categories','brand'],
   'data_path':'',
    'epochs':600,'train_batch_size':2048
}
run_recbole(model='GRU4RecF', dataset='yelp2022', config_dict=parameter_dict)

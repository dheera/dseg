import os, tempfile, urllib, urllib.request
#@title Download pre-trained checkpoint.

# MODEL_DIR The directory of the exported ViP-DeepLab model.
MODEL_URL = 'https://storage.googleapis.com/gresearch/tf-deeplab/saved_model/resnet50_beta_os32_vip_deeplab_cityscapes_dvps_train_saved_model.tar.gz' #@param {type:"string"}

model_name = 'resnet50_beta_os32_vip_deeplab_cityscapes_dvps_train_saved_model'
model_dir = tempfile.mkdtemp()
download_path = os.path.join(model_dir, 'model.tar.gz')
urllib.request.urlretrieve(MODEL_URL, download_path)
os.system(f'tar -xzvf {download_path} -C {model_dir}')
model_path = os.path.join(model_dir, model_name, 'exports')

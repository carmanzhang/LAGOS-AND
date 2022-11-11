import argparse
import os

gpu_id = 0
device = "cuda:%d" % gpu_id

parser = argparse.ArgumentParser()
parser.add_argument("--which_model", type=int, default=5)
cli_args = parser.parse_args()
best_hac_clustering_parameters = [0.45, 0.25, 0.2, 0.2, 0.25, 0.2]
# best_hac_clustering_parameters = [None, None, None, None, None, None]
tuned_best_cluster_setting = best_hac_clustering_parameters[cli_args.which_model]

# resource config
latex_doc_base_dir = '/home/zhangli/ssd-1t/repo/manuscripts/ongoning-works/and-dataset/src/'
src_base_path = os.path.dirname(os.path.abspath(__file__))
cached_dir = os.path.join(src_base_path, 'cached')

pretrained_model_path = proj_base_path = os.path.abspath('/home/zhangli/pre-trained-models/')
glove6b_path = os.path.join(pretrained_model_path, 'glove.6B/')
glove840b300d_path = os.path.join(pretrained_model_path, 'glove.840B/')
fasttextcrawl300d2m_path = os.path.join(pretrained_model_path, 'fastText/crawl-300d-2M.vec')
infersent_based_path = os.path.join(pretrained_model_path, 'infersent')

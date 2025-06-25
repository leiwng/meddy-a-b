import os

# PATH
# 工程根目录
base_path = str(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 静态文件路径
static_dir = os.path.join(base_path, 'static')

# 文件路径
file_dir = os.path.join(static_dir, 'file')

images_dir = os.path.join(static_dir, 'images')

# pdf文件路径
pdf_dir = os.path.join(file_dir, 'pdf')

# pdf文件中图像
pdf_image_dir = os.path.join(file_dir, 'pdf_image')

# 向量文件路径
vector_dir = os.path.join(file_dir, 'vector')

# 网络模型目录
nn_model_dir = os.path.join(base_path, 'source_code', 'agent', 'models')

# vit模型目录
vit_model_dir = os.path.join(nn_model_dir, 'vit-base-patch16-224')

# vit模型参数路径
vit_model_params_path = os.path.join(nn_model_dir, 'vit_best_model.pth')

# vit模型输出类别映射文件路径
vit_model_class_map_path = os.path.join(nn_model_dir, 'class_names.json')
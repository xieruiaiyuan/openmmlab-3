from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset

@DATASETS.register_module()  # 这一步将我们的网络register到整个程序中
class ADataset(BaseSegDataset): # 名字可以自己起，但是注意大小写
    classes = ('sky', 'tree', 'road', 'grass', 'water', 'bldg', 'mntn', 'fg obj') # 分别对应于[0,1,...]的类别名称，注意，如果忽略0值，则对应于[1,2,...]的类别名称
    palette = [[128, 128, 128], [129, 127, 38], [120, 69, 125], [53, 125, 34], 
           [0, 11, 123], [118, 20, 12], [122, 81, 25], [241, 134, 51]]# 在预测时，给预测结果上色，仅用于可视化
    METAINFO = dict(classes=classes, palette=palette)
    def __init__(self, **kwargs):
        super(ADataset, self).__init__(
            img_suffix='.jpg', # 默认图像的后缀为‘.jpg’,根据数据修改
            seg_map_suffix='.png', # 默认标注的后缀为‘.png’,根据数据修改
            **kwargs)

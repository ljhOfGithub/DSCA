import sys
import os
import os.path as osp
import shutil
import h5py
import numpy as np
from tqdm import tqdm


# LEVEL=1#再高分辨率
# LEVEL=2 #先低分辨率 数字越大，分辨率越低，则坐标越少
"""
path_patchi: level = L,   size = 256
path_patcho: level = L-1, size = 256
path_patchi * patch_scale -> path_patcho
# LEVEL=1#再高分辨率
# LEVEL=2 #先低分辨率 数字越大，分辨率越低
"""
def save_hdf5(output_path, asset_dict, attr_dict=None, mode='a'):#将数据保存到 HDF5 文件中。
    file = h5py.File(output_path, mode)
    for key, val in asset_dict.items():
        data_shape = val.shape
        if key not in file:
            data_type = val.dtype
            chunk_shape = (1, ) + data_shape[1:]
            maxshape = (None, ) + data_shape[1:]
            dset = file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
            dset[:] = val
            if attr_dict is not None:
                if key in attr_dict.keys():
                    for attr_key, attr_val in attr_dict[key].items():
                        dset.attrs[attr_key] = attr_val
        else:
            dset = file[key]
            dset.resize(len(dset) + data_shape[0], axis=0)
            dset[-data_shape[0]:] = val
    import pdb
    # pdb.set_trace()
    file.close()
    return output_path

def get_scaled_matrix(width, height, scale=4):#用于生成缩放后的坐标矩阵。
    mat = np.zeros((scale, scale, 2))
    for j in range(scale):
        for i in range(scale):
            mat[j][i] = np.array([i * width, j * height])
    mat = np.reshape(mat, (-1, 2))
    return mat

def get_scaled_attrs(origin_attrs, scale=4):#用于生成缩放后的属性字典。
    attrs = {
        'downsample': origin_attrs['downsample'] / scale,
        'downsampled_level_dim': origin_attrs['downsampled_level_dim'] * scale,
        'level_dim': origin_attrs['level_dim'] * scale,
        'name': origin_attrs['name'],
        'patch_level': origin_attrs['patch_level'] - 1,
        'patch_size': origin_attrs['patch_size'],
    }
    import pdb
    # pdb.set_trace()
    return attrs

def coords_x5_to_x20(path_patchi, path_patcho, patch_scale=4):
    #用于将坐标数据从 level = L（原始文件） 缩放到 level = L-1，并保存到 HDF5 文件中。
    scaled_coords = np.zeros((1,2), dtype=np.int32)
    scaled_attrs  = None
    import pdb
    # pdb.set_trace()
    with h5py.File(path_patchi, 'r') as hf:
        data_coords = hf['coords']
        scaled_attrs = get_scaled_attrs(data_coords.attrs, patch_scale)

        psize = data_coords.attrs['patch_size']
        scaled_mat = get_scaled_matrix(psize, psize, patch_scale)
        coords = data_coords[:]
        for coord in coords:
            cur_coords = scaled_mat + coord
            scaled_coords = np.concatenate((scaled_coords, cur_coords), axis=0)

    scaled_coords = scaled_coords[1:] # ignore the first row
    scaled_attrs['save_path'] = osp.dirname(path_patcho)
    import pdb
    # pdb.set_trace()
    save_hdf5(path_patcho, {'coords': scaled_coords}, {'coords': scaled_attrs}, mode='w')
#用于处理指定目录下的所有 h5 文件。它会遍历指定目录下的所有文件，如果文件名不以 .h5 结尾，则会跳过处理。
#对于以 .h5 结尾的文件，它将调用 coords_x5_to_x20 函数对坐标数据进行缩放，并将缩放后的数据保存到指定目录中。
#同时，它会将原始处理列表文件 process_list_autogen.csv 复制到保存目录中。
def process_coords(dir_read, dir_save):
    if not osp.exists(dir_save):
        os.makedirs(dir_save)
    dir_read = dir_read + '/patches'
    dir_save = dir_save + '/patches'
    files = os.listdir(dir_read)
    for fname in tqdm(files):
        if fname[-2:] != 'h5':
            print('invalid file {}, skipped'.format(fname))
            continue

        path_read = osp.join(dir_read, fname)
        path_save = osp.join(dir_save, fname)
        coords_x5_to_x20(path_read, path_save)

# python3 big_to_small_patching.py READ_PATCH_DIR SAVE_PATCH_DIR 
if __name__ == '__main__':
    # READ_PATCH_DIR = sys.argv[1] # full read path to the patch coordinates at level = 2.分辨率*5
    # SAVE_PATCH_DIR = sys.argv[2] # full save path to the patch coordinates at level = 1.分辨率*20
    READ_PATCH_DIR = '/home/jupyter-ljh/data/mntdata/data0/LI_jihao/DSCA-BRCA/copy/tiles-l2-s256' # full read path to the patch coordinates at level = 2.分辨率*5
    SAVE_PATCH_DIR = '/home/jupyter-ljh/data/mntdata/data0/LI_jihao/DSCA-BRCA/copy/tiles-l1-s256' # full save path to the patch coordinates at level = 1.分辨率*20 注意必须先在tiles-l1-s256下面手动新建文件夹patches
    process_coords(READ_PATCH_DIR, SAVE_PATCH_DIR)
    # at the same time, copy the processing record file to SAVE_PATCH_DIR
    shutil.copy(osp.join(READ_PATCH_DIR, 'process_list_autogen.csv'), SAVE_PATCH_DIR)
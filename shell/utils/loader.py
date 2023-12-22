"""
Copyright (c) 2023 Samsung Electronics Co., Ltd.

Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc/4.0/
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE file.
"""
import os
import glob
import json
import PIL
import meshio

import numpy as np
import pyvista as pv
from pathlib import Path
from matplotlib import pyplot as plt

class ObjectLoader:
    def __init__(self, verbose=False):
        self.loader_registry = {}
        self.saver_registry = {}
        self.verbose = verbose

    def register_ext_loader(self, ext):
        def register(f):
            self.loader_registry[ext] = f
            return f
        return register

    def register_cls_saver(self, cls):
        def register(f):
            self.saver_registry[cls] = f
            return f
        return register

    def register_cls(self, cls):
        assert hasattr(cls, 'save')
        assert hasattr(cls, 'load')
        assert hasattr(cls, 'get_file_extension')

        self.saver_registry[cls] = cls.save

        def cls_load(*kargs, **kwargs):
            return cls.load(*kargs, **kwargs)

        self.loader_registry[cls.get_file_extension()] = cls_load

        return cls

    def warn(self, msg):
        if self.verbose:
            print(msg)

    def load(self, path):
        fname = os.path.basename(path)
        if '.' not in fname:
            match_list = glob.glob(path + '.*')
            if len(match_list) > 1:
                raise Exception(f"Mulitple matches for {path}: {match_list}")
            elif len(match_list) == 0:
                self.warn(f"No extensions match {path}")
                return None
            else:
                path = match_list[0]

        self.warn(f"loading {path}")
        ext = os.path.basename(path).split('.')[-1]
        if ext not in self.loader_registry:
            raise Exception(f"No loader for object extension '{ext}'")
        else:
            if os.path.exists(os.path.abspath(path)):
                return self.loader_registry[ext](os.path.abspath(path), self)
            else:
                return None

    def save(self, obj, path):
        if obj is None:
            return
        else:
            obj_type = type(obj)
            if obj_type not in self.saver_registry:
                raise Exception(f"No saver for object type '{obj_type}'")
            else:
                self.warn(f"saving to {path}")
                self.saver_registry[obj_type](obj, path, self)

    def load_bundle(self, path, keys_to_load):
        result = {}

        for k in keys_to_load:
            result[k] = self.load(os.path.join(path, k))

        return result

    def save_bundle(self, path, bundle):
        for k, v in bundle.items():
            self.save(v, os.path.join(path, k))




Loader = ObjectLoader()


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except:
        return False


@Loader.register_cls_saver(dict)
def save_dict(d, path, loader):
    if is_jsonable(d):
        full_path = path + '.json'
        Path(os.path.dirname(full_path)).mkdir(parents=True, exist_ok=True)
        json.dump(d, open(full_path, 'w'))
    else:
        full_path = path + '.dict'
        Path(full_path).mkdir(parents=True, exist_ok=True)

        for k, v in d.items():
            child_path = os.path.join(full_path, str(k))
            loader.save(v, child_path)


@Loader.register_ext_loader('dict')
def load_dict(path, loader):
    result = {}
    for child_path in Path(path).glob('*'):
        child_fname = os.path.basename(child_path)
        assert len(child_fname.split('.')) == 2
        name = child_fname.split('.')[0]
        result[name] = loader.load(child_path)

    return result


@Loader.register_cls_saver(list)
def save_list(l, path, loader):
    full_path = path + '.list'
    Path(full_path).mkdir(parents=True, exist_ok=True)

    for i, v in enumerate(l):
        child_path = os.path.join(full_path, f"item_{i}")
        loader.save(v, child_path)


@Loader.register_ext_loader('list')
def load_list(path, loader):
    assert '.list' in path
    result = []
    result_dict = {}

    for child_path in Path(path).glob('*'):
        child_fname = os.path.basename(child_path)
        child_name = child_fname.split('.')[0]

        if not child_name.startswith('item_'):
            raise Exception(f"Unexpected list item name: '{child_name}'")

        child_id = int(child_name[len('item_'):])
        result_dict[child_id] = loader.load(child_path)

    result = [v for k, v in sorted(result_dict.items())]

    return result


@Loader.register_cls_saver(pv.core.pyvista_ndarray)
def save_np(a, path, loader):
    full_path = path + '.npy'
    Path(os.path.dirname(full_path)).mkdir(parents=True, exist_ok=True)
    np.save(open(full_path, 'wb'), a)



@Loader.register_cls_saver(np.ndarray)
def save_np(a, path, loader):
    full_path = path + '.npy'
    Path(os.path.dirname(full_path)).mkdir(parents=True, exist_ok=True)
    np.save(open(full_path, 'wb'), a)


@Loader.register_ext_loader('npy')
def load_np(path, loader):
    assert '.npy' in path
    result = np.load(open(path, 'rb'))

    return result


@Loader.register_ext_loader('npz')
def load_np(path, loader):
    assert '.npz' in path
    result = np.load(open(path, 'rb'))

    return result


@Loader.register_ext_loader('jpg')
def load_np(path, loader):
    assert '.jpg' in path
    result = np.array(PIL.Image.open(path).convert('RGB'))
    return result


@Loader.register_cls_saver(pv.PolyData)
def save_pv(p, path, loader):
    full_path = path + '.obj'
    Path(os.path.dirname(full_path)).mkdir(parents=True, exist_ok=True)
    try:
        pv.save_meshio(full_path, p)
    except meshio._exceptions.WriteError:
        ply_full_path = path + '.ply'
        p.save(ply_full_path)


import open3d as o3d
@Loader.register_cls_saver(o3d.open3d.geometry.TriangleMesh)
def save_o3d_mesh(p, path, loader):
    full_path = path + '.obj'
    Path(os.path.dirname(full_path)).mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(full_path, p)


@Loader.register_cls_saver(o3d.cpu.pybind.geometry.TriangleMesh)
def save_o3d_mesh_pybind(p, path, loader):
    full_path = path + '.obj'
    Path(os.path.dirname(full_path)).mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(full_path, p)


@Loader.register_cls_saver(o3d.open3d.geometry.PointCloud)
def save_o3d_mesh(p, path, loader):
    full_path = path + '.ply'
    Path(os.path.dirname(full_path)).mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(full_path, p)


@Loader.register_ext_loader('vtk')
def load_vtk(path, loader):
    assert '.vtk' in path
    result = pv.read(path)
    return result


@Loader.register_ext_loader('obj')
def load_vtk(path, loader):
    assert '.obj' in path
    result = pv.read(path)
    return result


@Loader.register_ext_loader('off')
def load_off(path, loader):
    assert '.off' in path
    result = pv.read(path)
    return result


@Loader.register_ext_loader('ply')
def load_ply(path, loader):
    assert '.ply' in path
    result = pv.read(path)
    return result


@Loader.register_ext_loader('stl')
def load_stl(path, loader):
    assert '.stl' in path
    result = pv.read(path)
    return result

import pickle
@Loader.register_ext_loader('pkl')
def load_pkl(path, loader):
    assert '.pkl' in path
    with open(path, 'rb') as f:
        result = pickle.load(f)
    return result

@Loader.register_cls_saver(pv.Texture)
def save_png(tex, path, loader):
    full_path = path + '.png'
    Path(os.path.dirname(full_path)).mkdir(parents=True, exist_ok=True)
    plt.imsave(full_path, tex.to_array())


@Loader.register_ext_loader('png')
def load_png(path, loader):
    assert '.png' in path
    result = pv.read_texture(path)
    return result


@Loader.register_ext_loader('json')
def load_json(path, loader):
    assert '.json' in path
    result = json.load(open(path))
    return result


def load_bundle(path, keys_to_load):
    return Loader.load_bundle(path, keys_to_load)


def save_bundle(path, bundle):
    for k, v in bundle.items():
        Loader.save(v, os.path.join(path, k))


def save(*kargs, **kwargs):
    Loader.save(*kargs, **kwargs)


def load(*kargs, **kwargs):
    return Loader.load(*kargs, **kwargs)

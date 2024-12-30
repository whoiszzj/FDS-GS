from plyfile import PlyData, PlyElement
import numpy as np
import argparse
import os


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def load_ply(ply_path):
    plydata = PlyData.read(ply_path)
    xyz = np.stack(
        (
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"])
        ), axis=1
    )
    normal = np.stack(
        (
            np.asarray(plydata.elements[0]["nx"]),
            np.asarray(plydata.elements[0]["ny"]),
            np.asarray(plydata.elements[0]["nz"])
        ),
        axis=1
    )

    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names) == 3 * (3 + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (3 + 1) ** 2 - 1))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    R = np.zeros((xyz.shape[0], 1))
    R[:, 0] = np.asarray(plydata.elements[0]["R"])

    return {
        "xyz": xyz,
        "normal": normal,
        "opacity": opacities,
        "f_dc": features_dc,
        "f_rest": features_extra,
        "scales": scales,
        "rots": rots,
        "R": R
    }


def convert(gaussian_dict):
    rel_scale = sigmoid(gaussian_dict["scales"])
    abs_scale = np.exp(gaussian_dict["R"])
    scale = rel_scale * abs_scale
    scale_raw = np.log(scale)
    return {
        "xyz": gaussian_dict["xyz"],
        "normal": gaussian_dict["normal"],
        "opacity": gaussian_dict["opacity"],
        "f_dc": gaussian_dict["f_dc"],
        "f_rest": gaussian_dict["f_rest"],
        "scales": scale_raw,
        "rots": gaussian_dict["rots"],
    }


def construct_list_of_attributes():
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels except the 3 DC
    for i in range(3):
        l.append('f_dc_{}'.format(i))
    for i in range(45):
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(3):
        l.append('scale_{}'.format(i))
    for i in range(4):
        l.append('rot_{}'.format(i))
    return l


def export_gs_ply(gaussian_dict, gs_path):
    xyz = gaussian_dict["xyz"]
    normal = gaussian_dict["normal"]
    opacity = gaussian_dict["opacity"]
    f_dc = gaussian_dict["f_dc"].reshape((gaussian_dict["f_dc"].shape[0], 3))
    f_rest = gaussian_dict["f_rest"].reshape((gaussian_dict["f_rest"].shape[0], 45))  # [N, 45]
    scales = gaussian_dict["scales"]
    rots = gaussian_dict["rots"]

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normal, f_dc, f_rest, opacity, scales, rots), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(gs_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--ply_path", type=str, required=True)
    parser.add_argument("-o", "--output_path", type=str, default=None, required=False)
    args = parser.parse_args()

    if not os.path.exists(args.ply_path):
        raise FileNotFoundError("File does not exist")

    gaussian_dict = load_ply(args.ply_path)
    gs_dict = convert(gaussian_dict)
    export_path = args.output_path if args.output_path is not None else args.ply_path.replace(".ply", "_gs.ply")
    export_gs_ply(gs_dict, export_path)
    print("Conversion completed, saved to {}".format(export_path))

"""Neuroglancer IP visualization as an annotation layer."""

import json
import os
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from ng_link import NgState, link_utils
from ng_link.parsers import XmlParser
import s3fs
import zarr


def read_json(json_path: str) -> dict:  # pragma: no cover
    """Read a json file and return a dict."""
    with open(json_path) as f:
        return json.load(f)


def get_tile_positions(dataset_path: str):  # pragma: no cover
    """Get the tile positions from the zattrs files."""
    tile_positions = {}
    for tile_path in Path(dataset_path).iterdir():
        if tile_path.name == ".zgroup":
            continue

        zattrs_file = tile_path / ".zattrs"
        zattrs_json = read_json(zattrs_file)

        scale = zattrs_json["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"]
        translation = zattrs_json["multiscales"][0]["datasets"][0]["coordinateTransformations"][1][
            "translation"
        ]

        scale = np.array(scale[2:][::-1])
        translation = np.array(translation[2:][::-1])
        translation /= scale
        translation = np.round(translation, 4)

        tile_positions[tile_path.name] = translation

    return tile_positions


def read_in_n5_ips(tile_setupId, transform_3x4=None):  # pragma: no cover
    """Read in intereset point coordinates from the n5 binary files.
    Interest points are not sorted.
    """
    n5s = zarr.n5.N5FSStore("../results/interestpoints.n5/")
    zg = zarr.open(store=n5s, mode="r")

    id = zg[f"tpId_0_viewSetupId_{tile_setupId}/beads/interestpoints/id"]
    loc = zg[f"tpId_0_viewSetupId_{tile_setupId}/beads/interestpoints/loc"]

    # Now return the first 10 for testing in ng annotation dict
    if id.ndim < 2 or loc.ndim < 2 or id.shape[0] != loc.shape[0]:
        raise ValueError(f"n5 ip arrays shape error id.shape={id.shape} loc.shape={loc.shape}")

    P = []
    N = min(loc.shape[0], 1000)
    # Read in the interest points from disk in one go
    loc = np.array(loc[:N])
    if transform_3x4 is not None:
        vec = np.ones(4, dtype=float)
        for i in range(N):
            vec[:3] = loc[i]
            v2 = np.matmul(transform_3x4, vec)
            P.append({'x': v2[2], 'y': v2[1], 'z': v2[0]})
    else:
        for i in range(10):
            P.append({'x': loc[i][2], 'y': loc[i][1], 'z': loc[i][0]})
    return P


def get_tile_positions_s3(dataset_uri: str):  # pragma: no cover
    """Get the tile positions from the zattrs files.
    Parameters
    ----------

    dataset_uri: The S3 prefix to the dataset (e.g. "bucket_name/dataset_name/SPIM.ome.zarr")
    """
    tile_positions = {}
    s3 = s3fs.S3FileSystem()
    url = urlparse(dataset_uri)
    dataset_path = url.netloc + "/" + url.path.strip("/")
    for tile_path in s3.ls(dataset_path):
        if not s3.isdir(tile_path):
            continue

        tile_path = Path(tile_path)
        zattrs_file = str(tile_path / ".zattrs")
        with s3.open(zattrs_file, "r") as f:
            zattrs_json = json.load(f)

        scale = zattrs_json["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"]
        translation = zattrs_json["multiscales"][0]["datasets"][0]["coordinateTransformations"][1][
            "translation"
        ]

        scale = np.array(scale[2:][::-1])
        translation = np.array(translation[2:][::-1])
        translation /= scale
        translation = np.round(translation, 4)

        tile_positions[tile_path.name] = translation

    return tile_positions

# Copy the relevant bits from create_ng_link.py and create ng link with
# annotation layer


def create_ng_link_with_annotation(
    dataset_uri: str,
    alignment_output_uri: str,
    xml_path: str = "../results/bigstitcher.xml",
    output_json: str = "../results/ng/process_output.json",
) -> str:  # pragma: no cover
    """ "Create a Neuroglancer json file and link file for the given alignment solution.

    The link is appended to the file ../results/ng/ng_link.txt.

    Parameters
    ----------
    dataset_uri: str
        URI to the tiled dataset
    xml_path: str
        Path to the BigStitcher output xml file with the alignment result.
    output_json: str
        Path and file name to the Neuroglancer json file will be saved in the capsule.
    """
    dataset_uri = dataset_uri.rstrip("/")
    # Never put trailing slashes
    # output_uri = 's3://aind-open-data/exaSPIM_659146_2023-11-10_14-02-06/SPIM.ome.zarr'
    # xml_path = '/root/capsule/data/2023-11-27_s18_th03_l150k_659146/bigstitcher_2023-11-27.xml'
    # dataset_path = '/root/capsule/data/exaSPIM_659146_2023-11-10_14-02-06/SPIM.ome.zarr'
    # alignment_run = '2023-11-27'  # Will be attached to upload name: process_output_{alignment_run}.json
    # output_json_path = '/results'

    # XML info
    vox_sizes: tuple[float, float, float] = XmlParser.extract_tile_vox_size(xml_path)
    tile_paths: dict[int, str] = XmlParser.extract_tile_paths(xml_path)
    tile_transforms: dict[int, list[dict]] = XmlParser.extract_tile_transforms(xml_path)

    # Color info
    channel: int = link_utils.extract_channel_from_tile_path(tile_paths[0])
    hex_val: int = link_utils.wavelength_to_hex(channel)
    hex_str = f"#{str(hex(hex_val))[2:]}"

    # Zattrs info
    zattrs_positions = get_tile_positions_s3(dataset_uri)

    # The points do not need the zattrs correction
    point_net_transforms: dict[int, np.ndarray] = link_utils.calculate_net_transforms(tile_transforms)

    # Update Translation -- undo zattrs transform
    for tile_id, tf in tile_transforms.items():
        t_path = tile_paths[tile_id]
        zattrs_offset = zattrs_positions[t_path]

        nums = [float(val) for val in tf[0]["affine"].split(" ")]
        nums[3] = nums[3] - zattrs_offset[0]
        nums[7] = nums[7] - zattrs_offset[1]
        nums[11] = nums[11] - zattrs_offset[2]

        tf[0]["affine"] = "".join(f"{n} " for n in nums)
        tf[0]["affine"] = tf[0]["affine"].strip()

        tile_transforms[tile_id] = tf

    net_transforms: dict[int, np.ndarray] = link_utils.calculate_net_transforms(tile_transforms)

    # Generate input config
    layers = []  # Neuroglancer Tabs
    input_config = {
        "dimensions": {
            "x": {"voxel_size": vox_sizes[0], "unit": "microns"},
            "y": {"voxel_size": vox_sizes[1], "unit": "microns"},
            "z": {"voxel_size": vox_sizes[2], "unit": "microns"},
            "c'": {"voxel_size": 1, "unit": ""},
            "t": {"voxel_size": 0.001, "unit": "seconds"},
        },
        "layers": layers,
        "showScaleBar": False,
        "showAxisLines": False,
    }

    sources = []  # Tiles within tabs
    layers.append(
        {
            "type": "image",  # Optional
            "source": sources,
            "channel": 0,  # Optional
            "shaderControls": {"normalized": {"range": [0, 200]}},  # Optional  # Exaspim has low HDR
            "shader": {
                "color": hex_str,
                "emitter": "RGB",
                "vec": "vec3",
            },
            "visible": True,  # Optional
            "opacity": 1,
            "name": f"CH_{channel}",
            "blend": "default",
        }
    )

    for tile_id, _ in enumerate(net_transforms):
        net_tf = net_transforms[tile_id]
        t_path = tile_paths[tile_id]

        url = f"{dataset_uri}/{t_path}"
        final_transform = link_utils.convert_matrix_3x4_to_5x6(net_tf)

        sources.append({"url": url, "transform_matrix": final_transform.tolist()})
        # Mark the corner of each tile
        # ips = []
        # vec = np.zeros(4, dtype=float)
        # vec[3] = 1
        # vec = np.matmul(point_net_transforms[tile_id], vec)
        # ips.append({'x': vec[2],'y': vec[1], 'z': vec[0]})
        ips = read_in_n5_ips(tile_id, point_net_transforms[tile_id])
        layers.append({
                "type": "annotation",
                "source": f"precomputed://ng/tile_{tile_id}/precomputed",
                "tool": "annotatePoint",
                "name": f"IPs_{tile_id}",
                "annotations": ips,
                # Pass None or delete limits if
                # you want to include all the points
                # "limits": [100, 200],  # None # erase line
            })

    # ng_dir, json_name = os.path.split(output_json)
    ng_dir = "ng"
    json_name = "process_output.json"
    # url = urlparse(dataset_uri)
    url = urlparse(alignment_output_uri)
    # if url.scheme != "s3":
    #     raise ValueError(f"Dataset URI must be an S3 URI, got {dataset_uri}")
    # bucket_name = url.netloc
    bucket_path = f"{url.netloc}{url.path}"

    # Generate the link
    # AnnotationLayer has the assumption that if "source" is a local_path
    # then it will be replaced by "s3://bucket_path/local_path" and files are
    # written out to local_path.
    # So we need to change pwd.
    os.chdir("../results")
    if ng_dir:
        os.makedirs(ng_dir, exist_ok=True)
    neuroglancer_link = NgState(
        input_config=input_config,
        mount_service="s3",
        bucket_path=bucket_path,
        output_dir=ng_dir,
        json_name=json_name,
    )
    neuroglancer_link.save_state_as_json()
    print(neuroglancer_link.get_url_link())
    os.chdir("../code")  # Back to our standard dir
    # thelink = f"https://neuroglancer-demo.appspot.com/#!{alignment_output_uri}/ng/{json_name}"
    # Local installation
    thelink = f"https://aind-neuroglancer-sauujisjxq-uw.a.run.app/#!{alignment_output_uri}/ng/{json_name}"
    with open("../results/ng/ng_link.txt", "a") as f:
        print(thelink, file=f)
    return thelink

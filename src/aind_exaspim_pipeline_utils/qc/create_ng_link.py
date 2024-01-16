"""Create Neuroglancer links for the alignments."""

# Copied from Jonathan's NG link generator capsule code

import json
from pathlib import Path
from urllib.parse import urlparse

from ng_link import NgState, link_utils, xml_parsing
import numpy as np
import os


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


def create_ng_link(
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
    vox_sizes: tuple[float, float, float] = xml_parsing.extract_tile_vox_size(xml_path)
    tile_paths: dict[int, str] = xml_parsing.extract_tile_paths(xml_path)
    tile_transforms: dict[int, list[dict]] = xml_parsing.extract_tile_transforms(xml_path)

    # Color info
    channel: int = link_utils.extract_channel_from_tile_path(tile_paths[0])
    hex_val: int = link_utils.wavelength_to_hex(channel)
    hex_str = f"#{str(hex(hex_val))[2:]}"

    # Zattrs info
    zattrs_positions = get_tile_positions("../data/exaspim_dataset/SPIM.ome.zarr")

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

    ng_dir, json_name = os.path.split(output_json)
    if ng_dir:
        os.makedirs(ng_dir, exist_ok=True)
    url = urlparse(dataset_uri)
    if url.scheme != "s3":
        raise ValueError(f"Dataset URI must be an S3 URI, got {dataset_uri}")
    bucket_name = url.netloc

    # Generate the link
    neuroglancer_link = NgState(
        input_config=input_config,
        mount_service="s3",
        bucket_path=bucket_name,
        output_json=ng_dir,
        json_name=json_name,
    )
    neuroglancer_link.save_state_as_json()
    print(neuroglancer_link.get_url_link())
    thelink = f"https://neuroglancer-demo.appspot.com/#!{alignment_output_uri}/ng/{json_name}"
    with open("../results/ng/ng_link.txt", "a") as f:
        print(thelink, file=f)
    return thelink

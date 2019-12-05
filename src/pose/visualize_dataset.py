import configparser
import logging

logger = logging.getLogger(__name__)
import cv2

from pose.hand_dataset.selector import select_dataset
from pose.hand_dataset import common_dataset


cv2.setNumThreads(0)


def visualize_dataset(config, debug=True, visualize=True, iterate_all=False):
    dataset = select_dataset(config, return_data=["train_set"], debug=debug)
    logger.info("done get dataset")
    color_map = common_dataset.COLOR_MAP
    keypoint_names = common_dataset.STANDARD_KEYPOINT_NAMES
    edges = common_dataset.EDGES

    if visualize:
        if config.getboolean("dataset", "use_rgb") and config.getboolean("dataset", "use_depth"):
            from pose.visualizations import visualize_both
            visualize_both(dataset, keypoint_names, edges, color_map, normalize=True)
        elif config.getboolean("dataset", "use_rgb"):
            from pose.visualizations import visualize_rgb
            visualize_rgb(dataset, keypoint_names, edges, color_map)
        elif config.getboolean("dataset", "use_depth"):
            from pose.visualizations import visualize_depth
            visualize_depth(dataset, keypoint_names, edges, color_map, normalize=True)
        else:
            pass

    if iterate_all:
        import chainer
        import tqdm
        iterator = chainer.iterators.MultiprocessIterator(dataset, batch_size=32, repeat=False)
        for batch in tqdm.tqdm(iterator):
            pass
        iterator = chainer.iterators.MultiprocessIterator(dataset, batch_size=32, repeat=False)
        for batch in tqdm.tqdm(iterator):
            pass


if __name__ == "__main__":
    dataset_type = "stb"
    import cv2

    config = configparser.ConfigParser()
    config.read("../config_pose.ini")
    config["dataset"]["train_set"] = dataset_type
    logging.basicConfig(level=logging.INFO)
    visualize_dataset(config, debug=True)

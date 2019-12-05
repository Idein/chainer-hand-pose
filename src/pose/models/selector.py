import logging

logger = logging.getLogger(__name__)


def select_model(config, hand_param):
    model_name = config["model"]["name"]
    logger.info("> use {}".format(model_name))
    if model_name == "ppn":
        from pose.models.model_ppn import PoseProposalNet
        feature_extractor = config["ppn"]["feature_extractor"]
        model_param = {}
        if feature_extractor == "mv2":
            model_param["model_name"] = "mv2"
            model_param["width_multiplier"] = config.getfloat("mv2", "width_multiplier")
        if feature_extractor == "resnet":
            model_param["model_name"] = "resnet" + config["resnet"]["n_layers"]
        logger.info("> model_param {}".format(model_param))
        model = PoseProposalNet(hand_param, model_param)
    else:
        raise NotImplementedError("[model] {} is not implemented".format(model_name))
    return model

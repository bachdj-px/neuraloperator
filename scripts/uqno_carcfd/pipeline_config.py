
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig


def get_config(config_path: str):
    # query points is [sdf_query_resolution] * 3 (taken from config ahmed)
    # Read the configuration
    config_name = "cfd"
    pipe = ConfigPipeline(
        [
            YamlConfig(
                config_path,
                config_name=config_name,
                config_folder="PATH/TO/CONFIG/FOLDER",
            ),
            ArgparseConfig(infer_types=True, config_name=None, config_file=None),
            YamlConfig(
                config_folder="PATH/TO/CONFIG/FOLDER",
            ),
        ]
    )
    config = pipe.read_conf()

    if config.data.sdf_query_resolution < config.fnogno.fno_n_modes[0]:
        config.fnogno.fno_n_modes = [config.data.sdf_query_resolution] * 3

    return config
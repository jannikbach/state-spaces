# MY_CW_MAIN.py
from cw2 import experiment, cw_error
from cw2.cw_data import cw_logging
from hydra.experimental import compose, initialize

import train


def compose_overrides(param_dict, parent_key='', sep='.'):
    items = []
    for k, v in param_dict.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(compose_overrides(v, new_key, sep=sep))
        else:
            items.append(f"{new_key}={v}")
    return items

class MyExperiment(experiment.AbstractExperiment):
    # ...

    def initialize(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        # Skip for Quickguide
        pass

    def run(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        # Perform your existing task)

        with initialize(config_path="./configs"):
            overrides = compose_overrides(config['params'])
            print(overrides)
            cfg = compose(config_name="config.yaml", overrides=overrides)
        train.main(cfg)

    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        # Skip for Quickguide
        pass





from cw2 import cluster_work

if __name__ == "__main__":
    # Give the MyExperiment Class, not MyExperiment() Object!!
    cw = cluster_work.ClusterWork(MyExperiment)

    # Optional: Add loggers
    # cw.add_logger(...)

    # RUN!
    cw.run()

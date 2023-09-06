# MY_CW_MAIN.py
from cw2 import experiment, cw_error
from cw2.cw_data import cw_logging
from hydra.experimental import compose, initialize

import train


class MyExperiment(experiment.AbstractExperiment):
    # ...

    def initialize(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        # Skip for Quickguide
        pass

    def run(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        # Perform your existing task)
        print(config) #config['params']['overrides'],
        with initialize(config_path="configs"):
            cfg = compose(config_name="config.yaml", overrides=[config['params']['overrides']])
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
import pathlib

from .handler.checkpoint import CheckpointHandler
from .handler.config import ConfigHandler
from .handler.folder import FolderHandler
from .log.logger import Logger


class Supporter():

    def __init__(self, experiments_dir=None, config_dir=None, config_dict=None, count_expt=False, reload_expt=False):

        if reload_expt:
            experiments_dir = pathlib.Path(experiments_dir)
            self.cfg = ConfigHandler(config_dir=experiments_dir / "config" / "config.yml", config_dict=None)
            self.folder = FolderHandler(experiments_dir)
        else:

            self.cfg = ConfigHandler(config_dir, config_dict)

            if experiments_dir is None and self.cfg.expt.experiments_dir is None:
                raise UserWarning("ConfigHandler: experiment_dir and config.expt.experiment_dir is None")
            elif experiments_dir is not None:
                self.cfg.expt.set_attr("experiments_dir", experiments_dir)
            else:
                experiments_dir = pathlib.Path(self.cfg.expt.experiments_dir)

            self.folder = FolderHandler(experiments_dir, self.cfg.expt.project_name, self.cfg.expt.session_name,
                                        self.cfg.expt.experiment_name, count_expt)
            self.cfg.save_config(self.folder.config_dir)

        self.logger = Logger(self.folder.log_dir)
        self.ckp = CheckpointHandler(self.folder.checkpoint_dir)

        self.logger.log("project_name", self.cfg.expt.project_name)
        self.logger.log("session_name", self.cfg.expt.session_name)
        self.logger.log("experiment_name", self.cfg.expt.experiment_name)

    def __enter__(self):
        self.logger.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.close()

    def get_logger(self):
        return self.logger

    def get_config(self):
        return self.cfg

    def get_checkpoint_handler(self):
        return self.ckp

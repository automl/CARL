"""
Handle the location, new folders and experiments sub-folder structure.

base_dir / project / session / experiment

experiment will be increased

"""
import pathlib

from .base_handler import Handler


class FolderHandler(Handler):

    def __init__(self, experiments_dir, project_name=None, session_name=None, experiment_name=None, count_expt=False):
        super().__init__()

        self.experiments_dir = pathlib.Path(experiments_dir)

        self.subfolder = ["log", "checkpoint", "config", "profile"]

        if project_name is not None:
            self.project_name = project_name
            self.session_name = session_name
            self.experiment_name = experiment_name
            self.count_expt = count_expt

            self.expt_dir = self.create_folders()
        else:
            self.expt_dir = self.experiments_dir

    def create_folders(self):

        dir = self.experiments_dir
        self.save_mkdir(dir)

        for folder in [self.project_name, self.session_name]:
            dir = dir / folder
            self.save_mkdir(dir)

        if self.count_expt:
            self.experiment_name = self.counting_name(dir, self.experiment_name)

        dir = dir / self.experiment_name
        self.save_mkdir(dir)

        for folder in self.subfolder:
            self.save_mkdir(dir / folder)

        return dir

    @property
    def dir(self):
        return self.expt_dir

    @property
    def config_dir(self):
        return self.expt_dir / "config"

    @property
    def log_dir(self):
        return self.expt_dir / "log"

    @property
    def profile_dir(self):
        return self.expt_dir / "profile"

    @property
    def checkpoint_dir(self):
        return self.expt_dir / "checkpoint"

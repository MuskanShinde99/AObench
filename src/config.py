from pathlib import Path
import os
import sys

class Config:
    """Central configuration of file system paths."""

    def __init__(self):
        self.opt_lab_root = Path(os.environ.get("OPT_LAB_ROOT", "/home/ristretto-dao/optlab-master"))
        self.project_root = Path(os.environ.get("PROJECT_ROOT", self.opt_lab_root / "PROJECTS_3/RISTRETTO/Banc AO"))
        # Ensure modules in optlab and project can be imported
        sys.path.append(str(self.opt_lab_root))
        sys.path.append(str(self.project_root))
        self.root_dir = self.project_root

        # Pre-compute commonly used output folders
        self.folder_calib = self.root_dir / 'outputs/Calibration_files'
        self.folder_pyr_mask = self.root_dir / 'outputs/3s_pyr_mask'
        self.folder_transformation_matrices = self.root_dir / 'outputs/Transformation_matrices'
        self.folder_closed_loop_tests = self.root_dir / 'outputs/Closed_loop_tests'
        self.folder_turbulence = self.root_dir / 'outputs/Phase_screens'
        self.folder_gui = self.root_dir / 'outputs/GUI_tests'

config = Config()

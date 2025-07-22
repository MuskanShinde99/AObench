from pathlib import Path
import dao
import toml

class ShmLoader:
    def __init__(self, toml_path: str | Path | None = None):
        if toml_path is None:
            toml_path = Path(__file__).with_name("shm_path.toml")
        self.toml_path = Path(toml_path)
        self.init_shm()

    def init_shm(self):
        with open(self.toml_path, 'r') as f:
            shm_path = toml.load(f)
        self.slopes_image_shm   = dao.shm(shm_path['slopes_image'])
        self.phase_screen_shm   = dao.shm(shm_path['phase_screen'])
        self.dm_phase_shm       = dao.shm(shm_path['dm_phase'])
        self.phase_residuals_shm= dao.shm(shm_path['phase_residuals'])
        self.normalized_psf_shm = dao.shm(shm_path['normalized_psf'])
        self.commands_shm       = dao.shm(shm_path['commands'])
        self.residual_modes_shm = dao.shm(shm_path['residual_modes'])
        self.computed_modes_shm = dao.shm(shm_path['computed_modes'])
        self.dm_kl_modes_shm    = dao.shm(shm_path['dm_kl_modes'])
        self.delay_shm          = dao.shm(shm_path['delay'])
        self.gain_shm           = dao.shm(shm_path['gain'])
        self.leakage_shm        = dao.shm(shm_path['leakage'])
        self.num_iterations_shm = dao.shm(shm_path['num_iterations'])
        self.cam1_shm           = dao.shm(shm_path['cam1'])
        self.cam2_shm           = dao.shm(shm_path['cam2'])

shm = ShmLoader()

__all__ = ["ShmLoader", "shm"]

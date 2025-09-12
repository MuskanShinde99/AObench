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
        """Load all shared memory handles defined in the TOML file."""
        with open(self.toml_path, "r") as f:
            shm_path = toml.load(f)

        for key, path in shm_path.items():
            setattr(self, f"{key}_shm", dao.shm(path))

shm = ShmLoader()

__all__ = ["ShmLoader", "shm"]



# with open('shm_path.toml', 'r') as f:
#     shm_path = toml.load(f)

# slopes_image_shm         = dao.shm(shm_path['slopes_image'])
# phase_screen_shm         = dao.shm(shm_path['phase_screen'])
# dm_phase_shm             = dao.shm(shm_path['dm_phase'])
# phase_residuals_shm      = dao.shm(shm_path['phase_residuals'])
# normalized_psf_shm       = dao.shm(shm_path['normalized_psf'])
# commands_shm             = dao.shm(shm_path['commands'])
# residual_modes_shm       = dao.shm(shm_path['residual_modes'])
# computed_modes_shm       = dao.shm(shm_path['computed_modes'])
# dm_kl_modes_shm          = dao.shm(shm_path['dm_kl_modes'])
# norm_flux_pyr_img_shm    = dao.shm(shm_path['norm_flux_pyr_img'])

# dm_act_shm               = dao.shm(shm_path['dm_act'])
# slm_shm                  = dao.shm(shm_path['slm'])

# # KL2Act_shm               = dao.shm(shm_path['KL2Act'])
# # KL2Phs_shm               = dao.shm(shm_path['KL2Phs'])

# # valid_pixels_mask_shm    = dao.shm(shm_path['valid_pixels_mask'])
# # reference_psf_shm        = dao.shm(shm_path['reference_psf'])
# # normalized_ref_psf_shm   = dao.shm(shm_path['normalized_ref_psf'])
# # reference_image_shm      = dao.shm(shm_path['reference_image'])
# normalized_ref_image_shm = dao.shm(shm_path['normalized_ref_image'])
# npix_valid_shm           = dao.shm(shm_path['npix_valid'])
# KL2PWFS_cube_shm         = dao.shm(shm_path['KL2PWFS_cube'])
# slopes_shm               = dao.shm(shm_path['slopes'])
# KL2S_shm                 = dao.shm(shm_path['KL2S'])
# S2KL_shm                 = dao.shm(shm_path['S2KL'])
# delay_set_shm            = dao.shm(shm_path['delay_set'])
# gain_shm                 = dao.shm(shm_path['gain'])
# leakage_shm              = dao.shm(shm_path['leakage'])
# num_iterations_shm       = dao.shm(shm_path['num_iterations'])
# cam1_shm                 = dao.shm(shm_path['cam1'])
# cam2_shm                 = dao.shm(shm_path['cam2'])
# cam1Fps_shm              = dao.shm(shm_path['cam1Fps'])
# cam2Fps_shm              = dao.shm(shm_path['cam2Fps'])
# KL2Act_papy_shm          = dao.shm(shm_path['KL2Act_papy'])

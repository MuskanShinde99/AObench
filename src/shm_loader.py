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




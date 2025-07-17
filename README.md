# AObench
The AO bench code to test the pyramid wavefront sensor

## Configuration

The repository relies on `src/config.py` and `dao_setup.py` for initializing
hardware and paths. By default paths are based on `/home/ristretto-dao/optlab-master`,
but you can override them using environment variables:

```
export OPT_LAB_ROOT=/path/to/optlab-master
export PROJECT_ROOT=/path/to/project
```

These variables allow running the code from different locations without
modifying the source files. Output directories are created under
`PROJECT_ROOT`, which defaults to `${OPT_LAB_ROOT}/PROJECTS_3/RISTRETTO/Banc AO`.

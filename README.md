Updated fork of the https://github.com/EkaterinaSe/wrapper_multi

Changes:
- Using Dask for multiprocessing
- Dask allows to not presplit jobs -> can be 10-20% faster in the long run
- Fixes and QOL updates to the combine_grids
- Updated m1d_output according to latest m1d updates (?) Not sure if changed anything tbh
- ABUND file is scaled to the atmospheric abundance (if given, e.g. MARCS) or scales M1D ABUND file according to the metallicity of the atmosphere
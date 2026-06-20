# Masters Thesis Git - Elias and Kajsa


## Installing CIL with Astra and Tigre - SSLERROR
If you encounter SSLError while installing CIL, then use the command:
* conda install --insecure -y -c conda-forge -c https://software.repos.intel.com/python/conda -c ccpi cil=24.2.0 ipp=2021.12 astra-toolbox=*=cuda* tigre k3d
or, if creating an environment 
* conda create --insecure -y -n your-env-name \
  -c conda-forge \
  -c https://software.repos.intel.com/python/conda \
  -c ccpi \
  cil=24.2.0 ipp=2021.12 astra-toolbox=*=cuda* tigre k3d

this temporarily disables ssl_verify which previously blocks the download


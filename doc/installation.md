# Installation

This program is written in Python and requires a number of external libraries. Dependencies can be installed either
using [conda](https://docs.conda.io/en/latest/) or [venv](https://docs.python.org/3/library/venv.html).

### Installation using Conda

```console
conda create -n mgraph python=3.8 -y
conda activate mgraph
conda install --file requirements_conda.txt -y
python -m pip install -r requirements.txt
```

### Installation using Venv

```console
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```


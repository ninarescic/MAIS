
# <img src="./doc/fig/mais.png" style="float: right;" width="5%"/> MAIS
## Multi-Agent Information/Infection Spread Model 



<!--- PDF BREAK -->

The MAIS Model is a collection of agent based network models for simulation of information or infection spread. 
You can use your own network (graph) or play with demo graphs included in this repository. You can derive your own models with customised set of agent states or encode customised policy modules.   

For information spread use:
  + [InfoSIRModel](src/models/agent_info_models.py)
    - the implementation of SIR model
    - parameters:
      - `beta`: transmision strenght
      - `I_duration`: duration in state I in days
    - policy functions:
      - [`Spreader`](src/policies/spreader_policy.py): seeds the source of information to the node with pagerank corresponding to given quantile             
  + InfoTippingModel
    - to be implemented soon 

 For infection spread use:
   + [SimulationDrivenModel](src/models/agent_based_network_model.py)
      - See the [model documentation](doc/model.md) for technical details.


## Examples of Simulation Results

Please follow the links to find out more details about the examples presented.
+ [InfoSIRModel](doc/sir.md) <br>
  Simple examples of information spread modelling using SIR model `InfoSIRModel`.
+ [TippingModel](doc/tipping.md) <br>
  Simple examples of information spread modelling using Tipping model `InfoTippingModel`.
+ [Demo](doc/demo.md) <br>
  Simple examples of infection transmission model using `SimulationDrivenModel`.


# Installation

All the requirements can be installed using [conda](https://docs.conda.io/en/latest/):

```console
conda create -n mgraph python=3.8 -y
conda activate mgraph
conda install --file requirements_conda.txt -y
python -m pip install -r requirements.txt
```

For other options and/or more help please refer to the [installation instructions](doc/installation.md).

# Usage

All the executable scripts are located in the [scripts](scripts) subfolder. So first of all run:

```console
cd scripts
```

Most of the following commands take as a parameter the name of an INI file. The INI file describes all the configuration
settings and locations of other files used. Please refer to [INI file specification](doc/inifile.md) for details.

There are several INIs provided so that you can base your experiments on these settings:

|filename|description|
|---|---|
|[demo.ini](config/demo.ini)&nbsp;<sup>1</sup>| Very small region (5k inhabitants) for demonstration purposes.|
|[papertown.ini](config/papertown.ini)&nbsp;<sup>2</sup>| Hodoninsko region as referred in paper [[preprint](https://doi.org/10.1101/2021.05.13.21257139)]|
|[hodoninsko.ini](config/hodoninsko.ini)| Hodonínsko region (57k persons)|
|[lounsko.ini](config/lounsko.ini)| Lounsko region (42k persons)|

<sup>1</sup> All the intermediate outputs of all scripts for `demo.ini` are included in this repository. Therefore you do not
need to run the scripts in the described order.

<sup>2</sup> Graph files of `papertown.ini` are included in this repository. It is not possible to generate this graph
from scratch again by `generate.py` script since several non-public data sources were used for its creation. This graph
was used for experiments presented in our preprints.

### 1. Generation of a graph

Unless you use `papertown` or `demo` graph that are included in this repository, you have to generate your graph. For example:

```console
python generate.py ../config/hodoninsko.ini 
```

would generate a m-graph for Hodonínsko region. For further information please refer to the documentation of
the [generate](doc/generate.md) command.

### 2. Running your experiments

Run your experiment. Note that the first time you run it, the graph is loaded from CSV files, which takes several minutes.

+ If you wish to run one simulation only, use `run_experiment.py`:

```console
python run_experiment.py -r ../config/hodoninsko.ini my_experiment
```

After the run finishes, you should find the file `history_my_experiment.csv` in the directory specified as `output_dir`
in your [INI file](doc/inifile.md#task). The INI files provided use `data/output/model` directory.

+ For a proper experiment, you should evaluate the model more times. You can do it in parallel using:

```console
python run_multi_experiment.py -R ../config/random_seeds.txt --n_repeat=100 --n_jobs=4 ../config/hodoninsko.ini my_experiment
```

By default it produces a ZIP file with the resulting history files. You can change `output_type` to FEATHER and the result
will be stored as one data frame in the feather format. The resulting file is stored in the directory specified
by `output_dir` and it is named `history_my_experiment.zip` or `history_my_experiment.feather`.

### 3. Result visualisation

Now you can create a plot from the resulting files and save it to the path specified by `--out_file PATH_TO_IMG`.

```console
python plot_experiments.py ../data/output/model/history_my_experiment.zip --out_file ./example_img.png
```

<!--- PDF BREAK --><!--- PDF BREAK -->

## Configuration and Advanced Features

Please consult [How to run simulations](doc/run.md) for options of individual scripts,
[INI file specification](doc/inifile.md), and [How to fit the paremeters](doc/run.md#6-fitting-your-model).

## Acknowledgement

**TODO**

## How to cite

**TODO**

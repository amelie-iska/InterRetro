
# Interactive Retrosynthesis Planning (InterRetro)

This is the official code repository for the paper *Retrosynthesis Planning via Worst-path Policy Optimisation in Tree-structured MDPs*.

## Setup

The complete codebase is organised into three subfolders:

```
InterRetro
|- Graph2Edits
|- InterRetro
|- dataset
```

The `Graph2Edits` and `InterRetro` folders have already been built and initialised; no further action is required.
For the `dataset` folder, please download it from [this link](https://drive.google.com/drive/folders/198WuPlSyMeMvvd4i2SM833jPAcGllzDu).

## Usage

To train the InterRetro model, run the following command:

```
python InterRetro/main.py
```

You can find available configuration options in the `config_parser.py` file.

## Acknowledgements

We acknowledge that portions of this code have been adapted from [Retro\*](https://github.com/binghong-ml/retro_star) and [Graph2Edits](https://github.com/Jamson-Zhong/Graph2Edits), both of which are distributed under the MIT License. This code is also released under the MIT License.


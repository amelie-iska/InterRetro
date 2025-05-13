# Interactive Retrosynthesis Planning (InterRetro)

This is the official code repository for the paper *Retrosynthesis Planning via Worst-path Policy Optimisation in Tree-structured MDPs*.

# Setup

The complete code is organised into four subfolders:
```
InterRetro
|- Graph2Edits # The Graph2Edits model
|- RetroRL # InterRetro
|- retro_star # Retro* search
|- dataset
```


## Setup I: Dataset Folder

## Setup II: Graph2Edits Folder

## Setup III: retro_star Folder


# Usage

To train the InterRetro model, run the following command:

```
python InterRetro/main.py
```
You can find available configuration options in the ```config_parser.py``` file.

# Acknowledgement

We acknowledge that portions of this code have been modified or copied from [Retro*](https://github.com/binghong-ml/retro_star) and [Graph2Edits](https://github.com/Jamson-Zhong/Graph2Edits), both of which are distributed under the MIT License. This code is also released under the MIT License.

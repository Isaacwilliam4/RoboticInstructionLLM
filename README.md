# RoboticInstructionLLM

First you need to setup an environment, for this project it should be python 3.10 you can do this in conda with the following command

```
conda create -n robotllm python=3.10 -y
conda activate robotllm
conda install -c conda-forge numpy matplotlib
```

Then you need to install the python package to get the environment to work correctly

```
cd multigrid/
pip install -e .
```

Once that is setup and the conda environment is activated, you can run the simulation to start training the agents

```
cd RoboticInstructionLLM
python train.py ./data/instructions.json
```

## Attribution
Portions of this project were adapted from [DeepRL-Grounding](https://github.com/devendrachaplot/DeepRL-Grounding) by Chaplot et al, [Minigrid](https://github.com/Farama-Foundation/Minigrid), which are available under the MIT license.

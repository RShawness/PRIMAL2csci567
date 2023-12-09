# PRIMAL_2: Pathfinding via Reinforcement and Imitation Multi_agent Learning - Lifelong

NOTE!!!: This code base is a modified version of [PRIMAL2](https://github.com/marmotlab/PRIMAL2) by Mehul Damani, Zhiyao Luo, Emerson Wenzel, and Guillaume Sartoretti. This version has a modified action space where agents are only permitted to wait, move forward, rotate clockwise, and rotate counter clockwise.

## Setting up Code
- cd into the CBS_for_PRIMAL folder.
- run ```git clone https://github.com/pybind/pybind11.git``` to install pybind for CBS
- Then, you can compile CBS by running
  ```
  mkdir build
  cd build
  cmake ..
  make -j
  ```
- go back to the parent directory to train or test the code


## Running Code
- Pick appropriate number of meta agents via variables `NUM_META_AGENTS` and `NUM_IL_META_AGENTS` in `parameters.py`. We aimed at around 35% to 50% RL to IL ratio when training out model
- The number of RL meta-agents is implicity defined by the difference between total meta-agents and IL meta-agents (`NUM_RL_META_AGENTS` = `NUM_META_AGENTS` - `NUM_IL_META_AGENTS`)
- Name training run via `training_version` in `parameters.py`
- call `python driver.py`

## Reproducing our Result
- [Here](https://drive.google.com/drive/folders/10cE1-lF5A3i0L85499-_S_pl4loPtS-n) we provide the 2 trained models which were used to produce our results for the final project. set the model name at line 759 of TestingEnv.py to the model you want to test
- You can create the result by running ```python TestingEnv.py```
- If instead you want to test the CBS result, run ```python TestingEnv.py -p CBS```

## Key Files
- `parameters.py` - Training parameters.
- `driver.py` - Driver of program. Holds global network for A3C.
- `Runner.py` - Compute node for training. Maintains a single meta agent.
- `Worker.py` - A single agent in a simulation environment. Majority of episode computation, including gradient calculation, occurs here.
- `Ray_ACNet.py` - Defines network architecture.
- `Env_Builder.py` - Defines the lower level structure of the Lifelong MAPF environment for PRIMAL2, including the world and agents class.
- `PRIMAL2Env.py` - Defines the high level environment class. 
- `Map_Generator2.py` - Algorithm used to generate worlds, parameterized by world size, obstacle density and wall components.
- `PRIMAL2Observer.py` - Defines the decentralized observation of each PRIMAL2 agent.
- `Obsever_Builder.py` - The high level observation class
- `TestingEnv.py` - The testing environment used to produce result for our final project


## Authors

[Nien-Shao Wang](nienshao@usc.edu)

[John C. Bush](johncbus@usc.edu)

[Ryan Shaw](rfshaw@usc.edu)

[Ayan Bhowmick](abhowmic@usc.edu)

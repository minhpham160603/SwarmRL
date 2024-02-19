# SwarmRL Environment
This project implements the reinforcement learning environment for the Swarm Rescue simulator. The implementation adheres to the [Gymnasium](https://gymnasium.farama.org/) interface for the single-agent environment, and [Pettingzoo](https://pettingzoo.farama.org/) interface for multi-agent settings. (A Gymnaisum interface version of multi-agent is provided, but the observation and action space are not within the standard spaces. That version is for custom MAPPO training using the implementation of [onpolicy](https://github.com/marlbenchmark/on-policy/tree/main)). All source code for the implementation can be found in the /swarm_env directory.

# Implementation details
- There is a slight modification in the original spg_overlay code to support optimal observation for the agents: There is an additional `special_semantic()` method in the `drone_distance_sensors` module that provides the id of the detected entity so that the drones can use one semantic data for each entity. This improves learning significantly and get rid of the need to use one-hot encoding for `entity_type`.

# Basic usage
- Before running, install the required packages: `pip install -r requirements.txt`. It is required to have `spg` (simple_playground) in order to run the env.
- The codes to render the env are provided in the `/demos` directory.
- The codes to train env with Stable Baselines3 are provided in the `/training` directory.

# SwarmRL Environment
This project implements the reinforcement learning environment for the Swarm Rescue simulator. The implementation adheres to the [Gymnasium](https://gymnasium.farama.org/) interface for the single-agent environment, and [Pettingzoo](https://pettingzoo.farama.org/) interface for multi-agent settings. (A Gymnaisum interface version of multi-agent is provided, but the observation and action space are not within the standard spaces. The version for custom MAPPO training can be found [here](https://github.com/minhpham160603/SwarmMARL). All source code for the implementation can be found in the `/swarm_env` directory.

# Implementation details
- There is a slight modification in the original spg_overlay code to support optimal observation for the agents: There is an additional `special_semantic()` method in the `drone_distance_sensors` module that provides the id of the detected entity so that the drones can use one semantic data for each entity. This improves learning significantly and get rid of the need to use one-hot encoding for `entity_type`.

# Basic usage
- Before running, install the required packages: `pip install -r requirements.txt`. It is required to have `spg` (simple_playground) to run the env.
- The codes to render the env are provided in the `/demos` directory.
- The codes to train env with Stable Baselines3 are provided in the `/training` directory.

# Trouble shooting
### To run via SSH:
Running SwarmRL via SSH requires Xvfb. In case you do not have root access on SSH server, you can use this executable version of Xvfb on Linux x86: [Link](https://app.box.com/s/jlhpq6dbet6594a26f71mbuux07jzhoh).

```
$ ./Xvfb :99 -screen 0 1024x768x24 &
$ export DISPLAY=:99
```
If there is missing shared libraries:
- Run `ldd ./Xvfb` to check for missing objects.
- Run `export LD_LIBRARY_PATH=[your_path_to_download]/lib:$LD_LIBRARY_PATH` to add new directory for executable to look for new objects file. (The link above has some objects file that are usually missing, you can add more files to it if missing).
- Run `ldd ./Xvfb` again to check if the new path is found.
- Try running the Python program again.

### Miscellanious
If you are using Ubuntu and face this problem with libGL: `libGL error: MESA-LOADER: failed to open iris: /usr/lib/dri/iris_dri.so`, try:
```
find / -name libstdc++.so.6 2>/dev/null

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 

source ~/.bashrc
```
Otherwise, using Xvfb will solve the problem.

# Example Display 
SAC in Intermediate map, trained for 500,000 steps.

![intermediate_SAC](intermediate_SAC.gif)

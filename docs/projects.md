# Projects using Crazyflow

Research and teaching projects built on Crazyflow. They show how the simulator is used beyond the self-contained [examples](examples/index.md).

## LSY Drone Racing

[learnsyslab/lsy_drone_racing](https://github.com/learnsyslab/lsy_drone_racing)

A course project for developing and evaluating autonomous drone racing algorithms in simulation and on real Crazyflie hardware. Crazyflow simulates the race tracks and is used to test racing controllers before sim-to-real transfer. See the [project documentation](https://learnsyslab.github.io/lsy_drone_racing/) to get started.

## Crazyflow Experiments

[learnsyslab/crazyflow_experiments](https://github.com/learnsyslab/crazyflow_experiments)

The experiment code for the [Crazyflow paper](https://arxiv.org/abs/2606.01478). It covers simulator speed benchmarks, sim-to-real validation, system identification, learned controllers, sampling-based obstacle avoidance, drone racing, and vision policies trained on Gaussian splats.

## SwarmGPT

[learnsyslab/swarmGPT](https://github.com/learnsyslab/swarmGPT)

Combines large language models with safe swarm motion planning to generate synchronized drone choreographies from natural language instructions. Crazyflow simulates the swarm to verify choreographies before deployment. See the [project website](https://utiasdsl.github.io/swarmGPT/) and the [paper](https://ieeexplore.ieee.org/document/11197931/).

## Crazyflow UWB

[learnsyslab/crazyflow_uwb](https://github.com/learnsyslab/crazyflow_uwb)

Simulates ultra-wideband (UWB) positioning as a Crazyflow plugin. A UWB tag measures noisy, biased ranges to a set of anchors and solves them for a position, which the controller then uses in place of the true state. See the [paper](https://arxiv.org/abs/2610.08225).

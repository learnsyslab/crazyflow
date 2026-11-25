"""Sensors for the simulation.

[depth][crazyflow.sim.sensors.depth] renders depth images with MuJoCo raycasting and is always
available. [rgbd][crazyflow.sim.sensors.rgbd] renders RGB-D images of the model geometry and
requires the optional ``rgbd`` extra. [splat][crazyflow.sim.sensors.splat] renders photorealistic
RGB(-D) images from gaussian splats and requires the optional ``splats`` extra.
"""

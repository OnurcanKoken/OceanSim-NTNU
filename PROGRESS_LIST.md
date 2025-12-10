# TODO

- [x] Publish point cloud to ros2 topic
- [ ] Make custom config file for the sonar
- [ ] Make code to add some random objects

# Notes

Make a mesh through create-mesh and then in property dont forget to add the physics-collision

to add an usd without messing up the stage do file-add reference-.usd file. Then press reset to be able to press the run button.

there is a python file (@ tools/isaacsim.sensors.rtx/) to convert a json config file of a lidar to an usd file that you can import in the scene and wrap with the lidar class. Only you can only wrap around it if the api schema is added, which i dont know now how to do: OmniSensorGenericLidarCoreAPI. Maybe ask gemini to add this or just choose to overwrite some key parameters. 
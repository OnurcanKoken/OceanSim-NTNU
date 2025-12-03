from isaacsim.sensors.rtx import LidarRtx

class Sonar3D(LidarRtx):
    def __init__(self, prim_path, translation, orientation, config_file_name):
        super().__init__(prim_path=prim_path,
                         name="Sonar3D",
                         translation=translation,
                         orientation=orientation,
                         config_file_name=config_file_name)
        
        self.attach_writer('RtxLidarDebugDrawPointCloudBuffer')
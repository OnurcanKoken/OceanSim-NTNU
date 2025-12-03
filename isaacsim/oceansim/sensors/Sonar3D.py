from isaacsim.sensors.rtx import LidarRtx
import omni.replicator.core as rep

class Sonar3D(LidarRtx):
    def __init__(self, prim_path, translation, orientation, config_file_name):
        super().__init__(prim_path=prim_path,
                         name="Sonar3D",
                         translation=translation,
                         orientation=orientation,
                         config_file_name=config_file_name)
        
        self.attach_writer('RtxLidarDebugDrawPointCloudBuffer')
        # self.attach_annotator('IsaacExtractRTXSensorPointCloudNoAccumulator')

        self._render_product_path = self.get_render_product_path()
        self.writer = rep.writers.get("RtxLidar" + "ROS2PublishPointCloud")
        self.writer.initialize(topicName="sonar3D_point_cloud", frameId="map")
        self.writer.attach([self._render_product_path])


    def _draw_pointcloud_simulation(self):
        self.attach_writer('RtxLidarDebugDrawPointCloudBuffer')

    def close(self):
        self.writer.detach()
        self.detach_all_annotators()
        self.detach_all_writers()

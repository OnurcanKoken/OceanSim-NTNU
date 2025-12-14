from isaacsim.sensors.rtx import LidarRtx
import omni.replicator.core as rep
import carb
import omni.kit.app

class Sonar3D(LidarRtx):
    def __init__(self, prim_path, translation, orientation, config_file_name,**sensor_atributes):
        super().__init__(prim_path=prim_path,
                         name="Sonar3D",
                         translation=translation,
                         orientation=orientation,
                         config_file_name=config_file_name,
                         **sensor_atributes)
    
    def initialize(self):
        
        # The point cloud you see in IsaacSim
        self.attach_writer('RtxLidarDebugDrawPointCloudBuffer')
        # This annotator is for the data, but we dont use it but publish it directly with the writer
        # self.attach_annotator('IsaacExtractRTXSensorPointCloudNoAccumulator')

        # The class automatically makes a render product 
        self._render_product_path = self.get_render_product_path()

        # This writer handles publishing the data to ROS2 topic
        self.writer = rep.writers.get("RtxLidar" + "ROS2PublishPointCloud")
        # Define the topic name that you see in the ros2 topics list and set frame ID to "map" so you can visualize it in Rviz2
        self.writer.initialize(topicName="sonar3D_point_cloud", frameId="map")
        # Attach the render product of the lidar to this writer
        self.writer.attach([self._render_product_path])

    def close(self):
        # These writers and annotators used memory and you need to release them
        self.writer.detach() 
        self.detach_all_annotators() 
        self.detach_all_writers()


class Sonar3D_timestamp(LidarRtx):
    def __init__(self, prim_path, translation, orientation, config_file_name,**sensor_atributes):
        super().__init__(prim_path=prim_path,
                         name="Sonar3D",
                         translation=translation,
                         orientation=orientation,
                         config_file_name=config_file_name,
                         **sensor_atributes)
        self.pointcloud_annotator = None
    
    def initialize(self):
        # The point cloud you see in IsaacSim
        self.attach_writer('RtxLidarDebugDrawPointCloudBuffer')
        
        self.pointcloud_annotator = self.attach_annotator(
            "IsaacCreateRTXLidarScanBuffer", 
            outputTimestamp=True,
        )
        

    def get_annotated_data(self):
        if self.pointcloud_annotator is None:
            return None

        frame = self.get_current_frame()
        node_path = self.pointcloud_annotator.get_node_path()

        if node_path not in frame:
            return None

        return frame[node_path]

    def close(self):
        # These writers and annotators used memory and you need to release them
        self.writer.detach() 
        self.detach_all_annotators() 
        self.detach_all_writers()

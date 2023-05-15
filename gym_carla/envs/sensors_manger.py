#!/usr/bin/env python

# Copyright (c) 2023: Changmeng Hou (houchangmeng@gmail.edu).
#
# This file is modified from <https://github.com/carla-simulator/carla>:
# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Script that render multiple sensors in the same pygame window

By default, it renders four cameras, one LiDAR and one Semantic LiDAR.
It can easily be configure for any different number of sensors. 
To do that, check lines 290-308.
"""

import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import argparse
import random
import time
import numpy as np
import weakref
from carla import ColorConverter as cc
import collections
import math

try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')


class CustomTimer:
    def __init__(self):
        try:
            self.timer = time.perf_counter
        except AttributeError:
            self.timer = time.time

    def time(self):
        return self.timer()


class DisplayManager:
    def __init__(self, grid_size, window_size,enable=True):
        
        pygame.init()
        pygame.font.init()
        
        if enable:
            self.display = pygame.display.set_mode(window_size, pygame.HWSURFACE | pygame.DOUBLEBUF )
        else:
            self.display = pygame.display.set_mode(window_size, pygame.HIDDEN | pygame.HWSURFACE | pygame.DOUBLEBUF )

        self.grid_size = grid_size
        self.window_size = window_size
        self.sensor_list = []

    def get_window_size(self):
        return [int(self.window_size[0]), int(self.window_size[1])]

    def get_display_size(self):
        return [int(self.window_size[0]/self.grid_size[1]), int(self.window_size[1]/self.grid_size[0])] # 2*3

    def get_display_offset(self, gridPos):
        dis_size = self.get_display_size()
        return [int(gridPos[1] * dis_size[0]), int(gridPos[0] * dis_size[1])]

    def add_sensor(self, sensor):
        self.sensor_list.append(sensor)

    def get_sensor_list(self):
        return self.sensor_list

    def render(self):
        if not self.render_enabled():
            return

        for s in self.sensor_list:
            s.render()

        pygame.display.flip()

    def clear(self):
        self.sensor_list.clear()

    def render_enabled(self):
        return self.display != None


Cameras = \
{
    'rgb':  ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
    'depth1': ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)', {}],
    'depth2': ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)', {}],
    'depth3': ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)', {}],
    'semantic1': ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)', {}],
    'semantic2': ['sensor.camera.semantic_segmentation', cc.CityScapesPalette, 'Camera Semantic Segmentation (CityScapes Palette)', {}],
    'instance1': ['sensor.camera.instance_segmentation', cc.Raw, 'Camera Instance Segmentation (Raw)', {}],
    'instance2': ['sensor.camera.instance_segmentation', cc.CityScapesPalette, 'Camera Instance Segmentation (CityScapes Palette)', {}],
    'lidar': ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)', {'range': '50','channels': '32'}],
    'dvs': ['sensor.camera.dvs', cc.Raw, 'Dynamic Vision Sensor', {}],
    'rgb_distorted': ['sensor.camera.rgb', cc.Raw, 'Camera RGB Distorted',
        {'lens_circle_multiplier': '3.0',
        'lens_circle_falloff': '3.0',
        'chromatic_aberration_intensity': '0.5',
        'chromatic_aberration_offset': '0'}],
    'optical_flow': ['sensor.camera.optical_flow', cc.Raw, 'Optical Flow', {}],
}


class CameraManager(object):
    def __init__(
        self, 
        sensor_type, 
        parent_actor,transform, 
        display_man, display_pos, options= {}):
        self.surface = None
        self._parent = parent_actor
        self.world = self._parent.get_world()
        self.bp_library = self.world.get_blueprint_library()

        self.display_man = display_man
        self.display_pos = display_pos
        self.timer = CustomTimer()

        self.data = None
        self.birdeye = None

        self.time_processing = 0.0
        self.tics_processing = 0
        self.display_man.add_sensor(self)

        self.sensor_item = sensor_type
        self.sensor = self.init_sensor(sensor_type, transform, self._parent, options)

    def init_sensor(self,sensor_item, transform, attached, sensor_options):

        sensor_bp = self.bp_library.find(sensor_item[0])
        if sensor_item[0].startswith('sensor.camera'):
            disp_size = self.display_man.get_display_size()
            sensor_bp.set_attribute('image_size_x', str(disp_size[0]))
            sensor_bp.set_attribute('image_size_y', str(disp_size[1]))
            if sensor_bp.has_attribute('gamma'):
                gamma_correction = 2.2
                sensor_bp.set_attribute('gamma', str(gamma_correction))
        elif sensor_item[0].startswith('sensor.lidar'):
            sensor_bp.set_attribute('dropoff_general_rate', '0.05')
            sensor_bp.set_attribute('points_per_second', '86000')
            sensor_bp.set_attribute('rotation_frequency', '10')
            sensor_bp.set_attribute('dropoff_intensity_limit', sensor_bp.get_attribute('dropoff_intensity_limit').recommended_values[0])
            sensor_bp.set_attribute('dropoff_zero_intensity', sensor_bp.get_attribute('dropoff_zero_intensity').recommended_values[0])
            
        for attr_name, attr_value in sensor_item[3].items():
            sensor_bp.set_attribute(attr_name, attr_value)
            
        for key in sensor_options:
            sensor_bp.set_attribute(key, sensor_options[key])

        sensor = self.world.spawn_actor(sensor_bp,transform,attach_to=attached)
        weak_self = weakref.ref(self)
        sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        return sensor

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return

        t_start = self.timer.time()
        if self.sensor_item[0].startswith('sensor.lidar'):
            
            disp_size = self.display_man.get_display_size() # 256
            points = np.frombuffer(image.raw_data, dtype=np.float32).reshape(-1, 4)

            d_behind = 8
            obs_range = 40
            pixel_per_meter = disp_size[0] / obs_range
            image = np.zeros((256, 256, 3), dtype=np.uint8)

            # set background is black
            image.fill(0)  
            # render points that is above ground with green.
            high_points = points[points[:, 2] >-2.0]
            high_x = (high_points[:, 0] + d_behind) * pixel_per_meter
            high_y = (high_points[:, 1] + obs_range / 2) * pixel_per_meter
            high_x = np.int32(np.clip(high_x, 0, disp_size[0] - 1))
            high_y = np.int32(np.clip(high_y, 0, disp_size[0] - 1))
            image[high_y, high_x, 1] = 255

            # render points that is below ground with green.
            ground_points = points[points[:, 2] <-2.0]
            ground_x = (ground_points[:, 0] + d_behind) * pixel_per_meter
            ground_y = (ground_points[:, 1] + obs_range / 2) * pixel_per_meter
            ground_x = np.int32(np.clip(ground_x, 0, disp_size[0] - 1))
            ground_y = np.int32(np.clip(ground_y, 0, disp_size[0] - 1))
            image[ground_y, ground_x, 0] = 255

            lidar_img = np.flip(image, axis=1)
            lidar_img = lidar_img.swapaxes(0, 1)
            self.data = lidar_img
            

        elif self.sensor_item[0].startswith('sensor.camera.dvs'):

            # Example of converting the raw_data from a carla.DVSEventArray
            # sensor into a NumPy array and using it as an image
            dvs_events = np.frombuffer(image.raw_data, dtype=np.dtype([
                ('x', np.uint16), ('y', np.uint16), ('t', np.int64), ('pol', bool)]))
            dvs_img = np.zeros((image.height, image.width, 3), dtype=np.uint8)
            # Blue is positive, red is negative
            dvs_img[dvs_events[:]['y'], dvs_events[:]['x'], dvs_events[:]['pol'] * 2] = 255
            
            self.data = dvs_img

        
        elif self.sensor_item[0].startswith('sensor.camera.optical_flow'):
            image = image.get_color_coded_flow()
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.data = array
            
        else:

            image.convert(self.sensor_item[1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.data = array

        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1
        pass

    def get_sensor(self):
        return self.sensor

    def get_data(self):
        return self.data

    def render(self):
        # if self.surface is not None:
        if self.data is None:
            return
        img = self.data.swapaxes(0, 1)
        
        if self.sensor_item[0].startswith('sensor.lidar'):
            #if sensor is lidar
            if self.birdeye is not None:
                birdeye = np.flip(self.birdeye, axis=1)
                birdeye = np.rot90(birdeye, 1)
                img[:,:,2] = (birdeye[:,:,0] <= 10) * (birdeye[:,:,1] <= 10) * (birdeye[:,:,2] >= 240) * 255
                
        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(img)

        offset = self.display_man.get_display_offset(self.display_pos)
        self.display_man.display.blit(self.surface, offset)

    def destroy(self):
        self.sensor.destroy()

class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud = None):
        self.sensor = None

        # If the spawn object is not a vehicle, we cannot use the Lane Invasion Sensor
        if parent_actor.type_id.startswith("vehicle."):
            self._parent = parent_actor
            self.history = []
            #self.hud = hud
            world = self._parent.get_world()
            bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
            self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid circular
            # reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))
    
    def get_invasion_history(self):
        return self.history

    def destroy(self):
        self.sensor.destroy()

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        self.history.append(lane_types)
        if len(self.history) > 4000:
            self.history.pop(0)
        
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        # print('lane invasion: ',text)

class CollisionSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history
    
    def destroy(self):
        self.sensor.destroy()

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)
        # print('collision intensity:' ,intensity)

class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        self.data = None
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    def destroy(self):
        self.sensor.destroy()
    
    def get_data(self):
        return self.data

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude
        self.data = [self.lat, self.lon]
        # print('gnss lat %.4f ,lon %.4f'%(self.lat,self.lon ))

class IMUSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        self.data = []
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.imu')
        self.sensor = world.spawn_actor(
            bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda sensor_data: IMUSensor._IMU_callback(weak_self, sensor_data))
    
    def get_data(self):
        return self.data
    
    def destroy(self):
        self.sensor.destroy()

    @staticmethod
    def _IMU_callback(weak_self, sensor_data):
        self = weak_self()
        if not self:
            return
        limits = (-99.9, 99.9)
        self.accelerometer = (
            max(limits[0], min(limits[1], sensor_data.accelerometer.x)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.y)),
            max(limits[0], min(limits[1], sensor_data.accelerometer.z)))
        self.gyroscope = (
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.x))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.y))),
            max(limits[0], min(limits[1], math.degrees(sensor_data.gyroscope.z))))
        self.compass = math.degrees(sensor_data.compass)
        self.data = [self.accelerometer,self.gyroscope,self.compass]
        # print('acc , gyroscope  , compass f' ,
        #     self.accelerometer,self.gyroscope,self.compass)
        
class RadarSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        bound_x = 0.5 + self._parent.bounding_box.extent.x
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        bound_z = 0.5 + self._parent.bounding_box.extent.z
        self.data = None

        self.velocity_range = 7.5 # m/s
        world = self._parent.get_world()
        self.debug = world.debug
        bp = world.get_blueprint_library().find('sensor.other.radar')
        bp.set_attribute('horizontal_fov', str(35))
        bp.set_attribute('vertical_fov', str(20))
        self.sensor = world.spawn_actor(
            bp,
            carla.Transform(
                carla.Location(x=bound_x + 0.05, z=bound_z+0.05),
                carla.Rotation(pitch=5)),
            attach_to=self._parent)
        # We need a weak reference to self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(
            lambda radar_data: RadarSensor._Radar_callback(weak_self, radar_data))
    
    def destroy(self):
        self.sensor.destroy()
    
    def get_data(self):
        return self.data
    
    def render(self):
        if self.surface is not None:
            offset = self.display_man.get_display_offset(self.display_pos)
            self.display_man.display.blit(self.surface, offset)

    @staticmethod
    def _Radar_callback(weak_self, radar_data):
        self = weak_self()
        if not self:
            return
        # To get a numpy [[vel, altitude, azimuth, depth],...[,,,]]:
        points = np.frombuffer(radar_data.raw_data, dtype=np.dtype('f4'))
        self.data = np.reshape(points, (len(radar_data), 4))

        current_rot = radar_data.transform.rotation
        for detect in radar_data:
            azi = math.degrees(detect.azimuth)
            alt = math.degrees(detect.altitude)
            # The 0.25 adjusts a bit the distance so the dots can
            # be properly seen
            fw_vec = carla.Vector3D(x=detect.depth - 0.25)
            carla.Transform(
                carla.Location(),
                carla.Rotation(
                    pitch=current_rot.pitch + alt,
                    yaw=current_rot.yaw + azi,
                    roll=current_rot.roll)).transform(fw_vec)

            def clamp(min_v, max_v, value):
                return max(min_v, min(value, max_v))

            norm_velocity = detect.velocity / self.velocity_range # range [-1, 1]
            r = int(clamp(0.0, 1.0, 1.0 - norm_velocity) * 255.0)
            g = int(clamp(0.0, 1.0, 1.0 - abs(norm_velocity)) * 255.0)
            b = int(abs(clamp(- 1.0, 0.0, - 1.0 - norm_velocity)) * 255.0)
            self.debug.draw_point(
                radar_data.transform.location + fw_vec,
                size=0.075,
                life_time=0.06,
                persistent_lines=False,
                color=carla.Color(r, g, b))

def run_simulation(args, client):
    """
    This function performed one test run using the args parameters
    and connecting to the carla client passed.
    """

    display_manager = None
    vehicle = None
    vehicle_list = []
    timer = CustomTimer()

    try:
        # Getting the world and
        world = client.get_world()
        original_settings = world.get_settings()

        if args.sync:
            traffic_manager = client.get_trafficmanager(8000)
            settings = world.get_settings()
            traffic_manager.set_synchronous_mode(True)
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            world.apply_settings(settings)


        # Instanciating the vehicle to which we attached the sensors
        bp = world.get_blueprint_library().filter('charger_2020')[0]
        vehicle = world.spawn_actor(bp, random.choice(world.get_map().get_spawn_points()))
        vehicle_list.append(vehicle)
        vehicle.set_autopilot(True)

        # Display Manager organize all the sensors an its display in a window
        # If can easily configure the grid and the total window size
        display_manager = DisplayManager(grid_size=[2, 3], window_size=[args.width, args.height])
        sensor_list = []
        camera1 = CameraManager(Cameras['rgb'], vehicle, carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=-90)), 
                      display_manager,  display_pos=[0, 0])
        sensor_list.append(camera1.sensor)
        camera2 = CameraManager(Cameras['rgb'], vehicle, carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=0)), 
                      display_manager,  display_pos=[0, 1])
        sensor_list.append(camera2.sensor)
        camera3 = CameraManager(Cameras['rgb'], vehicle, carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=90)), 
                      display_manager,  display_pos=[0, 2])
        sensor_list.append(camera3.sensor)
        camera4 = CameraManager(Cameras['rgb'], vehicle, carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=180)), 
                      display_manager,  display_pos=[1, 1])
        sensor_list.append(camera4.sensor)
        # camera5 = CameraManager(Cameras['lidar'], vehicle, carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=0)), 
        #               display_manager,  display_pos=[1, 0])
        # sensor_list.append(camera5.sensor)
        camera6 = CameraManager(Cameras['semantic2'], vehicle,carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(yaw=0)), 
                      display_manager,  display_pos=[1, 2])
        sensor_list.append(camera6.sensor)

        # collisin = CollisionSensor(vehicle)
        # sensor_list.append(collisin.sensor)
        # laneinvasion = LaneInvasionSensor(vehicle)
        # sensor_list.append(laneinvasion.sensor)
        # gnss = GnssSensor(vehicle)
        # sensor_list.append(gnss.sensor)
        # imu = IMUSensor(vehicle)
        # sensor_list.append(imu.sensor)
        # radar = RadarSensor(vehicle)
        # sensor_list.append(radar.sensor)

        #Simulation loop
        call_exit = False
        time_init_sim = timer.time()
        while True:
            # Carla Tick
            if args.sync:
                world.tick()
            else:
                world.wait_for_tick()

            # act = carla.VehicleControl(
            #     throttle=float(2.0),
            #     steer=float(0.0),
            #     brake=float(0.0))
            # vehicle.apply_control(act)
            # Render received data
            display_manager.render()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    call_exit = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == K_ESCAPE or event.key == K_q:
                        call_exit = True
                        break

            if call_exit:
                break

    finally:
        for s in sensor_list:
            s.stop()
            s.destroy()
        for v in vehicle_list:
            v.destroy()


        client.apply_batch([carla.command.DestroyActor(x) for x in vehicle_list])
        world.apply_settings(original_settings)

def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Sensor tutorial')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    argparser.add_argument(
        '--async',
        dest='sync',
        action='store_false',
        help='Asynchronous mode execution')
    argparser.set_defaults(sync=True)
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(5.0)

        run_simulation(args, client)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    print(__name__)
    main()

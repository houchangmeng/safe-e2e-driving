#!/usr/bin/env python

# Copyright (c) 2023: Changmeng Hou (houchangmeng@gmail.edu).
#
# This file is modified from <https://github.com/cjy1992/gym-carla>:
# Copyright (c) 2019:
# author: Jianyu Chen (jianyuchen@berkeley.edu)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from __future__ import division

import copy
import numpy as np
import random
import time
import gym
import carla

from gym import spaces
from gym_carla.envs.misc import *
from gym_carla.envs.sensors_manger import DisplayManager, CameraManager, Cameras, CollisionSensor
from gym_carla.envs.bev_render import BirdeyeRender
from gym_carla.envs.global_route_planner import GlobalRoutePlanner
from gym_carla.envs.local_route_planner import LocalRoutePlanner
from gym_carla.envs.controller import VehiclePIDController,args_lat_city_dict,args_long_city_dict

class CarlaEnv(gym.Env):
    """
    An OpenAI gym wrapper for CARLA simulator.
    """
    def __init__(self, params):
        super().__init__()
        # Parameters.
        self.dt = params['dt']
        self.port = params['port']
        self.town = params['town']
        self.max_time_episode = params['max_time_episode']
        self.max_past_step = params['max_past_step']
        self.obs_range = params['obs_range']
        self.d_behind = params['d_behind']
        self.obs_size = params['obs_size']
        self.grid_size = params['grid_size']
        self.desired_speed = params['desired_speed']
        self.max_ego_spawn_times = params['max_ego_spawn_times']
        self.ego_bp_filter = params['ego_vehicle_filter']
        self.number_of_walkers = params['number_of_walkers']
        self.number_of_vehicles = params['number_of_vehicles']
        self.distances_from_start = params['distances_from_start']
        self.enable_display = params['enable_display']
        self.start = params['start']
        self.destination = params['destination']

        # action and observation space
        self.action_space = spaces.Box(
            low=np.array([params['continuous_accel_range'][0],  params['continuous_steer_range'][0]]),
            high=np.array([params['continuous_accel_range'][1], params['continuous_steer_range'][1]]),
            dtype=np.float32)  # acc, steer brake
            
        observation_space_dict = {
        'camera': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
        'lidar': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
        'birdeye': spaces.Box(low=0, high=255, shape=(self.obs_size, self.obs_size, 3), dtype=np.uint8),
        'state': spaces.Box(low =-50.0,high= 50.0,shape=(4,),dtype=np.float32)
        }
        self.observation_space = spaces.Dict(observation_space_dict)

        self.client = None
        self.world = None
        self.map = None
        self.ego = None

        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.radar_sensor = None
        self.camera_sensor = None
        
        # Connect to carla server and get client, world ,map
        self._make_carla_client('localhost', self.port)

        # Load routes
        self.path = []
        self.local_route_planer = None
        self.global_route_planer = GlobalRoutePlanner(
            self.map, sampling_resolution=1.0)
            
        # Controller
        self.controller = None

        # Get spawn points
        self.vehicle_spawn_points = list(self.map.get_spawn_points())
        self.walker_spawn_points = []
        for _ in range(self.number_of_walkers):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                self.walker_spawn_points.append(spawn_point)

        # Set fixed simulation step for synchronous mode
        self.settings = self.world.get_settings()
        self.settings.fixed_delta_seconds = self.dt

        # Record the time of total steps and resetting steps
        self.time_step = 0
        self.reset_step = 0

        # Action info update
        self.last_action = np.array([0.0, 0.0])
        self.current_action = np.array([0.0, 0.0])

        # A dict used for storing info
        self.info = {}

        # A list stores the ids for each episode
        self.sensor_list = []
        self.vehicle_list = []
        self.walker_list = []
        self.walker_ai_list = []

        # Display Manger
        self._init_display()

        # Traffic Manager
        self.traffic_manager = self.client.get_trafficmanager(port=8000)

        # Spectator
        self.spectator = self.world.get_spectator()
    
    def _init_display(self):
        """
        Initialize display manager.
        """
        window_size = [self.obs_size*self.grid_size[1], self.obs_size * self.grid_size[0]]
        self.display_manager = DisplayManager(
            grid_size=self.grid_size, 
            window_size=window_size,
            enable= self.enable_display)
        pixels_per_meter = self.obs_size / self.obs_range
        pixels_ahead_vehicle = (self.obs_range / 2 - self.d_behind) * pixels_per_meter
        self.birdeye_params = {
            'screen_size': self.obs_size,
            'pixels_per_meter': pixels_per_meter,  
            'pixels_ahead_vehicle': pixels_ahead_vehicle,
            'max_past_step':self.max_past_step}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        while True:
            #try:
                # Delete NPC
                self._clear_all_actors()
                self.sensor_list.clear()
                self.vehicle_list.clear()
                self.walker_list.clear()
                self.walker_ai_list.clear()

                # Clear local route planer
                self.local_route_planer = None
                self.target_waypoint = None
                self.controller = None

                # Clear sensor objects
                self.ego = None
                self.collision_sensor = None
                self.camera_left = None
                self.camera_mid = None
                self.camera_right = None
                self.lidar = None

                # Disable sync mode
                self._set_synchronous_mode(False)

                # Action info update
                self.last_action = np.array([0.0, 0.0])
                self.current_action = np.array([0.0, 0.0])

                self.time_step = 0
                # Traffic Manager
                self._set_traffic_manager()

                # Spawn the ego vehicle 
    
                ego_spawn_times = 0
                while True:
                    self.start_transform = vec2carla_transform(self.start)
                    #self.start_transform  = random.choice(self.vehicle_spawn_points)
                    if ego_spawn_times > self.max_ego_spawn_times:
                        self.reset()
                    if self._try_spawn_ego_vehicle_at(self.start_transform):
                        break
                    else:
                        print("Spawn vehicle Again!")
                        ego_spawn_times += 1
                        time.sleep(1)

                # Spawn surrounding vehicles
                random.shuffle(self.vehicle_spawn_points)
                count = min(self.number_of_vehicles, len(self.vehicle_spawn_points)) - 1
                if count > 0:
                    # count = random.randint(0, count)
                    for spawn_point in self.vehicle_spawn_points:
                        if spawn_point.location.distance(self.start_transform.location) < 100:
                            if self._try_spawn_random_vehicle_at(spawn_point, number_of_wheels=[4]):
                                count -= 1
                            if count <= 0:
                                break
                while count > 0:
                    if self._try_spawn_random_vehicle_at(
                        random.choice(self.vehicle_spawn_points), number_of_wheels=[4]):
                        count -= 1

                # Spawn pedestrians
                random.shuffle(self.walker_spawn_points)
                count = self.number_of_walkers
                if count > 0:
                    for spawn_point in self.walker_spawn_points:
                        if self._try_spawn_random_walker_at(spawn_point):
                            count -= 1
                        if count <= 0:
                            break
                while count > 0:
                    if self._try_spawn_random_walker_at(random.choice(self.walker_spawn_points)):
                        count -= 1
                    
                # Reset Display_manager
                self.display_manager.clear()

                # Add sensors
                self.collision_sensor = CollisionSensor(self.ego)
                self.sensor_list.append(self.collision_sensor.sensor)

                # Add Road map
                self.global_birdeye_render = BirdeyeRender(
                    self.world,
                    self.birdeye_params, 
                    self.display_manager,
                    display_pos=[1, 0],
                    global_view = True)

                self.local_birdeye_render = BirdeyeRender(
                    self.world,
                    self.birdeye_params,
                    self.display_manager,
                    display_pos=[1, 2])

                self.global_birdeye_render.set_hero(self.ego)
                self.local_birdeye_render.set_hero(self.ego)
                
                # Add RGB sensor
                self.camera_left = CameraManager(
                    Cameras['rgb'], 
                    self.ego, carla.Transform(
                        carla.Location(x=0, z=2.1), carla.Rotation(yaw=-90)), 
                    self.display_manager,[0, 0])
                self.sensor_list.append(self.camera_left.sensor)

                self.camera_mid = CameraManager(
                    Cameras['rgb'],
                    self.ego, carla.Transform(
                        carla.Location(x=0, z=2.1), carla.Rotation(yaw=-00)), 
                    self.display_manager,[0, 1])
                self.sensor_list.append(self.camera_mid.sensor)

                self.camera_right = CameraManager(
                    Cameras['rgb'], 
                    self.ego, carla.Transform(
                        carla.Location(x=0, z=2.1), carla.Rotation(yaw=+90)), 
                    self.display_manager,[0, 2])
                self.sensor_list.append(self.camera_right.sensor)

                # Add Lidar
                self.lidar = CameraManager(
                    Cameras['lidar'],
                    self.ego, carla.Transform(
                        carla.Location(x=0, z=2.1)),
                    self.display_manager, [1, 1])
                self.sensor_list.append(self.lidar.sensor)

                # Set destination
                start_transform = self.ego.get_transform()
    
                self.dest_transform = vec2carla_transform(self.destination)
                while True:
                    self.dest = random.choice(self.vehicle_spawn_points)
                    self.dest = self.dest_transform
                    if start_transform.location.distance(self.dest.location) > self.distances_from_start:
                        break

                # route_planer
                route = self.global_route_planer.trace_route(
                    start_transform.location, self.dest.location)
                
                self.global_waypoints = []
                self.path = []
                for i,(w, r) in enumerate(route):
                    self.global_waypoints.append(w)
                    self.path.append(w.transform.location)
                self.global_birdeye_render.waypoints = self.global_waypoints

                self.local_route_planer = LocalRoutePlanner(self.ego,buffer_size=40)
                self.local_route_planer.set_global_plan(route)
                self.local_waypoints, _, self.front_vehicle, _  = self.local_route_planer.run_step()

                self.current_waypoint = self.map.get_waypoint(self.ego.get_location())
                self.update_target = False
                self.target_waypoint = self.local_waypoints[0]

                self.local_birdeye_render.waypoints = self.local_waypoints

                """
                #================================================================#
                #==================PID Controller for Collecting Dataset=========#
                #================================================================#
                #
                # self.controller = VehiclePIDController(
                #     vehicle = self.ego,
                #     args_lateral=args_lat_city_dict,
                #     args_longitudinal=args_long_city_dict)
                #
                #================================================================#
                """
                self.controller = VehiclePIDController(
                    vehicle = self.ego,
                    args_lateral=args_lat_city_dict,
                    args_longitudinal=args_long_city_dict)
                
                # Update timesteps
                self.time_step = 1
                self.reset_step += 1

                # Enable sync mode
                self._set_synchronous_mode(True)
                
                self.isCollided = False
                self.isTimeOut = False
                self.isSuccess = False
                self.isOutOfLane = False
                self.isOutOfRoute = False
                self.isSpecialSpeed = False
                
                #self._collect_data()
                # self.world.tick()
                # local birdeye render first in order to have bev data
                
                self.local_birdeye_render.render()
                self.lidar.birdeye = self.local_birdeye_render.data
                self.display_manager.render()
                
                obs = self._get_obs()
                self.info.update(self._get_cost())
                
                self._set_spectator()
                
                return obs, copy.deepcopy(self.info)
            # except:
            #     print("Env reset error!")
            #     time.sleep(2)
            #     self._make_carla_client('localhost', self.port)
                
    def _make_carla_client(self,host,port):
        """
        Get a random carla position on the line between start and dest.
        """
        while True:
            try:
                print("Connecting to Carla server...")
                self.client = carla.Client(host, port)
                self.client.set_timeout(10.0)
                # Set map
                self.world = self.client.load_world(self.town)
                self.map = self.world.get_map()
                ## Set weather
                # self.world.set_weather(carla.WeatherParameters.ClearNoon)
                break
            except Exception:
                print("Fail to connect to carla-server...sleeping for 2")
                time.sleep(2)
        print("=====Connecting sucess !!=====")
    
    def _set_traffic_manager(self):
        """
        Set up traffic manager.
        """
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.0)
        self.traffic_manager.set_respawn_dormant_vehicles(True)
        self.traffic_manager.set_random_device_seed(0)

    def _create_vehicle_bluepprint(self,
                                   actor_filter,
                                   color=None,
                                   number_of_wheels=[4]):
        """
        Create the blueprint for a specific actor type.
        Args:
            actor_filter: a string indicating the actor type, e.g, 'vehicle.lincoln*'.
        Returns:
            bp: the blueprint object of carla.
        """
        blueprints = self.world.get_blueprint_library().filter(actor_filter)
        blueprint_library = []
        for nw in number_of_wheels:
            blueprint_library = blueprint_library + [
                x for x in blueprints
                if int(x.get_attribute('number_of_wheels')) == nw
            ]
        bp = random.choice(blueprint_library)
        if bp.has_attribute('color'):
            if not color:
                color = random.choice(
                    bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        return bp


    def _set_synchronous_mode(self, synchronous=True):
        """
        Set whether to use the synchronous mode.
        """
        self.settings.synchronous_mode = synchronous
        self.world.apply_settings(self.settings)
    
    def _set_spectator(self):
        """
        Set spectator view.
        """
        transform = self.ego.get_transform()
        self.spectator.set_transform(
            carla.Transform(transform.location + 
            carla.Location(z=50),
            carla.Rotation(yaw=180,pitch=-90)))

    def _try_spawn_ego_vehicle_at(self, transform):
        """
        Try to spawn the ego vehicle at specific transform.
        Args:
            transform: the carla transform object.
        Returns:
            Bool indicating whether the spawn is successful.
        """
        blueprint = self._create_vehicle_bluepprint(
            self.ego_bp_filter, color='49,4,4')
        blueprint.set_attribute('role_name', 'hero')
        vehicle = self.world.try_spawn_actor(blueprint, transform)
        #self.traffic_manager.ignore_lights_percentage(vehicle, 100)
        if vehicle is not None:
            self.ego = vehicle
            self.vehicle_list.append(vehicle)
            return True
        return False
    
    def _try_spawn_random_vehicle_at(self, transform, number_of_wheels=[4]):
        """
        Try to spawn a surrounding vehicle at specific transform with random bluprint.
        Args:
        transform: the carla transform object.
        Returns:
        Bool indicating whether the spawn is successful.
        """
        blueprint = self._create_vehicle_bluepprint('vehicle.*', number_of_wheels=number_of_wheels)
        blueprint.set_attribute('role_name', 'autopilot')
        vehicle = self.world.try_spawn_actor(blueprint, transform)
        if vehicle is not None:
            vehicle.set_autopilot(True,self.traffic_manager.get_port())
            self.traffic_manager.ignore_lights_percentage(vehicle, 80)
            self.vehicle_list.append(vehicle)
            return True
        return False
    
    def _try_spawn_random_walker_at(self, transform):
        """
        Try to spawn a walker at specific transform with random bluprint.

        Args:
        transform: the carla transform object.

        Returns:
        Bool indicating whether the spawn is successful.
        """
        walker_bp = random.choice(self.world.get_blueprint_library().filter('walker.*'))
        # set as not invencible
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'False')
        walker_actor = self.world.try_spawn_actor(walker_bp, transform)
        
        if walker_actor is not None:
            walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
            walker_controller_actor = self.world.spawn_actor(walker_controller_bp, carla.Transform(), walker_actor)
            # start walker
            walker_controller_actor.start()
            # set walk to random point
            walker_controller_actor.go_to_location(self.world.get_random_location_from_navigation())
            # random max speed
            walker_controller_actor.set_max_speed(1 + random.random())    # max speed between 1 and 2 (default is 1.4 m/s)

            self.walker_list.append(walker_actor)
            self.walker_ai_list.append(walker_controller_actor)
            return True
        return False

    def step(self,action):
        try:
            
            self.update_target = False
            self.local_waypoints, red_light, self.front_vehicle , walker_state \
                = self.local_route_planer.run_step()
            if self.local_waypoints[0] != self.target_waypoint:
                self.target_waypoint = self.local_waypoints[0]
                self.update_target = True
            self.current_waypoint = self.map.get_waypoint(self.ego.get_location())
            
            
            #==================================================================#
            #==================PID Controller for Collecting Dataset===========#
            #==================================================================#
            # control = self.controller.run_step(self.desired_speed , 
            #                     self.target_waypoint)
            # if np.random.uniform() > 0.8:
            #     control.throttle = np.random.uniform()
            #     control.steer = np.clip((np.random.uniform() - 0.5) * 2, -1, 1)
            # # if red_light or self.front_vehicle or walker_state:
            # #     control.steer = 0.0
            # #     control.throttle = 0.0
            # #     control.brake = 1.0
            # #     control.hand_brake = False
            # act = control
            #==================================================================#
            
            
            #==================================================================#
            #============================RL Agent==============================#
            #==================================================================#
            
            self.last_action = self.current_action
            current_action = np.array(action) 
            self.current_action = current_action
            throttle , steer  = current_action

            if throttle > 0:
                throttle = np.clip(throttle /3 , 0, 1)
                brake = 0
            else:
                throttle = 0
                brake = np.clip(-throttle /8, 0, 1)

            act = carla.VehicleControl(
                throttle=float(throttle),
                steer=float(steer),
                brake=float(brake))
            #==================================================================#

            self.ego.apply_control(act)
        
            self.local_birdeye_render.waypoints = self.local_waypoints

            # Render
            self.local_birdeye_render.render()
            self.lidar.birdeye = self.local_birdeye_render.data
            self.display_manager.render()

            # Calculate return
            #self._collect_data()
            obs = self._get_obs()
            isDone = self._terminal()
            current_reward = self._get_reward()
            self.info.update(self._get_cost())
            
            self.time_step += 1

            self.world.tick()
            self._set_spectator()

            return (obs, current_reward, isDone, False, copy.deepcopy(self.info))
        except:
            print('Env step() error')
            time.sleep(2)
            return (self._get_obs(), 0.0, True, True, copy.deepcopy(self.info))
    
    def _get_obs(self):
        camera_image = self.camera_mid.data
        lidar_image = self.lidar.data
        bev_image = self.local_birdeye_render.data
        lidar_image[:,:,2] = (bev_image[:,:,0] <= 10) * (bev_image[:,:,1] <= 10) * (bev_image[:,:,2] >= 240) * 255
        instruction = self.local_route_planer._target_road_option
        high_level_command  = command2vector(instruction)
        state = self._get_state()
        obs = {
            'camera':camera_image.astype(np.uint8),
            'lidar':lidar_image.astype(np.uint8),
            'birdeye':bev_image.astype(np.uint8),
            'state': state.astype(np.float32)
            }
        
        return obs
    
    def _collect_data(self):
        """
        Collect dataset.
        """

        current_transform = carla_transform2vec(self.ego.get_transform())
        vehicle_control = carla_control2vec(self.ego.get_control())
        state = self._get_state()
        camera_image_m = self.camera_mid.data  # np.unit 8 
        camera_image_l = self.camera_left.data  # np.unit 8 
        camera_image_r = self.camera_right.data  # np.unit 8 
        lidar_image  = self.lidar.data  # np.unit 8 
        bev_image = self.local_birdeye_render.data  # np.unit 8 

        # self.time_step
        self.info = {
            'current_transform':current_transform,
            'vehicle_control':vehicle_control,
            'state':state, 
            'camera_image_m':camera_image_m,
            'camera_image_l':camera_image_l,
            'camera_image_r':camera_image_r,
            'lidar_image':lidar_image,
            'bev_image':bev_image
        }
        # draw_waypoints(self.world,[target_waypoint])

    def _get_state(self):
        """
        Calculate current state.
        """
        ego_trans = self.ego.get_transform()
        ego_x = ego_trans.location.x
        ego_y = ego_trans.location.y
        ego_yaw = ego_trans.rotation.yaw/180*np.pi
        lat_dis, w = get_preview_lane_dis(self.target_waypoint, ego_x, ego_y)
        delta_yaw = np.arcsin(np.cross(w, 
            np.array(np.array([np.cos(ego_yaw), np.sin(ego_yaw)]))))
        speed = get_speed(self.ego) / 3.6
        acc = get_acceleration(self.ego)
        state = np.array([lat_dis, -delta_yaw , speed , self.front_vehicle],dtype=np.float32)

        return state

    def render(self, mode='human'):
        pass

    def close(self):
        """
        Close environment.
        """
        self._clear_all_actors()
        self.sensor_list.clear()
        self.vehicle_list.clear()
        self.walker_list.clear()
        self.walker_ai_list.clear()

        self.ego = None
        self.collision_sensor = None
        self.camera_left = None
        self.camera_mid = None
        self.camera_right = None

        if self.settings:
            self._set_synchronous_mode(False)
        return True
    

    def _clear_all_actors(self):
        """
        Clear all created actors.
        """
        for s in  self.sensor_list:
            if (s is not None) and s.is_alive:
                s.stop()
                s.destroy()
                s = None

        for w_ai in  self.walker_ai_list:
            if (w_ai is not None) and w_ai.is_alive:
                w_ai.stop()
                w_ai.destroy()
                w_ai = None

        for v in  self.vehicle_list:
            if (v is not None) and v.is_alive:
                v.destroy()
                v = None
                del v

        for w in  self.walker_list:
            if (w is not None) and w.is_alive:
                w.destroy()
                w = None
 
    
    def _terminal(self):
        """
        Calculate whether to terminate the current episode.
        """
        # If collides
        if len(self.collision_sensor.history) > 0:
            print('Collision happened! Episode %d cost %d steps .' %(self.reset_step, self.time_step))
            self.isCollided = True
            return True
        
        #If at destination
        if self.dest is not None:
            if self.ego.get_location().distance(self.dest.location) < 5.0  :
                print('Get destination! Episode %d cost %d steps in route .' %(self.reset_step, self.time_step))
                self.isSuccess = True
                return True
            
        # If reach maximum timestep
        if self.time_step >= self.max_time_episode:
            print('Time out! Episode %d cost %d steps .'  %(self.reset_step, self.time_step))
            self.isTimeOut = True
            return True
        

        return False
    
    def _get_reward(self):
        """
        Calculate the reward of current state
        Args:
            
        Returns:
            Reward
                Update Reward.
                Step Reward.
        """
        
        v = self.ego.get_velocity()
       
        ego_x, ego_y = get_pos(self.ego)
        _ ,w = get_lane_dis(self.local_waypoints, ego_x, ego_y)
 
        # longitudinal speed
        r_speed_lon = 0.0
        lspeed = np.array([v.x, v.y])
        lspeed_lon = np.dot(lspeed, w)
        r_speed_lon = lspeed_lon
        r_speed_lon = 0.0

        # cost for collision
        r_collision = 0.0
        if len(self.collision_sensor.history) > 0:
            r_collision = -100.0
        
        # cost for path tracking
        ego_x, ego_y = get_pos(self.ego)
        lat_dis ,w = get_lane_dis(self.local_waypoints, ego_x, ego_y)
        r_out = 0.0
        if abs(lat_dis) > self.target_waypoint.lane_width * 1.5:
            r_out = -1.0
        
        # cost for velocity
        v = self.ego.get_velocity()
        lspeed = np.array([v.x, v.y])
        lspeed_lon = np.dot(lspeed, w)

        r_fast = 0.0
        if lspeed_lon > self.desired_speed + 1:
            r_fast = -10
        
        # cost for lateral acceleration
        r_lat = 0.0
        if abs(self.ego.get_control().steer) * lspeed_lon**2 > 4:
            r_lat = -1.0

        # cost for steer
        r_steer = 0.0
        if self.ego.get_control().steer**2 > 0.1:
            r_steer = -self.ego.get_control().steer**2

        # reward for update target waypoint
        r_update = 0
        if self.update_target:
            r_update = 1

        r_step = -0.1 

        r = 1.0 * r_update + r_step + r_collision + r_speed_lon + r_fast + r_out +  r_steer + r_lat

        #r = 1.0 * r_update + r_step 

        return r

    def _get_cost(self):
        """
        Calculate the cost of current state.
        Args:
            
        Returns:
            Cost
                Collision cost.
                Path tracking cost.
                Velocity cost.
                Lateral acceleration cost.
        """
        cost = {}

        # cost for collision
        cost["cost_collision"] = 0.0
        if len(self.collision_sensor.history) > 0:
            cost["cost_collision"] = 100.0
            self.collision_sensor.history.clear()
        
        # cost for path tracking
        ego_x, ego_y = get_pos(self.ego)
        lat_dis ,w = get_lane_dis(self.local_waypoints, ego_x, ego_y)
        cost["cost_out"] = 0.0
        if abs(lat_dis) > self.target_waypoint.lane_width * 1.5:
            cost["cost_out"] = 1.0
        
        # cost for velocity
        v = self.ego.get_velocity()
        lspeed = np.array([v.x, v.y])
        lspeed_lon = np.dot(lspeed, w)

        cost["cost_fast"] = 0.0
        if lspeed_lon > self.desired_speed + 1:
            cost["cost_fast"] = 10
        
        # cost for lateral acceleration
        cost["cost_lat"] = 0.0
        if abs(self.ego.get_control().steer) * lspeed_lon**2 > 4:
            cost["cost_lat"] = 1.0

        # cost for steer
        cost["cost_steer"] = 0.0
        if self.ego.get_control().steer**2 > 0.1:
            cost["cost_steer"] = self.ego.get_control().steer**2

        cost['cost'] = sum(v for k, v in cost.items() if k.startswith('cost_'))

        return cost
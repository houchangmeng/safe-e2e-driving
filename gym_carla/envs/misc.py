import math
import numpy as np
import carla
import skimage

command = [
        'ChangeLaneRight',
        'ChangeLaneLeft',
        'LaneFollow',
        'Right', 
        'Left',
        'Straight']

def command2vector(commandstr):
    """
    Convert commandstrc('str') to vector to be used in FC-net
    Args: 
        command str
    Return: 
        command vector(np.array, 5*1) [1 0 0 0 0 0]
        0 - 'LaneFollow',
        1 - 'Right', 
        2 - 'Left',
        3 - 'Straight',
        4 - 'ChangeLaneRight',
        5 - 'ChangeLaneLeft',   
    """
    command_vec = np.zeros(6, dtype='uint8')
    idx = command.index(commandstr)
    command_vec[idx] = 1.0
    return command_vec


def vec_decompose(vec_to_be_decomposed, direction):
    """
    Decompose the vector along the direction vec
    
    Args:
        vec_to_be_decomposed: np.array, shape:(2,)
        direction: np.array, shape:(2,); |direc| = 1
    Returns:
        vec_longitudinal
        vec_lateral
            both with sign
    """
    assert vec_to_be_decomposed.shape[0] == 2, direction.shape[0] == 2
    lon_scalar = np.inner(vec_to_be_decomposed, direction)
    lat_vec = vec_to_be_decomposed - lon_scalar * direction
    lat_scalar = np.linalg.norm(lat_vec) * np.sign(lat_vec[0] * direction[1] -
                                                   lat_vec[1] * direction[0])
    return np.array([lon_scalar, lat_scalar], dtype=np.float32)


def delta_angle_between(theta_1, theta_2):
    """
    Compute the delta angle between theta_1 & theta_2(both in degree)

    Args:
        theta: float, target: theta_2, ref theta_1
    Returns:
        delta_theta: float, in [-pi, pi]
    """
    theta_1 = theta_1 % 360
    theta_2 = theta_2 % 360
    delta_theta = theta_2 - theta_1
    if 180 <= delta_theta and delta_theta <= 360:
        delta_theta -= 360
    elif -360 <= delta_theta and delta_theta <= -180:
        delta_theta += 360
    return delta_theta

def get_speed(vehicle):
    """
    Compute speed of a vehicle in Kmh
    Args:
        vehicle: the vehicle for which speed is calculated
    Return: 
        speed as a float in Kmh
    """
    vel = vehicle.get_velocity()
    return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

def get_acceleration(vehicle):
    """
    Compute acceleration of a vehicle in Kmh
    Args:
        vehicle: the vehicle for which speed is calculated
    Return: 
        acceleration as a float in m/s^2
    """
    a = vehicle.get_acceleration()
    return np.sqrt(a.x**2 + a.y**2)

def get_angular_velocity(vehicle):
    """
    Compute speed of a vehicle in Kmh
    Args:
        vehicle: the vehicle for which speed is calculated
    Return: 
        speed as a float in Kmh
    """
    pass

def get_control(vehicle):
    """
    Compute speed of a vehicle in Kmh
    Args:
        vehicle: the vehicle for which speed is calculated
    Return: 
        speed as a float in Kmh
    """
    pass


def get_pos(vehicle):
    """
    Get the position of a vehicle
    Args:
        vehicle: the vehicle whose position is to get
    Return:
        speed as a float in Kmh
    """
    trans = vehicle.get_transform()
    x = trans.location.x
    y = trans.location.y
    return x, y

def get_info(vehicle):
    """
    Get the full info of a vehicle
    Args:
        vehicle: the vehicle whose info is to get
    Return:
        a tuple of x, y positon, yaw angle and half length, width of the vehicle
    """
    trans = vehicle.get_transform()
    x = trans.location.x
    y = trans.location.y
    yaw = trans.rotation.yaw / 180 * np.pi
    bb = vehicle.bounding_box
    l = bb.extent.x
    w = bb.extent.y
    info = (x, y, yaw, l, w)
    return info

def waypoint2xyz(waypoint):
    """
    Convert waypoint to [x,y,yaw]
    Args:
        waypoint: carla.waypoint
    Return:
        list [x, y, yaw]
    """
    return [
        waypoint.transform.location.x,
        waypoint.transform.location.y,
        waypoint.transform.rotation.yaw]

def get_preview_lane_dis(waypoints, x, y):
    """
    Calculate distance from (x, y) to a certain waypoint
    Args:
        waypoints: a certain waypoint type: carla.waypoint
        x: x position of vehicle
        y: y position of vehicle
    :param idx: index of the waypoint to which the distance is calculated
    :return: a tuple of the distance and the waypoint orientation
    """
    waypt = waypoint2xyz(waypoints)
    vec = np.array([x - waypt[0], y - waypt[1]])
    lv = np.linalg.norm(np.array(vec))
    w = np.array([np.cos(waypt[2]/180*np.pi), np.sin(waypt[2]/180*np.pi)])
    cross = np.cross(w, vec/lv)
    dis = - lv * cross
    return dis, w

def get_lane_dis(waypoints, x, y):
    """
    Calculate distance from (x, y) to waypoints.
    Args:
        waypoints: a list of list storing waypoints like [carla.waypoint, carla.waypoint, ...]
        x: x position of vehicle
        y: y position of vehicle
    Return: 
        a tuple of the lateral distance and the '##CLOSEST## waypoint orientation
    """
    dis_min = 1000
    waypt = waypoint2xyz(waypoints[0])
    for pt in waypoints:
        pt = waypoint2xyz(pt)
        d = np.sqrt((x-pt[0])**2 + (y-pt[1])**2)
        if d < dis_min:
            dis_min = d
            waypt=pt
    vec = np.array([x - waypt[0], y - waypt[1]])
    lv = np.linalg.norm(np.array(vec))
    w = np.array([np.cos(waypt[2]/180*np.pi), np.sin(waypt[2]/180*np.pi)])
    cross = np.cross(w, vec/lv)
    dis = - lv * cross
    return dis, w

def vector(location_1, location_2):
    """
    Returns the unit vector from location_1 to location_2
    Args:
        location_1, location_2: carla.Location objects
    """
    x = location_2.x - location_1.x
    y = location_2.y - location_1.y
    z = location_2.z - location_1.z
    norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps

    return [x / norm, y / norm, z / norm]

def compute_distance(location_1, location_2):
    """
    Euclidean distance between 3D points
    Args:
        location: carla.Location 
    Returns:
        norm: norm distance
    """
    x = location_2.x - location_1.x
    y = location_2.y - location_1.y
    z = location_2.z - location_1.z
    norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps
    return norm

def vec2carla_transform(pose):
    """Get a carla tranform object given pose.
    Args:
        pose: [x, y, z, pitch, roll, yaw].
    Returns:
        transform: the carla transform object
    """
    transform = carla.Transform()
    transform.location.x = pose[0]
    transform.location.y = pose[1]
    transform.location.z = pose[2]
    transform.rotation.pitch = pose[3]
    transform.rotation.roll = pose[4]
    transform.rotation.yaw = pose[5]
    return transform
    
def carla_transform2vec(transform):
    """Get a pose given carla tranform object.
    Args:
        transform: the carla transform object
    Returns:
        pose: [x, y, z, pitch, roll, yaw].
    """
    pose = np.zeros(6, dtype=np.float32)
    pose[0] = transform.location.x
    pose[1] = transform.location.y
    pose[2] = transform.location.z
    pose[3] = transform.rotation.pitch
    pose[4] = transform.rotation.roll
    pose[5] = transform.rotation.yaw
    return pose

def carla_control2vec(control):
    """Get a vector given carla tranform object.
    Args:
        control: the carla vehicle.control object
    Returns:
        vec: [acc , steer ].
    """
    
    throttle = control.throttle
    steer = control.steer
    brake = control.brake
    if brake > 0:
        acc = - brake
    else:
        acc = throttle
    return np.array([acc, steer],dtype='float32')

def is_within_distance_ahead(target_location, current_location, orientation, max_distance):
    """
    Check if a target object is within a certain distance in front of a reference object.

    Args:
        target_location: location of the target object
        current_location: location of the reference object
        orientation: orientation of the reference object
        max_distance: maximum allowed distance
    Return: 
        True if target object is within max_distance ahead of the reference object
    """
    target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
    norm_target = np.linalg.norm(target_vector)

    # If the vector is too short, we can simply stop here
    if norm_target < 0.001:
        return True

    if norm_target > max_distance:
        return False

    forward_vector = np.array(
        [math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])
    d_angle = math.degrees(math.acos(np.dot(forward_vector, target_vector) / norm_target))

    return d_angle < 90.0

def distance_vehicle(waypoint, vehicle_transform):
    loc = vehicle_transform.location
    dx = waypoint.transform.location.x - loc.x
    dy = waypoint.transform.location.y - loc.y

    return math.sqrt(dx * dx + dy * dy)

def compute_magnitude_angle(target_location, current_location, orientation):
    """
    Compute relative angle and distance between a target_location and a current_location

    Args:
        target_location: location of the target object
        current_location: location of the reference object
        orientation: orientation of the reference object
    Returns: 
        a tuple composed by the distance to the object and the angle between both objects
    """
    target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
    norm_target = np.linalg.norm(target_vector)

    forward_vector = np.array([math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])
    d_angle = math.degrees(math.acos(np.dot(forward_vector, target_vector) / norm_target))

    return (norm_target, d_angle)

def draw_waypoints(world, waypoints, z=0.5):
    """
    Draw a list of waypoints at a certain height given in z.

    Args:
        world: carla.world object
        waypoints: list or iterable container with the waypoints to draw
        z: height in meters
    """
    for wpt in waypoints:
        wpt_t = wpt.transform
        begin = wpt_t.location + carla.Location(z=z)
        angle = math.radians(wpt_t.rotation.yaw)
        end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
        world.debug.draw_arrow(begin, end, arrow_size=0.3, life_time=1.0)

def display_to_rgb(display, obs_size):
    """
    Transform image grabbed from pygame display to an rgb image uint8 matrix
    Args:
        display: pygame display input
        obs_size: rgb image size
    Returns:
        rgb image uint8 matrix
    """
    rgb = np.fliplr(np.rot90(display, 3))  # flip to regular view
    rgb = skimage.transform.resize(rgb, (obs_size[0], obs_size[1]))  # resize
    rgb = rgb * 255
    return rgb

if __name__ == '__main__':
    print(command2vector('LaneFollow'))


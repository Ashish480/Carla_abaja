import carla
import numpy as np
import time
import pygame
import random
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# PID Controller Class
class PIDController:
    def __init__(self, kp, ki, kd, integral_limit=10):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0.0
        self.integral = 0.0
        self.integral_limit = integral_limit

    def compute(self, error, dt):
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output
    
    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0

# Improved PID Controller for Distance Maintenance
class DistancePIDController:
    def __init__(self, kp, ki, kd, target_distance=8.0, integral_limit=5):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target_distance = target_distance
        self.integral_limit = integral_limit
        self.prev_error = 0.0
        self.integral = 0.0

    def compute(self, current_distance, dt):
        error = self.target_distance - current_distance
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error

        # Apply emergency braking when too close
        if current_distance < 4:
            return -1.0  # Full brake when dangerously close
        
        # Apply stronger braking when close
        if current_distance < 5:
            return -0.8 * (5 - current_distance) / 5  # Progressive braking based on distance
        
        # Apply mild braking when approaching target distance
        if current_distance < self.target_distance:
            return np.clip(output * 0.4, -0.6, 0.2)  # More conservative throttle response when close
            
        return np.clip(output, -1.0, 0.8)  # Clip to reasonable values
    
    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0

def fuzzy_logic_acc(distance_front, speed, target_distance=8.0):
    # Increase universe range for distance for smoother transitions
    distance = ctrl.Antecedent(np.arange(0, 51, 1), 'distance')  # 0 to 50 meters
    velocity = ctrl.Antecedent(np.arange(0, 101, 1), 'velocity')  # 0 to 100 km/h
    throttle = ctrl.Consequent(np.arange(-1.0, 1.1, 0.1), 'throttle')  # -1 to 1 (brake to throttle)

    # Redefine membership functions for better control
    # Critical zones for safety
    distance['critical'] = fuzz.trimf(distance.universe, [0, 0, 3])
    distance['dangerous'] = fuzz.trimf(distance.universe, [1, 3, 5])
    distance['very_close'] = fuzz.trimf(distance.universe, [3, 5, 7])
    distance['close'] = fuzz.trimf(distance.universe, [5, 7, target_distance-0.5])
    distance['target'] = fuzz.trimf(distance.universe, [target_distance-1, target_distance, target_distance+1])
    distance['comfortable'] = fuzz.trimf(distance.universe, [target_distance+0.5, target_distance+3, target_distance+6])
    distance['far'] = fuzz.trimf(distance.universe, [target_distance+5, 20, 50])

    velocity['stopped'] = fuzz.trimf(velocity.universe, [0, 0, 5])
    velocity['very_slow'] = fuzz.trimf(velocity.universe, [0, 10, 20])
    velocity['slow'] = fuzz.trimf(velocity.universe, [15, 25, 40])
    velocity['medium'] = fuzz.trimf(velocity.universe, [30, 50, 70])
    velocity['fast'] = fuzz.trimf(velocity.universe, [60, 80, 100])

    # More granular control outputs
    throttle['emergency_brake'] = fuzz.trimf(throttle.universe, [-1.0, -1.0, -0.8])
    throttle['hard_brake'] = fuzz.trimf(throttle.universe, [-1.0, -0.8, -0.6])
    throttle['medium_brake'] = fuzz.trimf(throttle.universe, [-0.7, -0.5, -0.3])
    throttle['soft_brake'] = fuzz.trimf(throttle.universe, [-0.4, -0.2, 0])
    throttle['coast'] = fuzz.trimf(throttle.universe, [-0.1, 0, 0.1])
    throttle['light_throttle'] = fuzz.trimf(throttle.universe, [0.05, 0.2, 0.35])
    throttle['medium_throttle'] = fuzz.trimf(throttle.universe, [0.3, 0.5, 0.7])
    throttle['full_throttle'] = fuzz.trimf(throttle.universe, [0.6, 0.8, 1.0])

    # Enhanced rule set for better safety and performance
    rules = [
        # Critical safety rules - override everything
        ctrl.Rule(distance['critical'], throttle['emergency_brake']),
        ctrl.Rule(distance['dangerous'], throttle['hard_brake']),
        
        # Very close distance rules
        ctrl.Rule(distance['very_close'] & velocity['stopped'], throttle['medium_brake']),
        ctrl.Rule(distance['very_close'] & velocity['very_slow'], throttle['medium_brake']),
        ctrl.Rule(distance['very_close'] & velocity['slow'], throttle['hard_brake']),
        ctrl.Rule(distance['very_close'] & velocity['medium'], throttle['hard_brake']),
        ctrl.Rule(distance['very_close'] & velocity['fast'], throttle['emergency_brake']),
        
        # Close distance rules
        ctrl.Rule(distance['close'] & velocity['stopped'], throttle['soft_brake']),
        ctrl.Rule(distance['close'] & velocity['very_slow'], throttle['soft_brake']),
        ctrl.Rule(distance['close'] & velocity['slow'], throttle['medium_brake']),
        ctrl.Rule(distance['close'] & velocity['medium'], throttle['medium_brake']),
        ctrl.Rule(distance['close'] & velocity['fast'], throttle['hard_brake']),
        
        # Target distance rules - maintain steady state
        ctrl.Rule(distance['target'] & velocity['stopped'], throttle['coast']),
        ctrl.Rule(distance['target'] & velocity['very_slow'], throttle['light_throttle']),
        ctrl.Rule(distance['target'] & velocity['slow'], throttle['coast']),
        ctrl.Rule(distance['target'] & velocity['medium'], throttle['coast']),
        ctrl.Rule(distance['target'] & velocity['fast'], throttle['soft_brake']),
        
        # Comfortable distance rules
        ctrl.Rule(distance['comfortable'] & velocity['stopped'], throttle['light_throttle']),
        ctrl.Rule(distance['comfortable'] & velocity['very_slow'], throttle['light_throttle']),
        ctrl.Rule(distance['comfortable'] & velocity['slow'], throttle['light_throttle']),
        ctrl.Rule(distance['comfortable'] & velocity['medium'], throttle['light_throttle']),
        ctrl.Rule(distance['comfortable'] & velocity['fast'], throttle['coast']),
        
        # Far distance rules - catch up but don't speed
        ctrl.Rule(distance['far'] & velocity['stopped'], throttle['medium_throttle']),
        ctrl.Rule(distance['far'] & velocity['very_slow'], throttle['medium_throttle']),
        ctrl.Rule(distance['far'] & velocity['slow'], throttle['medium_throttle']),
        ctrl.Rule(distance['far'] & velocity['medium'], throttle['light_throttle']),
        ctrl.Rule(distance['far'] & velocity['fast'], throttle['light_throttle'])
    ]

    acc_ctrl = ctrl.ControlSystem(rules)
    acc = ctrl.ControlSystemSimulation(acc_ctrl)

    # Ensure distance is within the defined universe
    bounded_distance = np.clip(distance_front, 0, 50)
    acc.input['distance'] = bounded_distance
    acc.input['velocity'] = np.clip(speed, 0, 100)

    try:
        acc.compute()
        throttle_value = acc.output['throttle']
        
        # Additional safety checks
        if distance_front < 3:
            return -1.0  # Emergency brake regardless of fuzzy output
        if distance_front < 5 and throttle_value > -0.3:
            return -0.5  # Ensure braking when close, even if fuzzy logic suggests otherwise
            
        return throttle_value
    except Exception as e:
        print(f"Error in fuzzy computation: {e}")
        # Fallback control based on distance
        if distance_front < 3:
            return -1.0  # Emergency brake
        elif distance_front < 5:
            return -0.7  # Strong brake
        elif distance_front < target_distance:
            return -0.2  # Mild brake
        elif distance_front < target_distance + 5:
            return 0.0  # Coast
        else:
            return 0.4  # Moderate throttle

# Lane Waypoints
def get_lane_waypoints(world, distance=2.0):
    return [wp for wp in world.get_map().generate_waypoints(distance) if wp.lane_type == carla.LaneType.Driving]

def find_closest_waypoint(vehicle_location, waypoints):
    closest = min(waypoints, key=lambda wp: vehicle_location.distance(wp.transform.location))
    return closest

# Improved Steering Control with Stanley Controller
def compute_steering_angle(vehicle, waypoints, pid_controller, k=0.5):
    vehicle_transform = vehicle.get_transform()
    vehicle_location = vehicle_transform.location
    vehicle_yaw = np.radians(vehicle_transform.rotation.yaw)

    # Find closest waypoint
    closest_waypoint = find_closest_waypoint(vehicle_location, waypoints)
    if closest_waypoint is None:
        return 0.0

    # Calculate cross-track error
    dx = closest_waypoint.transform.location.x - vehicle_location.x
    dy = closest_waypoint.transform.location.y - vehicle_location.y
    cte = dy * np.cos(vehicle_yaw) - dx * np.sin(vehicle_yaw)

    # Calculate heading error
    path_yaw = np.radians(closest_waypoint.transform.rotation.yaw)
    heading_error = np.arctan2(np.sin(path_yaw - vehicle_yaw), np.cos(path_yaw - vehicle_yaw))

    # Calculate vehicle speed for Stanley controller
    vehicle_velocity = vehicle.get_velocity()
    speed = max(3.6 * np.sqrt(vehicle_velocity.x**2 + vehicle_velocity.y**2 + vehicle_velocity.z**2), 1.0)

    # Stanley steering calculation
    stanley_term = np.arctan(k * cte / speed)
    
    # Combine heading error and stanley term
    steering_angle = heading_error + stanley_term
    
    # Apply PID correction for smoother steering
    dt = 0.05  # Fixed time step from simulation
    pid_correction = pid_controller.compute(cte, dt)
    
    # Limit PID influence for stability
    pid_correction = np.clip(pid_correction, -np.radians(10), np.radians(10))
    
    # Combine and limit final steering
    final_steering = np.clip(steering_angle + pid_correction, -np.radians(30), np.radians(30))
    
    return final_steering

# Improved Process Camera Image
def process_img(image, display, vehicle, steering_angle, font, distance_front=100, target_distance=8.0, throttle_value=0.0):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))[:, :, :3][:, :, ::-1]
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    
    velocity = vehicle.get_velocity()
    speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

    # Create text surfaces with enhanced information
    text_surface_speed = font.render(f'Speed: {speed:.2f} km/h', True, (255, 255, 255))
    text_surface_steering = font.render(f'Steering: {np.degrees(steering_angle):.2f}°', True, (255, 255, 255))
    
    # Change color based on distance
    if distance_front < 5:
        distance_color = (255, 0, 0)  # Red if too close
    elif distance_front < target_distance:
        distance_color = (255, 255, 0)  # Yellow if approaching target
    else:
        distance_color = (0, 255, 0)  # Green if at or beyond target
    
    text_surface_distance = font.render(f'Distance: {distance_front:.2f} m', True, distance_color)
    target_dist_text = font.render(f'Target: {target_distance:.1f} m', True, (255, 255, 255))
    
    # Show throttle/brake status
    if throttle_value < 0:
        control_text = f'Braking: {abs(throttle_value):.2f}'
        control_color = (255, 0, 0)  # Red for braking
    else:
        control_text = f'Throttle: {throttle_value:.2f}'
        control_color = (0, 255, 0)  # Green for throttle
    
    text_surface_control = font.render(control_text, True, control_color)
    
    # Display all information
    display.blit(surface, (0, 0))
    display.blit(text_surface_speed, (10, 20))
    display.blit(text_surface_steering, (10, 50))
    display.blit(text_surface_distance, (10, 80))
    display.blit(target_dist_text, (10, 110))
    display.blit(text_surface_control, (10, 140))

# Improved radar processing with more robust filtering
radar_history = []
last_valid_distance = 100  # Initial value

def radar_callback(data, ego_vehicle=None, target_vehicle=None, radar=None):
    global distance_front, radar_history, last_valid_distance
    
    if ego_vehicle is None or not ego_vehicle.is_alive or target_vehicle is None or not target_vehicle.is_alive:
        print("Vehicle not available or not alive")
        return
    
    # Get locations in world space
    ego_location = ego_vehicle.get_location()
    target_location = target_vehicle.get_location()
    
    # Calculate direct distance between vehicles
    direct_distance = ego_location.distance(target_location)
    
    # Process radar detections
    if len(data) > 0:
        # Get radar transform in world coordinates
        radar_transform = radar.get_transform()
        radar_location = radar_transform.location
        radar_forward = radar_transform.get_forward_vector()
        
        # Calculate direction from ego to target
        to_target = carla.Vector3D(target_location.x - ego_location.x,
                                  target_location.y - ego_location.y,
                                  target_location.z - ego_location.z)
        to_target_normalized = carla.Vector3D(to_target.x, to_target.y, to_target.z)
        to_target_length = np.sqrt(to_target.x**2 + to_target.y**2 + to_target.z**2)
        if to_target_length > 0:
            to_target_normalized.x /= to_target_length
            to_target_normalized.y /= to_target_length
            to_target_normalized.z /= to_target_length
        
        # Collect all valid detections
        valid_detections = []
        for detection in data:
            # Calculate world position of the detection
            detection_location = radar_location + detection.depth * carla.Vector3D(
                radar_forward.x * detection.depth * np.cos(detection.azimuth),
                radar_forward.y * detection.depth * np.sin(detection.azimuth),
                radar_forward.z * detection.depth * np.sin(detection.altitude)
            )
            
            # Check if this is likely the target vehicle
            dist_to_target = detection_location.distance(target_location)
            if dist_to_target < 3.0:  # If detection is close to target vehicle
                valid_detections.append(detection.depth)
        
        # Update distance based on valid detections
        if valid_detections:
            # Use the smallest valid detection
            closest_detection = min(valid_detections)
            
            # Add to history with weighting toward closer objects
            radar_history.append(closest_detection)
            
            # Keep history limited to avoid stale data
            if len(radar_history) > 5:
                radar_history.pop(0)
            
            # Use median filtering for more stability
            distance_front = sorted(radar_history)[len(radar_history)//2]
            last_valid_distance = distance_front
        else:
            # If no valid detections but we see target vehicle
            if direct_distance < 50:
                # Use direct distance minus vehicle offset
                distance_front = max(direct_distance - 3.0, 0.0)  # Account for vehicle size
                last_valid_distance = distance_front
                
                # Update history with direct measurement
                radar_history.append(distance_front)
                if len(radar_history) > 5:
                    radar_history.pop(0)
            else:
                # No valid detections and target too far - use last valid with decay
                if radar_history:
                    distance_front = last_valid_distance + 0.5  # Gradually increase distance if no detection
                    distance_front = min(distance_front, 100)  # Cap at reasonable value
                else:
                    distance_front = 100  # Default to large value
    else:
        # No detections at all - use direct line of sight with safety margin
        if direct_distance < 50:
            distance_front = max(direct_distance - 3.0, 0.0)
        else:
            distance_front = last_valid_distance  # Maintain last known distance
    
    # Safety check - ensure positive distance
    distance_front = max(distance_front, 0.1)
    
    print(f"Distance to target vehicle: {distance_front:.2f}m (direct: {direct_distance:.2f}m)")

# Improved target vehicle spawn function
def spawn_target_vehicle(world, vehicle_bp, spawn_transform, target_speed=30, target_distance=8.0):
    """
    Spawn a target vehicle ahead of the player vehicle at the desired distance.
    """
    # Get location and forward vector from the transform
    spawn_location = spawn_transform.location
    forward_vector = spawn_transform.get_forward_vector()
    
    # Position target vehicle at the desired distance ahead
    target_location = carla.Location(
        x=spawn_location.x + forward_vector.x * target_distance,
        y=spawn_location.y + forward_vector.y * target_distance,
        z=spawn_location.z + 0.5  # Slight height offset to avoid collisions
    )
    
    # Create transform with same rotation as player
    target_transform = carla.Transform(target_location, spawn_transform.rotation)
    
    # Try to spawn at the calculated position
    target_vehicle = world.try_spawn_actor(vehicle_bp, target_transform)
    
    # If first attempt fails, try spawn points near the player
    if target_vehicle is None:
        print("Direct spawn failed, trying alternative positions...")
        spawn_points = world.get_map().get_spawn_points()
        
        # Sort spawn points by distance from desired position
        spawn_points.sort(key=lambda sp: target_location.distance(sp.location))
        
        # Try the closest few spawn points
        for sp in spawn_points[:10]:
            target_vehicle = world.try_spawn_actor(vehicle_bp, sp)
            if target_vehicle is not None:
                print(f"Target vehicle spawned at alternative location")
                break
    
    # Final fallback - try any spawn point
    if target_vehicle is None:
        print("Alternative positions failed, trying any valid spawn point...")
        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        
        for sp in spawn_points[:20]:
            target_vehicle = world.try_spawn_actor(vehicle_bp, sp)
            if target_vehicle is not None:
                print(f"Target vehicle spawned at random location")
                break
    
    if target_vehicle is None:
        print("All spawn attempts failed.")
        return None
    
    # Set initial velocity (lower than requested to avoid immediate collisions)
    safe_initial_speed = min(target_speed, 20)  # Cap initial speed
    
    # Get road direction
    waypoint = world.get_map().get_waypoint(target_vehicle.get_location())
    if waypoint:
        road_dir = waypoint.transform.get_forward_vector()
        target_vehicle.set_target_velocity(carla.Vector3D(
            x=road_dir.x * safe_initial_speed / 3.6,
            y=road_dir.y * safe_initial_speed / 3.6,
            z=0
        ))
    else:
        # Fallback to vehicle's transform
        forward = target_vehicle.get_transform().get_forward_vector()
        target_vehicle.set_target_velocity(carla.Vector3D(
            x=forward.x * safe_initial_speed / 3.6,
            y=forward.y * safe_initial_speed / 3.6,
            z=0
        ))
    
    # Enable autopilot for more realistic movement
    target_vehicle.set_autopilot(True)
    
    print(f"Target vehicle initialized with speed {safe_initial_speed:.2f} km/h")
    return target_vehicle

# Helper function to position ego vehicle behind target vehicle with proper spacing
def position_ego_behind_target(world, ego_vehicle, target_vehicle, distance=20.0):
    """
    Move the ego vehicle behind the target vehicle at the specified distance.
    """
    if not target_vehicle or not target_vehicle.is_alive:
        print("Target vehicle is not available for positioning")
        return False
        
    # Get target vehicle's transform
    target_transform = target_vehicle.get_transform()
    target_location = target_transform.location
    target_rotation = target_transform.rotation
    
    # Get backward vector (opposite of forward)
    forward_vector = target_transform.get_forward_vector()
    backward_vector = carla.Vector3D(-forward_vector.x, -forward_vector.y, -forward_vector.z)
    
    # Find waypoint at the target vehicle's location
    waypoint = world.get_map().get_waypoint(target_location)
    
    if waypoint:
        # Position ego vehicle behind target vehicle, following the road
        back_waypoint = waypoint.previous(distance)[0] if waypoint.previous(distance) else None
        
        if back_waypoint:
            # Use waypoint for better road alignment
            ego_location = back_waypoint.transform.location
            ego_rotation = back_waypoint.transform.rotation
            ego_transform = carla.Transform(ego_location, ego_rotation)
        else:
            # Fallback to simple positioning
            ego_location = carla.Location(
                x=target_location.x + backward_vector.x * distance,
                y=target_location.y + backward_vector.y * distance,
                z=target_location.z + 0.5  # Slight height offset
            )
            ego_transform = carla.Transform(ego_location, target_rotation)
    else:
        # Fallback if no waypoint found
        ego_location = carla.Location(
            x=target_location.x + backward_vector.x * distance,
            y=target_location.y + backward_vector.y * distance,
            z=target_location.z + 0.5
        )
        ego_transform = carla.Transform(ego_location, target_rotation)
    
    # Set ego vehicle's transform
    ego_vehicle.set_transform(ego_transform)
    
    # Ensure zero initial velocity
    ego_vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
    
    print(f"Positioned ego vehicle behind target at distance {distance:.2f}m")
    return True
    
# Main simulation loop with improved safety
def main():
    global distance_front
    distance_front = 100  # Initialize to a large value
    
    # ACC parameters
    target_distance = 8.0  # Target gap in meters
    
    # Better tuned PID controllers
    steering_pid = PIDController(kp=0.15, ki=0.01, kd=0.1, integral_limit=3)
    distance_pid = DistancePIDController(kp=0.8, ki=0.05, kd=0.4, target_distance=target_distance, integral_limit=3)

    # Setup CARLA client
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(15.0)  # Longer timeout for more reliability
    
    try:
        world = client.get_world()
    except Exception as e:
        print(f"Failed to connect to CARLA: {e}")
        return
    
    # Configure simulation settings for better realism
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05  # 20 FPS
    world.apply_settings(settings)
    
    # Clear existing vehicles for a clean slate
    for actor in world.get_actors():
        if actor.type_id.startswith('vehicle') or actor.type_id.startswith('sensor'):
            actor.destroy()
    
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
    
    # Find a suitable spawn point
    spawn_points = world.get_map().get_spawn_points()
    
    # Sort spawn points by road complexity (prefer straighter roads for testing)
    def road_complexity(sp):
        waypoint = world.get_map().get_waypoint(sp.location)
        if not waypoint:
            return 1000  # High complexity if no waypoint
        # Check if on a straight segment
        try:
            next_wps = waypoint.next(5.0)
            if not next_wps:
                return 900
            next_wp = next_wps[0]
            heading_diff = abs(waypoint.transform.rotation.yaw - next_wp.transform.rotation.yaw)
            heading_diff = min(heading_diff, 360 - heading_diff)
            return heading_diff
        except:
            return 800
    
    spawn_points.sort(key=road_complexity)
    
    vehicle = None
    for sp in spawn_points[:10]:  # Try the 10 straightest roads
        if sp.location is not None:
            vehicle = world.try_spawn_actor(vehicle_bp, sp)
            if vehicle is not None:
                break
    
    # Fallback to any spawn point
    if vehicle is None:
        random.shuffle(spawn_points)
        for sp in spawn_points:
            if sp.location is not None:
                vehicle = world.try_spawn_actor(vehicle_bp, sp)
                if vehicle is not None:
                    break
    
    if vehicle is None:
        print("Failed to find a suitable spawn point for ego vehicle.")
        return
    
    # Wait a moment for the physics to settle
    for _ in range(5):
        world.tick()
    
    # First spawn target vehicle - using ego vehicle as reference
    target_vehicle = spawn_target_vehicle(world, blueprint_library.find('vehicle.tesla.model3'), 
                                         vehicle.get_transform(), target_speed=25, 
                                         target_distance=30)  # Spawn further ahead initially
    
    if target_vehicle is None:
        print("Failed to spawn target vehicle.")
        return
    
    # Wait for target vehicle to initialize
    for _ in range(10):
        world.tick()
        
    # Now position ego vehicle properly
    position_ego_behind_target(world, vehicle, target_vehicle, distance=20.0)
    
    # Wait again for physics to settle
    for _ in range(10):
        world.tick()
    
    # Setup waypoints for lane following
    waypoints = get_lane_waypoints(world, distance=2.0)
    
    # Setup display
    pygame.init()
    display = pygame.display.set_mode((800, 600), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("CARLA ACC Simulation - 8m Target")
    font = pygame.font.Font(None, 36)
    
    # Improved radar setup
    radar_bp = blueprint_library.find('sensor.other.radar')
    radar_bp.set_attribute('horizontal_fov', '60')  # Wider horizontal FOV
    radar_bp.set_attribute('vertical_fov', '30')    # Wider vertical FOV
    radar_bp.set_attribute('range', '100')          # Good range
    radar_bp.set_attribute('points_per_second', '4000')  # More points for better detection
    
    radar_transform = carla.Transform(carla.Location(x=2.0, y=0, z=1.0))
    radar = world.spawn_actor(radar_bp, radar_transform, attach_to=vehicle)
    
    # Setup improved radar callback
    radar.listen(lambda data: radar_callback(data, vehicle, target_vehicle, radar))
    
    # Setup camera
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute("image_size_x", "800")
    camera_bp.set_attribute("image_size_y", "600")
    camera_bp.set_attribute("fov", "110")
    
    # Two camera views
    camera_transform_normal = carla.Transform(carla.Location(x=1.5, y=0, z=2.0))
    camera = world.spawn_actor(camera_bp, camera_transform_normal, attach_to=vehicle)
    
    # Data collection setup
    distance_data = []
    speed_data = []
    throttle_data = []
    time_data = []
    start_time = time.time()
    
    # Setup camera callback
    camera.listen(lambda image: process_img(image, display, vehicle, 0.0, font, distance_front, 
                                         target_distance, 0.0))
    
    # Main control loop
    try:
        print("Starting ACC simulation...")
        print(f"Target following distance: {target_distance} meters")
        
        # Flag to ensure we wait for valid distance measurements
        valid_measurement = False
        
        # Give some time to initialize
        for _ in range(20):
            world.tick()
            pygame.display.flip()
            time.sleep(0.05)
        
        frame_count = 0
        steering_angle = 0.0
        throttle_value = 0.0
        
        while True:
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return
                    # Allow manual override of target distance
                    elif event.key == pygame.K_UP:
                        target_distance += 1.0
                        distance_pid.target_distance = target_distance
                        print(f"Increased target distance to {target_distance:.1f}m")
                    elif event.key == pygame.K_DOWN:
                        target_distance = max(4.0, target_distance - 1.0)
                        distance_pid.target_distance = target_distance
                        print(f"Decreased target distance to {target_distance:.1f}m")
            
            # Ensure target vehicle exists and is alive
            if target_vehicle is None or not target_vehicle.is_alive:
                print("Target vehicle lost, attempting to respawn...")
                # Try to respawn target vehicle
                target_vehicle = spawn_target_vehicle(world, blueprint_library.find('vehicle.tesla.model3'), 
                                                   vehicle.get_transform(), target_speed=25)
                if target_vehicle is None:
                    print("Failed to respawn target vehicle. Ending simulation.")
                    break
                    
                # Reset radar history when target changes
                radar_history.clear()
                
                # Wait for target to initialize
                for _ in range(10):
                    world.tick()
            
            # Wait for valid measurements
            if distance_front == 100 and not valid_measurement:
                if frame_count > 100:  # If too many frames without valid measurement
                    print("Warning: No valid distance measurements. Check target vehicle visibility.")
                    # Try looking for target vehicle directly
                    ego_location = vehicle.get_location()
                    if target_vehicle and target_vehicle.is_alive:
                        target_location = target_vehicle.get_location()
                        direct_distance = ego_location.distance(target_location)
                        if direct_distance < 50:
                            distance_front = max(direct_distance - 3.0, 0.1)
                            valid_measurement = True
                            print(f"Using direct line of sight: {distance_front:.2f}m")
                world.tick()
                pygame.display.flip()
                frame_count += 1
                continue
            
            valid_measurement = True
            frame_count = 0
            
            # Calculate control inputs
            dt = 0.05  # Fixed time step
            
            # Compute steering using Stanley controller
            steering_angle = compute_steering_angle(vehicle, waypoints, steering_pid, k=0.5)
            steering_control = np.clip(steering_angle / np.radians(30), -1.0, 1.0)
            
            # Get vehicle speed
            velocity = vehicle.get_velocity()
            speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            
            # Compute throttle/brake using fuzzy logic ACC
            throttle_value = fuzzy_logic_acc(distance_front, speed, target_distance)
            
            # Verify with PID controller as a safety check
            pid_throttle = distance_pid.compute(distance_front, dt)
            
            # Use more conservative of the two controllers
            if pid_throttle < throttle_value and distance_front < target_distance:
                throttle_value = pid_throttle
                control_mode = "PID (safety override)"
            else:
                control_mode = "Fuzzy Logic"
            
            print(f"Speed: {speed:.2f} km/h | Distance: {distance_front:.2f}m | " +
                 f"Control: {control_mode} | Throttle: {throttle_value:.2f} | " +
                 f"Steering: {np.degrees(steering_angle):.2f}°")
            
            # Apply control to vehicle
            control = carla.VehicleControl()
            
            if throttle_value >= 0:
                control.throttle = float(throttle_value)
                control.brake = 0.0
            else:
                control.throttle = 0.0
                control.brake = float(abs(throttle_value))
                
            control.steer = float(steering_control)
            control.hand_brake = False
            control.manual_gear_shift = False
            
            vehicle.apply_control(control)
            
            # Collect data for analysis
            current_time = time.time() - start_time
            distance_data.append(distance_front)
            speed_data.append(speed)
            throttle_data.append(throttle_value)
            time_data.append(current_time)
            
            # Update camera display with latest steering angle and throttle
            camera.listen(lambda image: process_img(image, display, vehicle, steering_angle, font, 
                                                 distance_front, target_distance, throttle_value))
            
            world.tick()
            pygame.display.flip()
    
    except Exception as e:
        print(f"Error in simulation: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Plot results before cleanup
        if len(time_data) > 10:  # Only if we have meaningful data
            plt.figure(figsize=(12, 8))
            
            plt.subplot(3, 1, 1)
            plt.plot(time_data, distance_data, 'b-')
            plt.axhline(y=target_distance, color='r', linestyle='--', label=f'Target ({target_distance}m)')
            plt.ylabel('Distance (m)')
            plt.title('ACC Performance')
            plt.grid(True)
            plt.legend()
            
            plt.subplot(3, 1, 2)
            plt.plot(time_data, speed_data, 'g-')
            plt.ylabel('Speed (km/h)')
            plt.grid(True)
            
            plt.subplot(3, 1, 3)
            plt.plot(time_data, throttle_data, 'r-')
            plt.axhline(y=0, color='k', linestyle='-')
            plt.ylabel('Throttle/Brake')
            plt.xlabel('Time (s)')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig('acc_performance.png')
            plt.show()
        
        # Clean up actors
        print('Cleaning up actors...')
        if camera:
            camera.destroy()
        if radar:
            radar.destroy()
        if vehicle:
            vehicle.destroy()
        if target_vehicle and target_vehicle.is_alive:
            target_vehicle.destroy()
        
        pygame.quit()
        print('Actors cleaned up. Simulation ended.')

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Cancelled by user')
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
    finally:
        print('Exiting...')

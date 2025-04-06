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

        # Apply stronger braking when too close
        if current_distance < 5:
            return -1.0 * (5 - current_distance) / 5  # Progressive braking based on distance
        
        # Apply mild braking when approaching target distance
        if current_distance < self.target_distance:
            return output * 0.5  # Dampen throttle response when close
            
        return output

# Improved Fuzzy Logic System for Adaptive Cruise Control (ACC)
def fuzzy_logic_acc(distance_front, speed, target_distance=8.0):
    # Expanded distance range for better granularity
    distance = ctrl.Antecedent(np.arange(0, 31, 1), 'distance')  # 0 to 30 meters
    velocity = ctrl.Antecedent(np.arange(0, 101, 1), 'velocity')  # 0 to 100 km/h
    throttle = ctrl.Consequent(np.arange(-1.0, 1.1, 0.1), 'throttle')  # -1 to 1 (brake to throttle)

    # More granular distance membership functions centered around target distance
    distance['dangerous'] = fuzz.trimf(distance.universe, [0, 0, 4])
    distance['very_close'] = fuzz.trimf(distance.universe, [2, 5, 7])
    distance['close'] = fuzz.trimf(distance.universe, [5, 7, 8])
    distance['target'] = fuzz.trimf(distance.universe, [7, 8, 9])
    distance['comfortable'] = fuzz.trimf(distance.universe, [8, 10, 12])
    distance['far'] = fuzz.trimf(distance.universe, [11, 15, 30])

    velocity['very_slow'] = fuzz.trimf(velocity.universe, [0, 0, 20])
    velocity['slow'] = fuzz.trimf(velocity.universe, [10, 25, 40])
    velocity['medium'] = fuzz.trimf(velocity.universe, [30, 50, 70])
    velocity['fast'] = fuzz.trimf(velocity.universe, [60, 80, 100])
    
    # Expanded throttle range to include braking
    throttle['hard_brake'] = fuzz.trimf(throttle.universe, [-1.0, -1.0, -0.6])
    throttle['brake'] = fuzz.trimf(throttle.universe, [-0.8, -0.4, 0])
    throttle['coast'] = fuzz.trimf(throttle.universe, [-0.2, 0, 0.2])
    throttle['light_throttle'] = fuzz.trimf(throttle.universe, [0.1, 0.3, 0.5])
    throttle['medium_throttle'] = fuzz.trimf(throttle.universe, [0.4, 0.6, 0.8])
    throttle['full_throttle'] = fuzz.trimf(throttle.universe, [0.7, 1.0, 1.0])

    # Enhanced rule set for smoother ACC
    rules = [
        # Dangerous distance - brake hard regardless of speed
        ctrl.Rule(distance['dangerous'], throttle['hard_brake']),
        
        # Very close distance
        ctrl.Rule(distance['very_close'] & velocity['very_slow'], throttle['brake']),
        ctrl.Rule(distance['very_close'] & velocity['slow'], throttle['brake']),
        ctrl.Rule(distance['very_close'] & velocity['medium'], throttle['hard_brake']),
        ctrl.Rule(distance['very_close'] & velocity['fast'], throttle['hard_brake']),
        
        # Close distance
        ctrl.Rule(distance['close'] & velocity['very_slow'], throttle['coast']),
        ctrl.Rule(distance['close'] & velocity['slow'], throttle['coast']),
        ctrl.Rule(distance['close'] & velocity['medium'], throttle['brake']),
        ctrl.Rule(distance['close'] & velocity['fast'], throttle['brake']),
        
        # Target distance - maintain current speed
        ctrl.Rule(distance['target'] & velocity['very_slow'], throttle['light_throttle']),
        ctrl.Rule(distance['target'] & velocity['slow'], throttle['light_throttle']),
        ctrl.Rule(distance['target'] & velocity['medium'], throttle['coast']),
        ctrl.Rule(distance['target'] & velocity['fast'], throttle['coast']),
        
        # Comfortable distance
        ctrl.Rule(distance['comfortable'] & velocity['very_slow'], throttle['medium_throttle']),
        ctrl.Rule(distance['comfortable'] & velocity['slow'], throttle['medium_throttle']),
        ctrl.Rule(distance['comfortable'] & velocity['medium'], throttle['light_throttle']),
        ctrl.Rule(distance['comfortable'] & velocity['fast'], throttle['light_throttle']),
        
        # Far distance
        ctrl.Rule(distance['far'] & velocity['very_slow'], throttle['full_throttle']),
        ctrl.Rule(distance['far'] & velocity['slow'], throttle['full_throttle']),
        ctrl.Rule(distance['far'] & velocity['medium'], throttle['medium_throttle']),
        ctrl.Rule(distance['far'] & velocity['fast'], throttle['medium_throttle'])
    ]

    acc_ctrl = ctrl.ControlSystem(rules)
    acc = ctrl.ControlSystemSimulation(acc_ctrl)

    # Bound the distance input to avoid errors
    bounded_distance = min(max(distance_front, 0), 30)
    acc.input['distance'] = bounded_distance
    acc.input['velocity'] = speed

    try:
        acc.compute()
        if 'throttle' in acc.output:
            return acc.output['throttle']
        else:
            print("Warning: Throttle output not computed!")
            # Fallback logic
            if distance_front < 5:
                return -1.0  # Emergency brake
            elif distance_front < target_distance:
                return 0.0  # Coast
            else:
                return 0.5  # Moderate throttle
    except Exception as e:
        print(f"Error in fuzzy computation: {e}")
        # Simple fallback logic
        if distance_front < 5:
            return -1.0
        elif distance_front < target_distance:
            return 0.0
        else:
            return 0.5

# Lane Waypoints
def get_lane_waypoints(world, distance=2.0):
    return [wp for wp in world.get_map().generate_waypoints(distance) if wp.lane_type == carla.LaneType.Driving]

def find_closest_waypoint(vehicle_location, waypoints):
    return min(waypoints, key=lambda wp: vehicle_location.distance(wp.transform.location))

# Compute Steering Angle for Stanley Controller
def compute_steering_angle(vehicle, waypoints, pid_controller, k=0.2):
    vehicle_transform = vehicle.get_transform()
    vehicle_location = vehicle_transform.location
    vehicle_yaw = np.radians(vehicle_transform.rotation.yaw)

    closest_waypoint = find_closest_waypoint(vehicle_location, waypoints)
    if closest_waypoint is None:
        return 0.0

    # Fix: Access the location from transform before accessing x and y
    dx = closest_waypoint.transform.location.x - vehicle_location.x
    dy = closest_waypoint.transform.location.y - vehicle_location.y
    cte = dy * np.cos(vehicle_yaw) - dx * np.sin(vehicle_yaw)

    path_yaw = np.radians(closest_waypoint.transform.rotation.yaw)
    heading_error = np.arctan2(np.sin(path_yaw - vehicle_yaw), np.cos(path_yaw - vehicle_yaw))

    vehicle_velocity = vehicle.get_velocity()
    speed = max(3.6 * np.sqrt(vehicle_velocity.x**2 + vehicle_velocity.y**2 + vehicle_velocity.z**2), 5)
    stanley_steering = heading_error + np.arctan(k * cte / speed)

    dt = 0.05
    pid_correction = np.clip(pid_controller.compute(cte, dt), -np.radians(10), np.radians(10))

    steering_angle = stanley_steering + pid_correction
    steering_angle = np.clip(steering_angle, -np.radians(30), np.radians(30))

    return steering_angle

# Improved Process Camera Image
def process_img(image, display, vehicle, steering_angle, font, distance_front=100, target_distance=8.0):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))[:, :, :3][:, :, ::-1]
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    
    velocity = vehicle.get_velocity()
    speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

    text_surface_speed = font.render(f'Speed: {speed:.2f} km/h', True, (255, 255, 255))
    text_surface_steering = font.render(f'Steering: {np.degrees(steering_angle):.2f}°', True, (255, 255, 255))
    text_surface_distance = font.render(f'Distance: {distance_front:.2f} m', True, (255, 255, 255))
    
    # Change color based on distance
    if distance_front < 5:
        distance_color = (255, 0, 0)  # Red if too close
    elif distance_front < target_distance:
        distance_color = (255, 255, 0)  # Yellow if approaching target
    else:
        distance_color = (0, 255, 0)  # Green if at or beyond target
    
    target_dist_text = font.render(f'Target: {target_distance:.1f} m', True, distance_color)
    
    display.blit(surface, (0, 0))
    display.blit(text_surface_speed, (10, 20))
    display.blit(text_surface_steering, (10, 50))
    display.blit(text_surface_distance, (10, 80))
    display.blit(target_dist_text, (10, 110))

def radar_callback(data, target_vehicle=None, radar=None):
    global distance_front
    
    # Print debug info
    print(f"Radar detected {len(data)} objects")
    
    if target_vehicle is None or not target_vehicle.is_alive:
        print("Target vehicle is missing or not alive")
        # Keep previous distance or set to default
        return
    
    # Get target vehicle location
    target_location = target_vehicle.get_location()

    # Ensure radar object is passed correctly and accessible
    if radar is None:
        print("Radar is not available.")
        return
    
    # Get the radar sensor's location directly from the radar's transform
    radar_location = radar.get_transform().location
    
    # Calculate direct distance to target (more reliable)
    direct_distance = radar_location.distance(target_location)
    print(f"Direct distance to target: {direct_distance:.2f}m")
    
    # Also check radar readings
    closest_distance = float('inf')
    for detection in data:
        if detection.depth < closest_distance and detection.depth > 0:
            closest_distance = detection.depth
    
    print(f"Radar closest distance: {closest_distance:.2f}m")
    
    # Use the direct distance if radar isn't detecting properly
    if closest_distance < float('inf'):
        distance_front = closest_distance
    else:
        distance_front = direct_distance if direct_distance < 100 else 100
    
    print(f"Setting distance_front to: {distance_front:.2f}m")


    
    
def spawn_target_vehicle(world, vehicle_bp, spawn_transform, target_speed=20, target_distance=8.0):
    """
    Spawn a target vehicle ahead of the player vehicle at the desired distance.
    Uses a more robust approach to find valid spawn positions.
    """
    # Get location from the transform
    spawn_location = spawn_transform.location
    
    # Find all spawn points in the world
    all_spawn_points = world.get_map().get_spawn_points()
    
    # Calculate the player's forward vector
    forward_vector = spawn_transform.get_forward_vector()
    
    # Find a suitable spawn point ahead of the player
    print(f"Looking for spawn points for target vehicle...")
    suitable_points = []
    
    for sp in all_spawn_points:
        # Calculate vector from player to spawn point
        to_sp = carla.Location(sp.location.x - spawn_location.x, 
                              sp.location.y - spawn_location.y,
                              0)  # Ignore height difference
        
        # Project this vector onto player's forward vector to get distance ahead
        distance_ahead = to_sp.x * forward_vector.x + to_sp.y * forward_vector.y
        
        # Get perpendicular distance (lane offset)
        perpendicular = abs(to_sp.x * forward_vector.y - to_sp.y * forward_vector.x)
        
        # Check if this point is ahead and in a reasonable lane position
        if distance_ahead > 0 and distance_ahead < 30 and perpendicular < 4.0:
            suitable_points.append((sp, distance_ahead))
    
    # Sort by distance from ideal target distance
    suitable_points.sort(key=lambda x: abs(x[1] - target_distance))
    
    # Try each suitable point until successful
    target_vehicle = None
    
    if suitable_points:
        print(f"Found {len(suitable_points)} potential spawn points.")
        for sp, dist in suitable_points[:10]:  # Try the 10 best points
            target_vehicle = world.try_spawn_actor(vehicle_bp, carla.Transform(sp.location, spawn_transform.rotation))
            if target_vehicle is not None:
                print(f"Target vehicle spawned successfully at {sp.location}, distance: {dist:.2f}m")
                break
            
    # If no suitable spawn points, try manual positioning as fallback
    if target_vehicle is None:
        print("No suitable spawn points found. Trying manual positioning...")
        
        # Try several distances ahead
        for distance_multiplier in [1.0, 1.5, 2.0, 3.0, 5.0]:
            test_distance = target_distance * distance_multiplier
            
            # Position target vehicle at the desired distance ahead
            target_location = carla.Location(
                x=spawn_location.x + forward_vector.x * test_distance,
                y=spawn_location.y + forward_vector.y * test_distance,
                z=spawn_location.z + 0.5  # Slight height offset to avoid collisions
            )
            
            # Create transform with same rotation as player
            target_transform = carla.Transform(target_location, spawn_transform.rotation)
            
            # Try to spawn
            target_vehicle = world.try_spawn_actor(vehicle_bp, target_transform)
            
            if target_vehicle is not None:
                print(f"Target vehicle spawned via manual positioning at distance {test_distance:.2f}m")
                break
    
    # Final fallback - try to spawn at valid spawn points that aren't occupied
    if target_vehicle is None:
        print("Manual positioning failed. Trying random spawn points...")
        random.shuffle(all_spawn_points)
        
        for sp in all_spawn_points[:20]:  # Try up to 20 random points
            target_vehicle = world.try_spawn_actor(vehicle_bp, sp)
            if target_vehicle is not None:
                print(f"Target vehicle spawned at random location {sp.location}")
                break
    
    if target_vehicle is None:
        print("All methods to spawn target vehicle failed.")
        return None
    
    # Set initial speed and direction based on the road
    waypoint = world.get_map().get_waypoint(target_vehicle.get_location())
    if waypoint:
        # Get road direction from waypoint
        road_dir = waypoint.transform.get_forward_vector()
        target_vehicle.set_target_velocity(carla.Vector3D(
            x=road_dir.x * target_speed / 3.6,
            y=road_dir.y * target_speed / 3.6,
            z=0
        ))
    else:
        # Fallback to using vehicle's forward vector
        forward = target_vehicle.get_transform().get_forward_vector()
        target_vehicle.set_target_velocity(carla.Vector3D(
            x=forward.x * target_speed / 3.6,
            y=forward.y * target_speed / 3.6,
            z=0
        ))
    
    print(f"Target vehicle initialized with speed {target_speed:.2f} km/h")
    return target_vehicle  # Return the target vehicle actor

# Helper function to position ego vehicle behind target vehicle
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
    
    # Position ego vehicle behind target vehicle
    ego_location = carla.Location(
        x=target_location.x + backward_vector.x * distance,
        y=target_location.y + backward_vector.y * distance,
        z=target_location.z + 0.5  # Slight height offset
    )
    
    # Set ego vehicle's transform
    ego_transform = carla.Transform(ego_location, target_rotation)
    success = ego_vehicle.set_transform(ego_transform)
    
    print(f"Positioned ego vehicle behind target at distance {distance:.2f}m")
    return True
    
# Main simulation loop
def main():
    global distance_front
    distance_front = 100  # Initialize to a large value
    
    # ACC parameters - changed to 8.0 meters
    target_distance = 8.0  # Target gap in meters
    
    # PID controllers with tuned parameters for 8m distance
    steering_pid = PIDController(kp=0.1, ki=0.01, kd=0.05, integral_limit=5)
    distance_pid = DistancePIDController(kp=0.5, ki=0.02, kd=0.3, target_distance=target_distance, integral_limit=5)

    # Setup CARLA client
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    
    # Configure simulation settings for better realism
    settings = world.get_settings()
    settings.synchronous_mode = True  # Enable synchronous mode
    settings.fixed_delta_seconds = 0.05  # 20 FPS
    world.apply_settings(settings)
    
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
    
    # Find a suitable spawn point
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    
    vehicle = None
    for sp in spawn_points:
        if sp.location is not None:
            vehicle = world.try_spawn_actor(vehicle_bp, sp)
            if vehicle is not None:
                break
    
    if vehicle is None:
        print("Failed to find a suitable spawn point.")
        return
    
    # Spawn target vehicle - pass the transform instead of just location
    target_vehicle = spawn_target_vehicle(world, blueprint_library.find('vehicle.tesla.model3'), 
                                          vehicle.get_transform(), target_speed=20, 
                                          target_distance=target_distance)
    
    # Position ego vehicle behind target vehicle
    if target_vehicle:
        # Wait briefly for the target vehicle to initialize
        for _ in range(10):
            world.tick()
            
        # Position ego (our) vehicle behind the target vehicle
        position_ego_behind_target(world, vehicle, target_vehicle, distance=20.0)
    
    # Setup waypoints
    waypoints = get_lane_waypoints(world, distance=2.0)
    
    # Setup display
    pygame.init()
    display = pygame.display.set_mode((800, 600), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("CARLA ACC Simulation - 8m Target")
    font = pygame.font.Font(None, 36)
    
    # Setup radar sensor
    radar_bp = blueprint_library.find('sensor.other.radar')
    radar_bp.set_attribute('horizontal_fov', '45')  # Wider FOV
    radar_bp.set_attribute('vertical_fov', '20')    # Add vertical FOV
    radar_bp.set_attribute('range', '100')          # Longer range
    radar_bp.set_attribute('points_per_second', '2000')  # More points for better detection
    
    radar_transform = carla.Transform(carla.Location(x=2, y=0, z=1.5))  # Better position
    radar = world.spawn_actor(radar_bp, radar_transform, attach_to=vehicle)
    
    # Pass radar to the callback with the target_vehicle
    radar.listen(lambda data: radar_callback(data, target_vehicle, radar))

    
    # Setup camera
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute("image_size_x", "800")
    camera_bp.set_attribute("image_size_y", "600")
    camera_bp.set_attribute("fov", "110")
    
    camera_transform_normal = carla.Transform(carla.Location(x=0, y=0, z=5), carla.Rotation(pitch=-90))
    camera_transform_above = carla.Transform(carla.Location(x=-10, y=0, z=10), carla.Rotation(pitch=-30))
    
    camera_transform_state = camera_transform_normal
    camera = world.spawn_actor(camera_bp, camera_transform_state, attach_to=vehicle)
    
    steering = 0.0
    
    # Setup camera callback with captured parameters
    def camera_callback(image):
        process_img(image, display, vehicle, steering, font, distance_front, target_distance)
    
    camera.listen(camera_callback)
    
    # Main simulation loop
    clock = pygame.time.Clock()
    start_time = time.time()
    
    # Data recording
    time_list = []
    steering_angle_list = []
    distance_list = []
    throttle_list = []
    
    try:
        while time.time() - start_time < 120:  # Run for 2 minutes
            # Tick the world in synchronous mode
            world.tick()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_c:  # Camera view toggle
                        if camera_transform_state == camera_transform_normal:
                            camera_transform_state = camera_transform_above
                        else:
                            camera_transform_state = camera_transform_normal
                        camera.set_transform(camera_transform_state)
                    elif event.key == pygame.K_ESCAPE:  # Exit on ESC
                        raise KeyboardInterrupt
            
            # Get current vehicle state
            velocity = vehicle.get_velocity()
            speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            
            # Calculate throttle using fuzzy logic
            fuzzy_throttle = fuzzy_logic_acc(distance_front, speed, target_distance)
            
            dt = world.get_settings().fixed_delta_seconds
            
            # Calculate distance-based PID adjustment
            distance_pid_output = distance_pid.compute(distance_front, dt)
            
            # Combine fuzzy and PID outputs
            # Weight more toward fuzzy logic when far from target, more toward PID when close
            if abs(distance_front - target_distance) < 3:
                # Close to target, prioritize PID for fine control
                throttle_brake = 0.3 * fuzzy_throttle + 0.7 * distance_pid_output
            else:
                # Far from target, prioritize fuzzy for responsive control
                throttle_brake = 0.7 * fuzzy_throttle + 0.3 * distance_pid_output
            
            # Calculate steering
            steering = compute_steering_angle(vehicle, waypoints, steering_pid)
            
            # Apply control
            if throttle_brake < 0:
                # Negative value = braking
                control = carla.VehicleControl(
                    throttle=0.0, 
                    brake=abs(throttle_brake),
                    steer=steering
                )
            else:
                # Positive value = throttle
                control = carla.VehicleControl(
                    throttle=np.clip(throttle_brake, 0, 1), 
                    brake=0.0,
                    steer=steering
                )
            
            vehicle.apply_control(control)
            
            # Occasionally vary target vehicle speed for realism
            if target_vehicle and target_vehicle.is_alive:
                if random.random() < 0.01:  # 1% chance each frame
                    forward_vector = target_vehicle.get_transform().get_forward_vector()
                    new_speed = random.uniform(15, 25)  # Random speed between 15-25 km/h
                    target_vehicle.set_target_velocity(carla.Vector3D(
                        x=forward_vector.x * new_speed / 3.6, 
                        y=forward_vector.y * new_speed / 3.6,
                        z=0
                    ))
            
            # Record data
            time_list.append(time.time() - start_time)
            steering_angle_list.append(np.degrees(steering))
            distance_list.append(distance_front)
            throttle_list.append(throttle_brake)
            
            pygame.display.flip()
            clock.tick(20)  # Match the simulation FPS
    
    except KeyboardInterrupt:
        print("Simulation manually stopped.")
    
    finally:
        # Reset synchronous mode
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        
        # Destroy all actors
        print("Destroying actors...")
        camera.destroy()
        radar.destroy()
        vehicle.destroy()
        if target_vehicle and target_vehicle.is_alive:
            target_vehicle.destroy()
        pygame.quit()
        print("Simulation stopped.")
        
        # Plot results
        plt.figure(figsize=(12, 10))
        
        plt.subplot(3, 1, 1)
        plt.plot(time_list, steering_angle_list, label="Steering Angle (°)", color='b')
        plt.xlabel("Time (s)")
        plt.ylabel("Steering Angle (°)")
        plt.title("Steering Angle vs. Time")
        plt.grid(True)
        
        plt.subplot(3, 1, 2)
        plt.plot(time_list, distance_list, label=f"Distance (m)", color='g')
        plt.axhline(y=target_distance, color='r', linestyle='--', label=f"Target ({target_distance} m)")
        plt.xlabel("Time (s)")
        plt.ylabel("Distance (m)")
        plt.title("Distance to Lead Vehicle vs. Time")
        plt.legend()
        plt.grid(True)
        
        plt.subplot(3, 1, 3)
        plt.plot(time_list, throttle_list, label="Throttle/Brake Control", color='purple')
        plt.axhline(y=0, color='k', linestyle='-')
        plt.xlabel("Time (s)")
        plt.ylabel("Control Value")
        plt.title("Throttle/Brake Control vs. Time")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error in main function: {e}")


import carla
import numpy as np
import time
import pygame
import random
import matplotlib.pyplot as plt

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

# TimeGapController Class - ADD THIS HERE
class TimeGapController:
    def __init__(self, time_gap=1.5, min_gap=3.0, max_gap=20.0):
        self.time_gap = time_gap
        self.min_gap = min_gap
        self.max_gap = max_gap

    def compute(self, speed, distance_front, relative_velocity=0):
        """
        Compute the time gap and adjust the throttle/brake based on distance.
        """
        time_gap = np.clip(self.time_gap + (speed * 0.1), self.min_gap, self.max_gap)
        desired_distance = time_gap * speed
        distance_error = distance_front - desired_distance
        
        # Return a simple throttle/brake value for simplicity
        return np.clip(distance_error * 0.1, -1.0, 1.0)
        
# Advanced Time Gap Controller with Fuzzy Logic Elements
class AdaptiveTimeGapController:
    def __init__(self, 
                 base_time_gap=1.5, 
                 min_gap=3.0, 
                 max_gap=20.0, 
                 comfort_factor=0.8):
        self.base_time_gap = base_time_gap
        self.min_gap = min_gap
        self.max_gap = max_gap
        self.comfort_factor = comfort_factor
        
        # Fuzzy-inspired dynamic parameters
        self.kp = 0.6
        self.ki = 0.03
        self.kd = 0.1
        
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, speed, distance_front, relative_velocity=0):
        """
        Advanced ACC computation with consideration of:
        1. Dynamic time gap based on speed
        2. Relative velocity compensation
        3. Comfort-based acceleration/deceleration
        """
        # Adaptive time gap computation
        speed_adjusted_gap = np.clip(
            self.base_time_gap * (1 + np.log(speed + 1)), 
            self.min_gap, 
            self.max_gap
        )
        
        # Desired following distance with dynamic adjustment
        desired_distance = speed_adjusted_gap * speed
        
        # Distance error with relative velocity compensation
        distance_error = distance_front - desired_distance
        relative_velocity_factor = relative_velocity * 0.5
        
        # Emergency conditions
        if distance_front < 2.0:
            return -1.0  # Critical emergency braking
        
        # Dynamic PID parameters based on error magnitude
        adaptive_kp = self.kp * (1 + abs(distance_error)/10)
        dt = 0.05  # Fixed timestep
        
        # Integral term with advanced anti-windup
        self.integral += distance_error * dt
        self.integral = np.clip(self.integral, -10, 10)
        
        # Derivative term with relative velocity consideration
        derivative = (distance_error - self.prev_error) / dt + relative_velocity_factor
        
        # Compute output with comfort factor
        output = (
            adaptive_kp * distance_error + 
            self.ki * self.integral + 
            self.kd * derivative
        ) * self.comfort_factor
        
        # Symmetric clamping for smooth control
        self.prev_error = distance_error
        return np.clip(output, -1.0, 1.0)

# Lane Waypoints
def get_lane_waypoints(world, distance=2.0):
    return [wp for wp in world.get_map().generate_waypoints(distance) if wp.lane_type == carla.LaneType.Driving]

def find_closest_waypoint(vehicle_location, waypoints):
    if not waypoints:
        return None
    return min(waypoints, key=lambda wp: vehicle_location.distance(wp.transform.location))

# Compute Steering Angle for Stanley Controller
def compute_steering_angle(vehicle, waypoints, pid_controller, k=0.2):
    vehicle_transform = vehicle.get_transform()
    vehicle_location = vehicle_transform.location
    vehicle_yaw = np.radians(vehicle_transform.rotation.yaw)

    closest_waypoint = find_closest_waypoint(vehicle_location, waypoints)
    if closest_waypoint is None:
        return 0.0

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
    speed = min(speed, 30)  # Limit speed to 30 km/h

    text_surface_speed = font.render(f'Speed: {speed:.2f} km/h', True, (255, 255, 255))
    text_surface_steering = font.render(f'Steering: {np.degrees(steering_angle):.2f}°', True, (255, 255, 255))
    text_surface_distance = font.render(f'Distance: {distance_front:.2f} m', True, (255, 255, 255))

    if distance_front < 5:
        distance_color = (255, 0, 0)
    elif distance_front < target_distance:
        distance_color = (255, 255, 0)
    else:
        distance_color = (0, 255, 0)

    target_dist_text = font.render(f'Target: {target_distance:.1f} m', True, distance_color)

    display.blit(surface, (0, 0))
    display.blit(text_surface_speed, (10, 20))
    display.blit(text_surface_steering, (10, 50))
    display.blit(text_surface_distance, (10, 80))
    display.blit(target_dist_text, (10, 110))

# Radar Callback for Distance Measurement
def radar_callback(data, target_vehicle=None, ego_vehicle=None):
    distance_front = 100  # Default max distance

    if target_vehicle is None or not target_vehicle.is_alive or ego_vehicle is None:
        return distance_front

    direct_distances = []
    radar_distances = []

    target_location = target_vehicle.get_location()
    radar_location = ego_vehicle.get_location()
    direct_distance = radar_location.distance(target_location)
    direct_distances.append(direct_distance)

    for detection in data:
        if 0 < detection.depth < 100:
            radar_distances.append(detection.depth)

    if radar_distances:
        distance_front = np.median(radar_distances)
    elif direct_distances:
        distance_front = min(direct_distances)

    distance_front = np.clip(distance_front, 0, 100)

    return distance_front

def spawn_target_vehicle(world, vehicle_bp, spawn_transform, target_speed=20, target_distance=8.0):
    """
    Spawn a target vehicle ahead of the player vehicle at the desired distance.
    """
    spawn_location = spawn_transform.location
    all_spawn_points = world.get_map().get_spawn_points()
    forward_vector = spawn_transform.get_forward_vector()

    print(f"Looking for spawn points for target vehicle...")
    suitable_points = []

    for sp in all_spawn_points:
        to_sp = carla.Location(sp.location.x - spawn_location.x, 
                               sp.location.y - spawn_location.y, 0)
        distance_ahead = to_sp.x * forward_vector.x + to_sp.y * forward_vector.y
        perpendicular = abs(to_sp.x * forward_vector.y - to_sp.y * forward_vector.x)

        if distance_ahead > 0 and distance_ahead < 30 and perpendicular < 4.0:
            suitable_points.append((sp, distance_ahead))

    suitable_points.sort(key=lambda x: abs(x[1] - target_distance))
    target_vehicle = None

    if suitable_points:
        print(f"Found {len(suitable_points)} potential spawn points.")
        for sp, dist in suitable_points[:10]:
            target_vehicle = world.try_spawn_actor(vehicle_bp, carla.Transform(sp.location, spawn_transform.rotation))
            if target_vehicle is not None:
                print(f"Target vehicle spawned successfully at {sp.location}, distance: {dist:.2f}m")
                break

    if target_vehicle is None:
        print("No suitable spawn points found. Trying manual positioning...")
        for distance_multiplier in [1.0, 1.5, 2.0, 3.0, 5.0]:
            test_distance = target_distance * distance_multiplier
            target_location = carla.Location(
                x=spawn_location.x + forward_vector.x * test_distance,
                y=spawn_location.y + forward_vector.y * test_distance,
                z=spawn_location.z + 0.5
            )
            target_transform = carla.Transform(target_location, spawn_transform.rotation)
            target_vehicle = world.try_spawn_actor(vehicle_bp, target_transform)
            if target_vehicle is not None:
                print(f"Target vehicle spawned via manual positioning at distance {test_distance:.2f}m")
                break

    if target_vehicle is None:
        print("Manual positioning failed. Trying random spawn points...")
        random.shuffle(all_spawn_points)
        for sp in all_spawn_points[:20]:
            target_vehicle = world.try_spawn_actor(vehicle_bp, sp)
            if target_vehicle is not None:
                print(f"Target vehicle spawned at random location {sp.location}")
                break

    if target_vehicle is None:
        print("All methods to spawn target vehicle failed.")
        return None

    waypoint = world.get_map().get_waypoint(target_vehicle.get_location())
    if waypoint:
        road_dir = waypoint.transform.get_forward_vector()
        target_vehicle.set_target_velocity(carla.Vector3D(
            x=road_dir.x * target_speed / 3.6,
            y=road_dir.y * target_speed / 3.6,
            z=0
        ))
    else:
        forward = target_vehicle.get_transform().get_forward_vector()
        target_vehicle.set_target_velocity(carla.Vector3D(
            x=forward.x * target_speed / 3.6,
            y=forward.y * target_speed / 3.6,
            z=0
        ))

    print(f"Target vehicle initialized with speed {target_speed:.2f} km/h")
    return target_vehicle

# Helper function to position ego vehicle behind target vehicle
def position_ego_behind_target(world, ego_vehicle, target_vehicle, distance=20.0):
    """
    Move the ego vehicle behind the target vehicle at the specified distance.
    """
    if not target_vehicle or not target_vehicle.is_alive:
        print("Target vehicle is not available for positioning")
        return False

    target_transform = target_vehicle.get_transform()
    target_location = target_transform.location
    target_rotation = target_transform.rotation

    forward_vector = target_transform.get_forward_vector()
    backward_vector = carla.Vector3D(-forward_vector.x, -forward_vector.y, -forward_vector.z)

    ego_location = carla.Location(
        x=target_location.x + backward_vector.x * distance,
        y=target_location.y + backward_vector.y * distance,
        z=target_location.z + 0.5
    )

    ego_transform = carla.Transform(ego_location, target_rotation)
    success = ego_vehicle.set_transform(ego_transform)

    print(f"Positioned ego vehicle behind target at distance {distance:.2f}m")
    return True

# Compute Steering for Target Vehicle (Stanley Controller)
def compute_target_steering(target_vehicle, waypoints, pid_controller, k=0.2):
    target_transform = target_vehicle.get_transform()
    target_location = target_transform.location
    target_yaw = np.radians(target_transform.rotation.yaw)

    closest_waypoint = find_closest_waypoint(target_location, waypoints)
    if closest_waypoint is None:
        return 0.0

    dx = closest_waypoint.transform.location.x - target_location.x
    dy = closest_waypoint.transform.location.y - target_location.y
    cte = dy * np.cos(target_yaw) - dx * np.sin(target_yaw)

    path_yaw = np.radians(closest_waypoint.transform.rotation.yaw)
    heading_error = np.arctan2(np.sin(path_yaw - target_yaw), np.cos(path_yaw - target_yaw))

    target_velocity = target_vehicle.get_velocity()
    speed = max(3.6 * np.sqrt(target_velocity.x**2 + target_velocity.y**2 + target_velocity.z**2), 5)
    stanley_steering = heading_error + np.arctan(k * cte / speed)

    dt = 0.05
    pid_correction = np.clip(pid_controller.compute(cte, dt), -np.radians(10), np.radians(10))

    steering_angle = stanley_steering + pid_correction
    steering_angle = np.clip(steering_angle, -np.radians(30), np.radians(30))

    return steering_angle


    
def main():
    target_speed = 27.0
    # Enhanced simulation parameters
    target_distance = 8.0
    MIN_SPEED = 20.0  # Reduced minimum speed for more realistic behavior
    MAX_SPEED = 35.0  # Increased maximum speed range
    SIMULATION_DURATION = 180  # Extended simulation time

    # Advanced controllers with refined parameters
    steering_pid = PIDController(
        kp=0.15, 
        ki=0.02, 
        kd=0.08, 
        integral_limit=7
    )
    time_gap_controller = TimeGapController(
        time_gap=1.8,  # Slightly increased base time gap
        min_gap=3.0, 
        max_gap=25.0
    )

    # CARLA Client and World Setup
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # Synchronous Mode Configuration
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # Vehicle Blueprint Selection
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find('vehicle.tesla.model3')

    # Vehicle Spawning
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

    # Target Vehicle Spawning
    target_vehicle = spawn_target_vehicle(
        world, 
        blueprint_library.find('vehicle.tesla.model3'), 
        vehicle.get_transform(), 
        target_speed=27, 
        target_distance=target_distance
    )

    # Position Vehicles
    if target_vehicle:
        for _ in range(10):
            world.tick()
        position_ego_behind_target(world, vehicle, target_vehicle, distance=20.0)

    # Waypoint Generation
    waypoints = get_lane_waypoints(world, distance=2.0)

    # Pygame Initialization
    pygame.init()
    display = pygame.display.set_mode((800, 600), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("Advanced ACC Simulation")
    font = pygame.font.Font(None, 36)

    # Radar Sensor Setup
    radar_bp = blueprint_library.find('sensor.other.radar')
    radar_bp.set_attribute('horizontal_fov', '45')
    radar_bp.set_attribute('vertical_fov', '20')
    radar_bp.set_attribute('range', '100')
    radar_bp.set_attribute('points_per_second', '2000')

    radar_transform = carla.Transform(carla.Location(x=2, y=0, z=1.5))
    radar = world.spawn_actor(radar_bp, radar_transform, attach_to=vehicle)
    radar_data = {'distance': 100}
    
    def radar_listener(data):
        radar_data['distance'] = radar_callback(data, target_vehicle, vehicle)

    radar.listen(radar_listener)

    # Camera Setup
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute("image_size_x", "800")
    camera_bp.set_attribute("image_size_y", "600")
    camera_bp.set_attribute("fov", "110")

    camera_transform_normal = carla.Transform(carla.Location(x=0, y=0, z=5), carla.Rotation(pitch=-90))
    camera_transform_above = carla.Transform(carla.Location(x=-10, y=0, z=10), carla.Rotation(pitch=-30))
    camera_transform_state = camera_transform_normal
    camera = world.spawn_actor(camera_bp, camera_transform_state, attach_to=vehicle)

    steering = 0.0

    def camera_callback(image):
        process_img(image, display, vehicle, steering, font, radar_data['distance'], target_distance)

    camera.listen(camera_callback)

    # Simulation Initialization
    clock = pygame.time.Clock()
    start_time = time.time()

    # Data Logging Lists
    time_list = []
    steering_angle_list = []
    distance_list = []
    throttle_list = []
    speed_list = []

    try:
        while time.time() - start_time < SIMULATION_DURATION:
            world.tick()

            # Event Handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_c:
                        camera_transform_state = (
                            camera_transform_above 
                            if camera_transform_state == camera_transform_normal 
                            else camera_transform_normal
                        )
                        camera.set_transform(camera_transform_state)
                    elif event.key == pygame.K_ESCAPE:
                        raise KeyboardInterrupt

            # Vehicle State Computation
            velocity = vehicle.get_velocity()
            speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

            # Advanced Collision Detection
            if target_vehicle and target_vehicle.is_alive:
                distance_to_target = vehicle.get_location().distance(target_vehicle.get_location())
                
                # Enhanced Collision Avoidance
                if distance_to_target < 3.0:
                    print("[EMERGENCY] Potential collision detected! Emergency braking.")
                    control = carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0)
                    vehicle.apply_control(control)
                    continue

            # Compute Relative Velocity
            if target_vehicle and target_vehicle.is_alive:
                relative_velocity = (
                    vehicle.get_velocity().x - target_vehicle.get_velocity().x
                ) * 3.6  # Convert to km/h
            else:
                relative_velocity = 0

            # Advanced Time Gap Controller
            throttle_brake = time_gap_controller.compute(
                speed / 3.6, 
                radar_data['distance'], 
                relative_velocity
            )

            # Adaptive Speed Management
            speed_correction = (MAX_SPEED + MIN_SPEED) / 2
            if speed > MAX_SPEED:
                throttle_brake = min(throttle_brake, -0.3)
            elif speed < MIN_SPEED:
                throttle_brake = max(throttle_brake, 0.3)

            # Steering Computation for the ego vehicle
            steering = compute_steering_angle(vehicle, waypoints, steering_pid)

            # Vehicle Control Application for ego vehicle
            if throttle_brake < 0:
                control = carla.VehicleControl(
                    throttle=0.0, 
                    brake=abs(throttle_brake), 
                    steer=steering
                )
            else:
                control = carla.VehicleControl(
                    throttle=np.clip(throttle_brake, 0, 1), 
                    brake=0.0, 
                    steer=steering
                )

            vehicle.apply_control(control)

            # Compute steering for target vehicle using Stanley controller
            if target_vehicle and target_vehicle.is_alive:
                target_steering = compute_target_steering(target_vehicle, waypoints, steering_pid)

                # Apply the computed steering to the target vehicle
                control_target = carla.VehicleControl(
                    throttle=0.5,  # Set throttle to a constant or adjust as needed
                    brake=0.0,
                    steer=target_steering
                )
                target_vehicle.apply_control(control_target)

                # Ensure target vehicle velocity is set
                forward_vector = target_vehicle.get_transform().get_forward_vector()
                target_vehicle.set_target_velocity(carla.Vector3D(
                    x=forward_vector.x * target_speed / 3.6,
                    y=forward_vector.y * target_speed / 3.6,
                    z=0
                ))

            # Data Logging
            time_list.append(time.time() - start_time)
            steering_angle_list.append(np.degrees(steering))
            distance_list.append(radar_data['distance'])
            throttle_list.append(throttle_brake)
            speed_list.append(speed)

            pygame.display.flip()
            clock.tick(20)

    except KeyboardInterrupt:
        print("Simulation manually stopped.")

    finally:
        # Cleanup and Visualization
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)

        print("Destroying actors...")
        camera.destroy()
        radar.destroy()
        vehicle.destroy()
        if target_vehicle and target_vehicle.is_alive:
            target_vehicle.destroy()
        pygame.quit()
        print("Simulation stopped.")

        # Enhanced Plotting with Speed Information
        plt.figure(figsize=(15, 12))

        plt.subplot(4, 1, 1)
        plt.plot(time_list, steering_angle_list, label="Steering Angle (°)", color='b')
        plt.xlabel("Time (s)")
        plt.ylabel("Steering Angle (°)")
        plt.title("Steering Angle vs. Time")
        plt.grid(True)

        plt.subplot(4, 1, 2)
        plt.plot(time_list, speed_list, label="Vehicle Speed (km/h)", color='r')
        plt.axhline(y=MIN_SPEED, color='g', linestyle='--', label="Min Speed")
        plt.axhline(y=MAX_SPEED, color='orange', linestyle='--', label="Max Speed")
        plt.xlabel("Time (s)")
        plt.ylabel("Speed (km/h)")
        plt.title("Vehicle Speed vs. Time")
        plt.legend()
        plt.grid(True)

        plt.subplot(4, 1, 3)
        plt.plot(time_list, distance_list, label=f"Distance (m)", color='g')
        plt.axhline(y=target_distance, color='r', linestyle='--', label=f"Target ({target_distance} m)")
        plt.xlabel("Time (s)")
        plt.ylabel("Distance (m)")
        plt.title("Distance to Lead Vehicle vs. Time")
        plt.legend()
        plt.grid(True)

        plt.subplot(4, 1, 4)
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
        print(f"Detailed Simulation Error: {e}")
        import traceback
        traceback.print_exc()


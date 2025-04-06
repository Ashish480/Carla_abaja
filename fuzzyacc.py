import carla
import numpy as np
import time
import pygame
import random
import matplotlib.pyplot as plt
import math

# PID Controller Class
class PIDController:
    def __init__(self, kp, ki, kd, integral_limit=10, output_limit=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0.0
        self.integral = 0.0
        self.integral_limit = integral_limit
        self.output_limit = output_limit
        self.prev_output = 0.0

    def compute(self, error, dt):
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        
        output = (
            self.kp * error + 
            self.ki * self.integral + 
            self.kd * derivative
        )
        
        # Apply low-pass filter for smoother control
        output = np.clip(
            0.7 * self.prev_output + 0.3 * output,  # More aggressive filtering for smoother response
            -self.output_limit, 
            self.output_limit
        )
        
        self.prev_error = error
        self.prev_output = output
        return output

class AdaptiveTimeGapController:
    def __init__(self, target_distance=8.0, max_speed=30.0):
        # Fixed target distance of 8 meters (Tesla-like ACC)
        self.target_distance = target_distance
        # Hard speed limit of 30 km/h
        self.max_speed = max_speed
        
        # Improved PID parameters for smoother control
        self.pid = PIDController(
            kp=0.6,    # Increased for better distance tracking
            ki=0.15,   # Increased for eliminating steady-state error
            kd=0.25,   # Increased for better damping
            integral_limit=5.0, 
            output_limit=1.0
        )
        
        # Additional state variables for enhanced Tesla-like behavior
        self.prev_control = 0.0
        self.time_gap = 0.5  # Base time gap in seconds (used with speed)
        self.min_distance = target_distance  # Absolute minimum distance regardless of speed

    def compute(self, speed, distance_front, relative_velocity=0):
        # Convert speed to m/s for calculations (from km/h)
        speed_ms = speed / 3.6
        
        # We want exactly 8m gap so we use fixed value
        target = self.target_distance
        
        # Error is positive when we're too far, negative when too close
        distance_error = distance_front - target
        
        # Emergency braking - enhanced with relative velocity consideration
        if distance_front < 3.0 or (distance_front < 5.0 and relative_velocity > 5.0):
            return -1.0  # Maximum braking
        
        dt = 0.05  # Fixed time step
        control_output = self.pid.compute(distance_error, dt)
        
        # Speed limiting logic - Force braking when speed exceeds limit
        if speed > self.max_speed:
            # Apply proportional braking based on how much we exceed the limit
            overspeed_factor = min((speed - self.max_speed) / 5.0, 1.0)
            speed_control = -0.3 * overspeed_factor
            # Combine with distance control (prioritize speed limit)
            control_output = min(control_output, speed_control)
        
        # Smoothing the control output changes
        control_output = 0.8 * self.prev_control + 0.2 * control_output
        self.prev_control = control_output
        
        return np.clip(control_output, -1.0, 1.0)


# Lane Waypoints and Steering Functions
def get_lane_waypoints(world, distance=2.0):
    return [wp for wp in world.get_map().generate_waypoints(distance) if wp.lane_type == carla.LaneType.Driving]

def find_closest_waypoint(vehicle_location, waypoints):
    if not waypoints:
        return None
    return min(waypoints, key=lambda wp: vehicle_location.distance(wp.transform.location))

def compute_stanley_steering(vehicle, waypoints, pid_controller, k=0.2):
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
    pid_correction = pid_controller.compute(cte, dt)

    steering_angle = stanley_steering + pid_correction
    return np.clip(steering_angle, -np.radians(30), np.radians(30))

# Calculate actual distance between vehicles (bumper to bumper)
def calculate_actual_distance(ego_vehicle, target_vehicle):
    if not ego_vehicle or not target_vehicle or not ego_vehicle.is_alive or not target_vehicle.is_alive:
        return 100.0
    
    # Get vehicle locations
    ego_loc = ego_vehicle.get_location()
    target_loc = target_vehicle.get_location()
    
    # Get vehicle bounding boxes
    ego_bbox = ego_vehicle.bounding_box
    target_bbox = target_vehicle.bounding_box
    
    # Calculate center-to-center distance
    center_distance = ego_loc.distance(target_loc)
    
    # Calculate approximate vehicle lengths in the direction of travel
    ego_length = ego_bbox.extent.x * 2
    target_length = target_bbox.extent.x * 2
    
    # Calculate approximate bumper-to-bumper distance
    # Subtract half of each vehicle's length from the center distance
    bumper_distance = center_distance - (ego_length/2 + target_length/2)
    
    # Ensure we don't return negative distances
    return max(0.1, bumper_distance)

# Radar and Vehicle Spawning Functions
def radar_callback(data, target_vehicle=None, ego_vehicle=None):
    # Start with large value
    distance_front = 100
    
    # If vehicles aren't available, return default
    if target_vehicle is None or not target_vehicle.is_alive or ego_vehicle is None:
        return distance_front

    # Calculate direct inter-vehicle distance
    actual_distance = calculate_actual_distance(ego_vehicle, target_vehicle)
    
    # Process radar detections
    radar_distances = []
    for detection in data:
        if 0 < detection.depth < 100:
            radar_distances.append(detection.depth)

    # Use median of radar distances if available
    if radar_distances:
        radar_distance = np.median(radar_distances)
        # Blend radar and actual distance for stability
        distance_front = 0.7 * radar_distance + 0.3 * actual_distance
    else:
        distance_front = actual_distance

    return np.clip(distance_front, 0, 100)

def spawn_target_vehicle(world, vehicle_bp, spawn_transform, target_speed=20, target_distance=8.0):
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

    # Multiple spawn attempts with different strategies
    spawn_strategies = [
        lambda: spawn_from_suitable_points(world, vehicle_bp, suitable_points, spawn_transform.rotation),
        lambda: spawn_by_manual_positioning(world, vehicle_bp, spawn_location, forward_vector, target_distance),
        lambda: spawn_at_random_points(world, vehicle_bp, all_spawn_points)
    ]

    for strategy in spawn_strategies:
        target_vehicle = strategy()
        if target_vehicle is not None:
            break

    if target_vehicle is None:
        print("Failed to spawn target vehicle.")
        return None

    # Set initial target velocity - Between 20-30 km/h
    target_speed = min(target_speed, 30.0)
    waypoint = world.get_map().get_waypoint(target_vehicle.get_location())
    if waypoint:
        road_dir = waypoint.transform.get_forward_vector()
        target_vehicle.set_target_velocity(carla.Vector3D(
            x=road_dir.x * target_speed / 3.6,
            y=road_dir.y * target_speed / 3.6,
            z=0
        ))
    
    return target_vehicle

def spawn_from_suitable_points(world, vehicle_bp, suitable_points, rotation):
    for sp, dist in suitable_points[:10]:
        target_vehicle = world.try_spawn_actor(vehicle_bp, carla.Transform(sp.location, rotation))
        if target_vehicle is not None:
            print(f"Target vehicle spawned successfully at {sp.location}, distance: {dist:.2f}m")
            return target_vehicle
    return None

def spawn_by_manual_positioning(world, vehicle_bp, spawn_location, forward_vector, target_distance):
    for distance_multiplier in [1.0, 1.5, 2.0, 3.0, 5.0]:
        test_distance = target_distance * distance_multiplier
        target_location = carla.Location(
            x=spawn_location.x + forward_vector.x * test_distance,
            y=spawn_location.y + forward_vector.y * test_distance,
            z=spawn_location.z + 0.5
        )
        target_transform = carla.Transform(target_location)
        target_vehicle = world.try_spawn_actor(vehicle_bp, target_transform)
        if target_vehicle is not None:
            print(f"Target vehicle spawned via manual positioning at distance {test_distance:.2f}m")
            return target_vehicle
    return None

def spawn_at_random_points(world, vehicle_bp, all_spawn_points):
    random.shuffle(all_spawn_points)
    for sp in all_spawn_points[:20]:
        target_vehicle = world.try_spawn_actor(vehicle_bp, sp)
        if target_vehicle is not None:
            print(f"Target vehicle spawned at random location {sp.location}")
            return target_vehicle
    return None

def position_ego_behind_target(world, ego_vehicle, target_vehicle, distance=20.0):
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

# Camera and Display Functions
def process_img(image, display, vehicle, steering_angle, font, distance_front=100, target_distance=8.0):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))[:, :, :3][:, :, ::-1]
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

    velocity = vehicle.get_velocity()
    speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    speed = min(speed, 30)  # Display cap at 30 km/h

    text_surface_speed = font.render(f'Speed: {speed:.2f} km/h', True, (255, 255, 255))
    text_surface_steering = font.render(f'Steering: {np.degrees(steering_angle):.2f}°', True, (255, 255, 255))
    
    # Enhanced UI with color coding for distance
    distance_color = (
        (255, 0, 0) if distance_front < 5 else 
        (255, 255, 0) if distance_front < target_distance else 
        (0, 255, 0)
    )
    text_surface_distance = font.render(f'Current Distance: {distance_front:.2f} m', True, distance_color)
    target_dist_text = font.render(f'Target Distance: {target_distance:.1f} m', True, (255, 255, 255))

    display.blit(surface, (0, 0))
    display.blit(text_surface_speed, (10, 20))
    display.blit(text_surface_steering, (10, 50))
    display.blit(text_surface_distance, (10, 80))
    display.blit(target_dist_text, (10, 110))

def main():
    # Simulation Parameters
    target_distance = 8.0  # Fixed 8-meter gap (Tesla-like ACC)
    MIN_SPEED = 20.0
    MAX_SPEED = 30.0  # Hard speed limit of 30 km/h
    SIMULATION_DURATION = 180

    # PID Controllers - Tuned for smoother operation
    steering_pid = PIDController(
        kp=0.15, 
        ki=0.02, 
        kd=0.08, 
        integral_limit=7
    )
    
    # Improved Tesla-like ACC controller
    time_gap_controller = AdaptiveTimeGapController(
        target_distance=target_distance,  # Fixed 8-meter gap
        max_speed=MAX_SPEED  # Maximum speed of 30 km/h
    )

    # CARLA Setup
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

    # Target Vehicle Spawning - Initial speed within 20-30 km/h range
    initial_target_speed = random.uniform(20.0, 30.0)
    target_vehicle = spawn_target_vehicle(
        world, 
        blueprint_library.find('vehicle.tesla.model3'), 
        vehicle.get_transform(), 
        target_speed=initial_target_speed, 
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
    pygame.display.set_caption("Tesla-like ACC Simulation")
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

    # Camera Setup with Multiple Views
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
    last_speed_change_time = start_time

    # Data Logging Lists
    time_list, steering_angle_list, distance_list, throttle_list, speed_list, target_speed_list = [], [], [], [], [], []

    # Variable for target vehicle's current speed (varies between 20-30 km/h)
    current_target_speed = initial_target_speed
    
    try:
        while time.time() - start_time < SIMULATION_DURATION:
            world.tick()

            # Event Handling with Camera Toggle
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

            # Vary target vehicle speed between 20-30 km/h every few seconds
            current_time = time.time()
            if current_time - last_speed_change_time > random.uniform(3.0, 7.0):  # Change speed every 3-7 seconds
                # New target speed between 20-30 km/h
                current_target_speed = random.uniform(MIN_SPEED, MAX_SPEED)
                last_speed_change_time = current_time
                print(f"Target vehicle changing speed to: {current_target_speed:.2f} km/h")

            # Vehicle State Computation
            velocity = vehicle.get_velocity()
            speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

            # Improved Collision Detection and Emergency Braking
            if target_vehicle and target_vehicle.is_alive:
                distance_to_target = calculate_actual_distance(vehicle, target_vehicle)
                
                if distance_to_target < 3.0:
                    print("[EMERGENCY] Potential collision detected! Emergency braking.")
                    control = carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0)
                    vehicle.apply_control(control)
                    continue

            # Compute Relative Velocity with Enhanced Logic
            if target_vehicle and target_vehicle.is_alive:
                ego_vel = vehicle.get_velocity()
                target_vel = target_vehicle.get_velocity()
                
                # Project velocities onto the forward vector for proper relative velocity
                vehicle_forward = vehicle.get_transform().get_forward_vector()
                ego_speed = (ego_vel.x * vehicle_forward.x + ego_vel.y * vehicle_forward.y) * 3.6
                target_speed = (target_vel.x * vehicle_forward.x + target_vel.y * vehicle_forward.y) * 3.6
                
                relative_velocity = ego_speed - target_speed
            else:
                relative_velocity = 0

            # Strict Speed Limiting - Force this before applying ACC
            if speed > MAX_SPEED:
                # Apply stronger brake when exceeding the limit
                over_limit = (speed - MAX_SPEED) / 10.0  # Normalized overspeed
                brake_strength = min(0.3 + over_limit, 0.8)  # Increase braking with overspeed
                
                control = carla.VehicleControl(throttle=0.0, brake=brake_strength, steer=steering)
                vehicle.apply_control(control)
                
                # Skip ACC calculation for this step
                continue
            
            # Apply the Tesla-like Adaptive Cruise Control with fixed 8m gap
            throttle_brake = time_gap_controller.compute(
                speed, 
                radar_data['distance'], 
                relative_velocity
            )

            # Hard Speed Cap at 30 km/h
            if speed >= MAX_SPEED:
                throttle_brake = min(throttle_brake, -0.3)  # Force braking

            # Steering Computation
            steering = compute_stanley_steering(vehicle, waypoints, steering_pid)

            # Vehicle Control Application
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

            # Target Vehicle Control - Variable speed between 20-30 km/h
            if target_vehicle and target_vehicle.is_alive:
                target_steering = compute_stanley_steering(target_vehicle, waypoints, steering_pid)
                
                # Get current speed of target vehicle
                target_vel = target_vehicle.get_velocity()
                target_current_speed = 3.6 * np.sqrt(target_vel.x**2 + target_vel.y**2 + target_vel.z**2)
                
                # PID control for target vehicle to reach its variable target speed
                target_speed_error = current_target_speed - target_current_speed
                target_throttle = np.clip(0.5 + target_speed_error * 0.05, 0, 0.8)
                target_brake = 0.0
                
                # Apply braking if going too fast
                if target_current_speed > current_target_speed + 2.0:
                    target_brake = 0.3
                    target_throttle = 0.0
                
                control_target = carla.VehicleControl(
                    throttle=target_throttle,
                    brake=target_brake,
                    steer=target_steering
                )
                target_vehicle.apply_control(control_target)

                # Set target velocity in the direction of travel
                forward_vector = target_vehicle.get_transform().get_forward_vector()
                target_vehicle.set_target_velocity(carla.Vector3D(
                    x=forward_vector.x * current_target_speed / 3.6,
                    y=forward_vector.y * current_target_speed / 3.6,
                    z=0
                ))

            # Data Logging
            time_list.append(time.time() - start_time)
            steering_angle_list.append(np.degrees(steering))
            distance_list.append(radar_data['distance'])
            throttle_list.append(throttle_brake)
            speed_list.append(speed)
            
            # Log target vehicle speed if available
            if target_vehicle and target_vehicle.is_alive:
                target_vel = target_vehicle.get_velocity()
                target_speed = 3.6 * np.sqrt(target_vel.x**2 + target_vel.y**2 + target_vel.z**2)
                target_speed_list.append(target_speed)
            else:
                target_speed_list.append(0)

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

        # Enhanced Plotting with Target Vehicle Speed
        plt.figure(figsize=(15, 15))

        plt.subplot(5, 1, 1)
        plt.plot(time_list, steering_angle_list, label="Steering Angle (°)", color='b')
        plt.xlabel("Time (s)")
        plt.ylabel("Steering Angle (°)")
        plt.title("Steering Angle vs. Time")
        plt.grid(True)

        plt.subplot(5, 1, 2)
        plt.plot(time_list, speed_list, label="Ego Vehicle Speed (km/h)", color='r')
        plt.plot(time_list, target_speed_list, label="Target Vehicle Speed (km/h)", color='g', linestyle='-')
        plt.axhline(y=MIN_SPEED, color='c', linestyle='--', label="Min Speed (20 km/h)")
        plt.axhline(y=MAX_SPEED, color='orange', linestyle='--', label="Max Speed (30 km/h)")
        plt.xlabel("Time (s)")
        plt.ylabel("Speed (km/h)")
        plt.title("Vehicle Speeds vs. Time")
        plt.legend()
        plt.grid(True)

        plt.subplot(5, 1, 3)
        plt.plot(time_list, distance_list, label=f"Distance (m)", color='g')
        plt.axhline(y=target_distance, color='r', linestyle='--', label=f"Target Gap (8 m)")
        plt.xlabel("Time (s)")
        plt.ylabel("Distance (m)")
        plt.title("Distance to Lead Vehicle vs. Time")
        plt.legend()
        plt.grid(True)

        plt.subplot(5, 1, 4)
        plt.plot(time_list, throttle_list, label="Throttle/Brake Control", color='purple')
        plt.axhline(y=0, color='k', linestyle='-')
        plt.xlabel("Time (s)")
        plt.ylabel("Control Value")
        plt.title("Throttle/Brake Control vs. Time")
        plt.legend()
        plt.grid(True)
        
        # Speed Difference Plot
        plt.subplot(5, 1, 5)
        speed_diff = [ego - target for ego, target in zip(speed_list, target_speed_list)]
        plt.plot(time_list, speed_diff, label="Speed Difference (Ego - Target)", color='darkred')
        plt.axhline(y=0, color='k', linestyle='-')
        plt.xlabel("Time (s)")
        plt.ylabel("Speed Difference (km/h)")
        plt.title("Speed Difference vs. Time")
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

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

# Time Gap Controller (Tesla-style ACC)
def time_gap_controller(speed, distance_front, time_gap=1.5, min_gap=5.0):
    desired_distance = max(time_gap * speed, min_gap)
    distance_error = distance_front - desired_distance

    if distance_front < 2.5:
        return -1.0  # Emergency brake

    # PID-like controller
    kp = 0.5
    ki = 0.02
    kd = 0.1
    if not hasattr(time_gap_controller, "prev_error"):
        time_gap_controller.prev_error = 0.0
        time_gap_controller.integral = 0.0

    dt = 0.05
    time_gap_controller.integral += distance_error * dt
    derivative = (distance_error - time_gap_controller.prev_error) / dt
    output = kp * distance_error + ki * time_gap_controller.integral + kd * derivative
    time_gap_controller.prev_error = distance_error

    return np.clip(output, -1.0, 1.0)

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
    global distance_front

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
    Uses a more robust approach to find valid spawn positions.
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
    
# Main simulation loop
def main():
    global distance_front
    distance_front = 100
    target_distance = 8.0

    steering_pid = PIDController(kp=0.1, ki=0.01, kd=0.05, integral_limit=5)

    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find('vehicle.tesla.model3')

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

    target_vehicle = spawn_target_vehicle(world, blueprint_library.find('vehicle.tesla.model3'), 
                                          vehicle.get_transform(), target_speed=20, 
                                          target_distance=target_distance)

    if target_vehicle:
        for _ in range(10):
            world.tick()
        position_ego_behind_target(world, vehicle, target_vehicle, distance=20.0)

    waypoints = get_lane_waypoints(world, distance=2.0)

    pygame.init()
    display = pygame.display.set_mode((800, 600), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("CARLA ACC Simulation - 8m Target")
    font = pygame.font.Font(None, 36)

    radar_bp = blueprint_library.find('sensor.other.radar')
    radar_bp.set_attribute('horizontal_fov', '45')
    radar_bp.set_attribute('vertical_fov', '20')
    radar_bp.set_attribute('range', '100')
    radar_bp.set_attribute('points_per_second', '2000')

    radar_transform = carla.Transform(carla.Location(x=2, y=0, z=1.5))
    radar = world.spawn_actor(radar_bp, radar_transform, attach_to=vehicle)
    radar.listen(lambda data: radar_callback(data, target_vehicle, vehicle))

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
        process_img(image, display, vehicle, steering, font, distance_front, target_distance)

    camera.listen(camera_callback)

    clock = pygame.time.Clock()
    start_time = time.time()

    time_list = []
    steering_angle_list = []
    distance_list = []
    throttle_list = []

    try:
        while time.time() - start_time < 120:
            world.tick()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_c:
                        camera_transform_state = camera_transform_above if camera_transform_state == camera_transform_normal else camera_transform_normal
                        camera.set_transform(camera_transform_state)
                    elif event.key == pygame.K_ESCAPE:
                        raise KeyboardInterrupt

            velocity = vehicle.get_velocity()
            speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)

            # Use Tesla-style Time Gap Controller
            throttle_brake = time_gap_controller(speed / 3.6, distance_front)

            if speed < 1.0 and distance_front > target_distance + 3:
                throttle_brake = 0.3
                print("[INFO] Startup boost applied.")

            steering = compute_steering_angle(vehicle, waypoints, steering_pid)

            if throttle_brake < 0:
                control = carla.VehicleControl(throttle=0.0, brake=abs(throttle_brake), steer=steering)
            else:
                control = carla.VehicleControl(throttle=np.clip(throttle_brake, 0, 1), brake=0.0, steer=steering)

            vehicle.apply_control(control)

            if target_vehicle and target_vehicle.is_alive:
                if random.random() < 0.01:
                    forward_vector = target_vehicle.get_transform().get_forward_vector()
                    new_speed = 29
                    target_vehicle.set_target_velocity(carla.Vector3D(
                        x=forward_vector.x * new_speed / 3.6,
                        y=forward_vector.y * new_speed / 3.6,
                        z=0
                    ))

            time_list.append(time.time() - start_time)
            steering_angle_list.append(np.degrees(steering))
            distance_list.append(distance_front)
            throttle_list.append(throttle_brake)

            pygame.display.flip()
            clock.tick(20)

    except KeyboardInterrupt:
        print("Simulation manually stopped.")

    finally:
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



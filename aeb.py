import carla
import numpy as np
import time
import pygame
import random
import matplotlib.pyplot as plt

class PIDController:
    def __init__(self, kp, ki, kd, integral_limit=10, output_limits=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0.0
        self.integral = 0.0
        self.integral_limit = integral_limit
        self.output_limits = output_limits  # (min_output, max_output)

    def compute(self, error, dt):
        # Prevent division by zero
        if dt <= 0:
            dt = 0.01
            
        # Calculate integral with anti-windup
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        
        # Calculate derivative with low-pass filtering to reduce noise
        derivative = (error - self.prev_error) / dt
        
        # Compute PID output
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        # Apply output limits if specified
        if self.output_limits:
            output = np.clip(output, self.output_limits[0], self.output_limits[1])
            
        self.prev_error = error
        return output
    
    def reset(self):
        """Reset the PID controller state"""
        self.prev_error = 0.0
        self.integral = 0.0

def get_lane_waypoints(world, distance=2.0):
    return [wp for wp in world.get_map().generate_waypoints(distance) if wp.lane_type == carla.LaneType.Driving]

def find_closest_waypoint(vehicle_location, waypoints):
    return min(waypoints, key=lambda wp: vehicle_location.distance(wp.transform.location))

def compute_steering_angle(vehicle, waypoints, pid_controller, k=0.2):
    vehicle_transform = vehicle.get_transform()
    vehicle_location = vehicle_transform.location
    vehicle_yaw = np.radians(vehicle_transform.rotation.yaw)

    closest_waypoint = find_closest_waypoint(vehicle_location, waypoints)
    if closest_waypoint is None:
        return 0.0  

    # Calculate cross track error (CTE)
    dx = closest_waypoint.transform.location.x - vehicle_location.x
    dy = closest_waypoint.transform.location.y - vehicle_location.y
    cte = dy * np.cos(vehicle_yaw) - dx * np.sin(vehicle_yaw)

    # Calculate heading error
    path_yaw = np.radians(closest_waypoint.transform.rotation.yaw)
    heading_error = np.arctan2(np.sin(path_yaw - vehicle_yaw), np.cos(path_yaw - vehicle_yaw))

    # Calculate vehicle speed for Stanley controller
    vehicle_velocity = vehicle.get_velocity()
    speed = max(3.6 * np.sqrt(vehicle_velocity.x**2 + vehicle_velocity.y**2 + vehicle_velocity.z**2), 1.0)  # Minimum 1.0 to avoid division by zero
    
    # Stanley controller component
    stanley_steering = heading_error + np.arctan(k * cte / speed)

    # PID controller correction
    dt = 0.05  # Simulation time step
    pid_correction = pid_controller.compute(cte, dt)

    # Combine Stanley and PID controllers
    steering_angle = stanley_steering + pid_correction
    
    # Limit steering angle to realistic values
    steering_angle = np.clip(steering_angle, -np.radians(30), np.radians(30))

    return steering_angle

def linear_target_speed(distance, max_distance, max_speed):
    """
    Calculate target speed using a linear profile.
    At distance=0, speed=0. At distance=max_distance, speed=max_speed.
    This creates a gradual acceleration that reaches the target speed exactly at the end of the zone.
    """
    # Ensure distance doesn't exceed max_distance to avoid overshooting the target speed
    clamped_distance = min(distance, max_distance)
    # Linear interpolation: speed increases proportionally with distance
    return (clamped_distance / max_distance) * max_speed

def process_img(image, display, vehicle, steering_angle, font, distance_covered, speed, zone, target_speed=None):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))[:, :, :3][:, :, ::-1]
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    
    # Draw telemetry information
    text_surface_speed = font.render(f'Speed: {speed:.2f} km/h', True, (255, 255, 255))
    text_surface_steering = font.render(f'Steering: {np.degrees(steering_angle):.2f}°', True, (255, 255, 255))
    text_surface_distance = font.render(f'Distance: {distance_covered:.2f} m', True, (255, 255, 255))
    text_surface_zone = font.render(f'Zone: {zone}', True, (255, 255, 255))
    
    display.blit(surface, (0, 0))
    display.blit(text_surface_speed, (10, 20))
    display.blit(text_surface_steering, (10, 50))
    display.blit(text_surface_distance, (10, 80))
    display.blit(text_surface_zone, (10, 110))
    
    # If target speed is provided, display it
    if target_speed is not None:
        text_surface_target = font.render(f'Target: {target_speed:.2f} km/h', True, (255, 255, 0))
        display.blit(text_surface_target, (10, 140))

def main():
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # Set fixed delta seconds for deterministic physics
    settings = world.get_settings()
    settings.fixed_delta_seconds = 0.05  # 20 FPS
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

    # Adding a 2-second delay after the vehicle spawns to ensure physics are stabilized
    time.sleep(2)

    # Get waypoints for following the lane
    waypoints = get_lane_waypoints(world, distance=2.0)

    # Create enhanced PID controller with output limits for steering
    steering_pid = PIDController(
        kp=0.3,      # Proportional gain for responsiveness
        ki=0.01,     # Small integral gain to address steady-state errors
        kd=0.1,      # Derivative gain for damping oscillations
        integral_limit=5.0,
        output_limits=(-np.radians(10), np.radians(10))
    )

    # Create a PID controller for speed control
    speed_pid = PIDController(
        kp=0.8,       # Increased for better tracking of the linear speed profile
        ki=0.08,      # Slight increase for reducing steady-state error
        kd=0.05,      # Small derivative term to prevent overshoot
        integral_limit=3.0,
        output_limits=(0.0, 1.0)  # Throttle limits
    )

    pygame.init()
    display = pygame.display.set_mode((800, 600), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("CARLA AEB Simulation")
    font = pygame.font.Font(None, 36)

    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute("image_size_x", "800")
    camera_bp.set_attribute("image_size_y", "600")
    camera_bp.set_attribute("fov", "110")

    camera_transform = carla.Transform(carla.Location(x=0, y=0, z=5), carla.Rotation(pitch=-90))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    steering = 0.0
    distance_covered = 0.0
    current_speed = 0.0
    current_zone = "Start"
    target_speed = 0.0

    # Create a lambda function to process the camera image
    camera.listen(lambda image: process_img(image, display, vehicle, steering, font, distance_covered, current_speed, current_zone, target_speed))

    clock = pygame.time.Clock()
    start_time = time.time()

    # Ensure vehicle starts from a complete stop and apply initial throttle
    vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))  # Apply brake to stop
    time.sleep(0.5)  # Allow time for vehicle to settle

    # Release brake and apply initial throttle to overcome inertia
    vehicle.apply_control(carla.VehicleControl(throttle=0.3, brake=0.0))  # Apply some throttle
    time.sleep(0.5)  # Give it a moment with steady throttle

    # Gradual increase in throttle if the vehicle is not moving
    # Keep applying throttle until the vehicle reaches a minimum speed
    while current_speed < 1.0:  # Minimum speed threshold
        vehicle.apply_control(carla.VehicleControl(throttle=0.5, brake=0.0))  # Increase throttle if needed
        time.sleep(0.2)  # Wait before checking speed again
        vehicle_velocity = vehicle.get_velocity()
        current_speed = 3.6 * np.sqrt(vehicle_velocity.x**2 + vehicle_velocity.y**2 + vehicle_velocity.z**2)  # Speed in km/h

    # Reset speed PID controller to avoid initial integral windup
    speed_pid.reset()

    # Data collection for visualization
    time_points = []
    speed_points = []
    target_speed_points = []
    distance_points = []
    throttle_points = []
    brake_points = []

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt

            # Get the current speed of the vehicle
            vehicle_velocity = vehicle.get_velocity()
            current_speed = 3.6 * np.sqrt(vehicle_velocity.x**2 + vehicle_velocity.y**2 + vehicle_velocity.z**2)

            # Update distance covered based on the current speed
            dt = 0.05  # Simulation time step in seconds
            distance_covered += current_speed * dt / 3.6  # Speed * Time in hours converted to meters

            # Default control values
            throttle = 0.0
            brake = 0.0

            # Determine current zone and apply appropriate control strategy
            if distance_covered < 30:  # Zone 1: Linear acceleration to 30 km/h
                current_zone = "Zone 1: Acceleration"

                # Calculate target speed based on linear profile (0 km/h at 0m, 30 km/h at 30m)
                target_speed = linear_target_speed(distance_covered, 30.0, 30.0)

                # Use PID controller to track the target speed
                speed_error = target_speed - current_speed
                throttle = speed_pid.compute(speed_error, dt)

            elif 30 <= distance_covered < 40:  # Zone 2: Deceleration
                current_zone = "Zone 2: Deceleration"
 
                # Linear deceleration from 30 km/h at 30m to 0 km/h at 40m
                progress = (distance_covered - 30.0) / 10.0  # Progress from 0 to 1 in the deceleration zone
                target_speed = 30.0 * (1.0 - progress)  # Target speed decreases linearly

                # Calculate the desired deceleration range
                target_deceleration = np.random.uniform(5.0, 8.0)  # Random value between 5 and 8 m/s²
    
                # Convert target speed from km/h to m/s
                target_speed_mps = target_speed / 3.6

                # Calculate the deceleration required to reach the target speed
                speed_difference = current_speed - target_speed
                if speed_difference > 0:
                    brake_force = min(speed_difference / target_deceleration, 1.0)  # Normalized brake force
                    throttle = 0.0
                else:
                    brake_force = 0.0
                    throttle = max(0, speed_pid.compute(target_speed - current_speed, dt))

                brake = brake_force



            else:  # Zone 3: Full stop and hold
                current_zone = "Zone 3: Stopped"
                target_speed = 0.0
                throttle = 0.0

                # Apply strong braking initially, then lighter braking to hold position
                if current_speed > 1.0:
                    brake = 1.0  # Full brake to stop quickly
                else:
                    brake = 0.3  # Light braking to hold position once stopped

            # Update steering using Stanley controller with PID correction
            steering = compute_steering_angle(vehicle, waypoints, steering_pid)

            # Apply vehicle control
            vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steering, brake=brake))

            # Collect data for visualization
            current_time = time.time() - start_time
            time_points.append(current_time)
            speed_points.append(current_speed)
            target_speed_points.append(target_speed)
            distance_points.append(distance_covered)
            throttle_points.append(throttle)
            brake_points.append(brake)

            pygame.display.flip()
            clock.tick(20)  # Match the physics update rate

            # Exit condition - stop simulation after vehicle has stopped in zone 3
            if distance_covered >= 40 and current_speed < 0.5:
                time.sleep(2)  # Wait 2 seconds after stopping
                break

    except KeyboardInterrupt:
        print("Simulation manually stopped.")

    finally:
        # Generate visualization graph
        plt.figure(figsize=(12, 10))

        plt.subplot(4, 1, 1)
        plt.plot(time_points, speed_points, 'b-', label='Actual Speed (km/h)')
        plt.plot(time_points, target_speed_points, 'r--', label='Target Speed (km/h)')
        plt.grid(True)
        plt.legend()
        plt.title('AEB Simulation with Linear Speed Profile')
        plt.ylabel('Speed (km/h)')

        plt.subplot(4, 1, 2)
        plt.plot(distance_points, speed_points, 'b-', label='Speed vs Distance')
        plt.axvline(x=30, color='r', linestyle='--', label='Zone 1-2 Boundary')
        plt.axvline(x=40, color='m', linestyle='--', label='Zone 2-3 Boundary')
        plt.grid(True)
        plt.legend()
        plt.xlabel('Distance (m)')
        plt.ylabel('Speed (km/h)')

        plt.subplot(4, 1, 3)
        plt.plot(time_points, distance_points, 'g-', label='Distance (m)')
        plt.axhline(y=30, color='r', linestyle='--', label='Zone 1 End')
        plt.axhline(y=40, color='m', linestyle='--', label='Zone 2 End')
        plt.grid(True)
        plt.legend()
        plt.ylabel('Distance (m)')

        plt.subplot(4, 1, 4)
        plt.plot(time_points, throttle_points, 'g-', label='Throttle')
        plt.plot(time_points, brake_points, 'r-', label='Brake')
        plt.grid(True)
        plt.legend()
        plt.xlabel('Time (s)')
        plt.ylabel('Control Input')

        plt.tight_layout()
        plt.savefig('aeb_simulation_results.png')
        plt.show()

        print("Destroying actors...")
        camera.destroy()
        vehicle.destroy()
        pygame.quit()
        print("Simulation stopped.")
        print("Results saved to 'aeb_simulation_results.png'")

if __name__ == '__main__':
    main()


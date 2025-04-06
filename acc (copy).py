import carla
import time
import numpy as np
import pygame
import random

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("CARLA ACC Simulation")
clock = pygame.time.Clock()

# PID Controller Class
class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0

    def control(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return np.clip(output, -1, 1)  # Ensure control values are in [-1,1]

# Connect to CARLA
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
blueprint_library = world.get_blueprint_library()

# Spawn Ego Vehicle (Follower)
vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
spawn_points = world.get_map().get_spawn_points()
ego_vehicle = None

for spawn_point in spawn_points:
    ego_vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    if ego_vehicle:
        print("Spawned Ego Vehicle Successfully")
        break

if not ego_vehicle:
    raise RuntimeError("No available spawn points for Ego Vehicle.")

# Spawn Lead Vehicle (Target)
lead_vehicle = None
for spawn_point in spawn_points:
    lead_vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
    if lead_vehicle:
        lead_vehicle.set_autopilot(True)
        print("Spawned Lead Vehicle Successfully")
        break

if not lead_vehicle:
    raise RuntimeError("No available spawn points for Lead Vehicle.")

# Attach Camera to Ego Vehicle
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '800')
camera_bp.set_attribute('image_size_y', '600')
camera_bp.set_attribute('fov', '110')
camera_transform = carla.Transform(carla.Location(x=-5, z=2))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)

image_surface = None

def process_image(image):
    global image_surface
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = array.reshape((image.height, image.width, 4))
    array = array[:, :, :3][:, :, ::-1]  # Convert BGRA to RGB
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

camera.listen(lambda image: process_image(image))

# ACC Parameters
desired_distance = 8  # Desired gap in meters
speed_controller = PIDController(0.5, 0.01, 0.2)
steering_controller = PIDController(0.8, 0.02, 0.1)

# Sensor for Distance Measurement
sensor_bp = blueprint_library.find('sensor.other.radar')
sensor_transform = carla.Transform(carla.Location(x=2.5, z=1.0))
radar = world.spawn_actor(sensor_bp, sensor_transform, attach_to=ego_vehicle)

# Main ACC Loop
try:
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        ego_transform = ego_vehicle.get_transform()
        ego_velocity = ego_vehicle.get_velocity()
        ego_speed = np.linalg.norm([ego_velocity.x, ego_velocity.y])

        lead_transform = lead_vehicle.get_transform()
        lead_velocity = lead_vehicle.get_velocity()
        lead_speed = np.linalg.norm([lead_velocity.x, lead_velocity.y])
        
        # Calculate Distance
        distance = ego_transform.location.distance(lead_transform.location)
        
        # Compute Speed Error (Gap Control)
        speed_error = lead_speed - ego_speed + (distance - desired_distance)
        throttle_brake = speed_controller.control(speed_error, 0.05)
        
        # Compute Steering to Follow Lead Vehicle
        angle_error = np.deg2rad(lead_transform.rotation.yaw - ego_transform.rotation.yaw)
        steering = steering_controller.control(angle_error, 0.05)
        
        # Apply Control
        control = carla.VehicleControl()
        if throttle_brake > 0:
            control.throttle = throttle_brake
            control.brake = 0
        else:
            control.throttle = 0
            control.brake = -throttle_brake
        control.steer = steering
        ego_vehicle.apply_control(control)
        
        # Pygame Display
        screen.fill((0, 0, 0))
        if image_surface:
            screen.blit(image_surface, (0, 0))
        font = pygame.font.Font(None, 36)
        text = font.render(f"Distance: {distance:.2f}m  Speed: {ego_speed:.2f}m/s", True, (255, 255, 255))
        screen.blit(text, (20, 20))
        pygame.display.flip()
        
        clock.tick(20)
        time.sleep(0.05)

except KeyboardInterrupt:
    print("Stopping ACC...")
finally:
    ego_vehicle.destroy()
    lead_vehicle.destroy()
    radar.destroy()
    camera.destroy()
    pygame.quit()


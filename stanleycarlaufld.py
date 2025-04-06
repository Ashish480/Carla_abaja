import os
import cv2
import carla
import numpy as np
import time
import pygame
import random
import matplotlib.pyplot as plt

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

def compute_steering_angle(vehicle, pid_controller, k=0.2):
    vehicle_transform = vehicle.get_transform()
    vehicle_location = vehicle_transform.location
    vehicle_yaw = np.radians(vehicle_transform.rotation.yaw)

    # Wait until lane points are available
    while not os.path.exists("/tmp/lane_points.npy"):
        print("Waiting for lane points...")
        time.sleep(0.1)

    lane_points = np.load("/tmp/lane_points.npy", allow_pickle=True)
    if len(lane_points) == 0:
        print("No lane points detected, keeping steering straight.")
        return 0.0

    # Find the closest lane point
    closest_point = min(lane_points, key=lambda p: np.linalg.norm([p[0] - vehicle_location.x, p[1] - vehicle_location.y]))
    dx = closest_point[0] - vehicle_location.x
    dy = closest_point[1] - vehicle_location.y
    cte = dy * np.cos(vehicle_yaw) - dx * np.sin(vehicle_yaw)

    # Compute heading error
    heading_error = np.arctan2(dy, dx) - vehicle_yaw

    # Get vehicle speed
    vehicle_velocity = vehicle.get_velocity()
    speed = max(3.6 * np.sqrt(vehicle_velocity.x**2 + vehicle_velocity.y**2 + vehicle_velocity.z**2), 5)

    # Apply Stanley controller with PID correction
    stanley_steering = heading_error + np.arctan(k * cte / (speed + 1e-3))
    pid_correction = np.clip(pid_controller.compute(cte, 0.05), -np.radians(10), np.radians(10))

    final_steering = np.clip(stanley_steering + pid_correction, -np.radians(30), np.radians(30))

    print(f"CTE: {cte:.2f}, Heading Error: {np.degrees(heading_error):.2f}, PID Correction: {np.degrees(pid_correction):.2f}, Final Steering: {np.degrees(final_steering):.2f}")

    return final_steering

def process_img(image, display, vehicle, steering_angle, font):
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))[:, :, :3][:, :, ::-1]

    # Save the image to pass to UFLD
    cv2.imwrite("/tmp/carla_frame.jpg", array)

    velocity = vehicle.get_velocity()
    speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + vehicle_velocity.z**2)

    text_surface_speed = font.render(f'Speed: {speed:.2f} km/h', True, (255, 255, 255))
    text_surface_steering = font.render(f'Steering: {np.degrees(steering_angle):.2f}°', True, (255, 255, 255))

    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    display.blit(surface, (0, 0))
    display.blit(text_surface_speed, (10, 20))
    display.blit(text_surface_steering, (10, 50))

    pygame.display.update()  # Ensures display updates properly

def main():
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find('vehicle.tesla.model3')

    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    vehicle = None
    for sp in spawn_points:
        vehicle = world.try_spawn_actor(vehicle_bp, sp)
        if vehicle:
            print(f"Vehicle spawned at: {sp.location}")
            break
    
    if not vehicle:
        print("Failed to find a suitable spawn point.")
        return

    pid_controller = PIDController(kp=0.1, ki=0.01, kd=0.05, integral_limit=5)

    pygame.init()
    display = pygame.display.set_mode((800, 600), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("CARLA Vehicle Camera")
    font = pygame.font.Font(None, 36)

    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute("image_size_x", "800")
    camera_bp.set_attribute("image_size_y", "600")
    camera_bp.set_attribute("fov", "110")

    # Camera placement adjusted for better lane detection
    camera_transform = carla.Transform(carla.Location(x=1.5, y=0, z=2.0), carla.Rotation(pitch=-15))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    steering = 0.0
    camera.listen(lambda image: process_img(image, display, vehicle, steering, font))

    clock = pygame.time.Clock()
    start_time = time.time()

    time_list = []
    steering_angle_list = []

    try:
        while time.time() - start_time < 30:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt
            
            steering = compute_steering_angle(vehicle, pid_controller)
            vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer=steering))

            time_list.append(time.time() - start_time)
            steering_angle_list.append(np.degrees(steering))

            pygame.display.flip()
            clock.tick(30)
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("Simulation manually stopped.")

    finally:
        print("Destroying actors...")
        camera.destroy()
        vehicle.destroy()
        pygame.quit()
        print("Simulation stopped.")

        # Plot Steering Angle vs. Time
        plt.figure(figsize=(8, 5))
        plt.plot(time_list, steering_angle_list, label="Steering Angle", color='b')
        plt.xlabel("Time (s)")
        plt.ylabel("Steering Angle (°)")
        plt.title("Steering Angle vs. Time")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    main()


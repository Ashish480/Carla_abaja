import carla
import numpy as np
import time
import pygame
import random
import cv2
import os
from ufld import detect_lanes

class PIDController:
    def __init__(self, kp, ki, kd, integral_limit=3):
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

def compute_steering_angle(vehicle, lane_points, pid_controller, k=0.2):  # Increased gain for better corrections
    """
    Compute Stanley steering using the detected lane points.
    """
    if vehicle is None:
        print("⚠️ Vehicle not found! Steering neutral.")
        return 0.0  

    vehicle_transform = vehicle.get_transform()
    vehicle_location = vehicle_transform.location
    vehicle_yaw = np.radians(vehicle_transform.rotation.yaw)

    # ✅ Increase selection range to 40m for better lane detection
    filtered_points = [p for p in lane_points if np.hypot(p[0] - vehicle_location.x, p[1] - vehicle_location.y) < 40]

    if not filtered_points:
        print("⚠️ No valid lane points found! Reducing steering gradually.")
        return vehicle.get_control().steer * 0.9  # ✅ Reduce steering instead of keeping it neutral

    # ✅ Prioritize lane points in front of the vehicle
    filtered_points = [p for p in filtered_points if p[0] > vehicle_location.x]

    if not filtered_points:
        print("⚠️ No forward lane points found! Keeping steering neutral.")
        return 0.0

    # ✅ Select the closest valid lane point in front of the vehicle
    closest_point = min(filtered_points, key=lambda p: np.hypot(p[0] - vehicle_location.x, p[1] - vehicle_location.y))

    dx = closest_point[0] - vehicle_location.x
    dy = closest_point[1] - vehicle_location.y
    cte = dy * np.cos(vehicle_yaw) - dx * np.sin(vehicle_yaw)

    cte = np.clip(cte, -10, 10)  # ✅ Allow a slightly larger range for better steering

    lane_heading = np.arctan2(dy, dx)
    heading_error = np.arctan2(np.sin(lane_heading - vehicle_yaw), np.cos(lane_heading - vehicle_yaw))
    heading_error = np.clip(heading_error, -np.radians(8), np.radians(8))  # ✅ Allow a bit more steering response

    vehicle_velocity = vehicle.get_velocity()
    speed = max(3.6 * np.sqrt(vehicle_velocity.x**2 + vehicle_velocity.y**2 + vehicle_velocity.z**2), 5)

    stanley_steering = heading_error + np.arctan(k * cte / speed)

    dt = 0.05  
    pid_correction = np.clip(pid_controller.compute(cte, dt), -np.radians(5), np.radians(5))  # ✅ Increase PID correction

    steering_angle = stanley_steering + pid_correction
    steering_angle = np.clip(steering_angle, -np.radians(20), np.radians(20))  # ✅ Increase max steering to 20°

    print(f"✅ CTE: {cte:.2f}, Heading Error: {np.degrees(heading_error):.2f}, PID Correction: {np.degrees(pid_correction):.2f}, Final Steering: {np.degrees(steering_angle):.2f}")

    return steering_angle

def process_img(image, display, vehicle, pid_controller, font):
    """
    Processes CARLA camera images, detects lanes, applies Stanley controller, and renders in Pygame.
    """
    if vehicle is None or vehicle.get_location() is None:
        print("⚠️ Vehicle not available. Skipping frame.")
        return

    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))[:, :, :3][:, :, ::-1]

    try:
        surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        lane_points = detect_lanes(array)

        if isinstance(lane_points, list) and len(lane_points) > 0:
            # ✅ Select the lane closest to the vehicle's **heading**
            closest_lane = min(lane_points, key=lambda lane: abs(lane[0][1] - vehicle.get_location().y))
            steering_angle = compute_steering_angle(vehicle, closest_lane, pid_controller)
            vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer=steering_angle))
        else:
            print("⚠️ No valid lanes detected. Reducing steering.")
            vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer=vehicle.get_control().steer * 0.9))

        display.blit(surface, (0, 0))

    except Exception as e:
        print(f"⚠️ Error in process_img: {e}")

def main():
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find('vehicle.tesla.model3')

    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    for sp in spawn_points:
        if sp.location is not None:
            vehicle = world.try_spawn_actor(vehicle_bp, sp)
            if vehicle is not None:
                print(f"✅ Vehicle spawned at: {sp.location}")
                break
    else:
        print("❌ Failed to find a suitable spawn point.")
        return

    pid_controller = PIDController(kp=0.1, ki=0.01, kd=0.02, integral_limit=3)  # ✅ Increase PID influence

    pygame.init()
    display = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("CARLA Vehicle Camera")
    font = pygame.font.Font(None, 36)

    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute("image_size_x", "800")
    camera_bp.set_attribute("image_size_y", "600")
    camera_bp.set_attribute("fov", "110")

    camera_transform = carla.Transform(carla.Location(x=0, y=0, z=5), carla.Rotation(pitch=-90))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    def camera_callback(image):
        try:
            process_img(image, display, vehicle, pid_controller, font)
        except Exception as e:
            print(f"⚠️ Error in camera callback: {e}")

    camera.listen(camera_callback)

    clock = pygame.time.Clock()

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt

            pygame.display.flip()
            clock.tick(30)

    except KeyboardInterrupt:
        print("⏹️ Simulation manually stopped.")

    finally:
        if camera is not None and camera.is_alive:
            camera.destroy()
        if vehicle is not None and vehicle.is_alive:
            vehicle.destroy()
        pygame.quit()

if __name__ == '__main__':
    main()


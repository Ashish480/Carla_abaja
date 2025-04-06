import carla
import numpy as np
import cv2
import time
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

    print(f"CTE: {cte:.2f}, Heading Error: {np.degrees(heading_error):.2f}, PID Correction: {np.degrees(pid_correction):.2f}, Final Steering: {np.degrees(steering_angle):.2f}")

    return steering_angle

# Initialize Video Writer for V4L2 loopback
width, height = 800, 600
out = cv2.VideoWriter('/dev/video10', cv2.VideoWriter_fourcc(*'MJPG'), 30, (800, 600))



def process_img(image, vehicle, steering_angle):
    """Converts CARLA image to numpy format and writes to V4L2 loopback for UFLD."""
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))  # Convert BGRA to BGRA
    array = array[:, :, :3]  # Drop alpha channel

    # Convert BGRA to BGR for OpenCV
    array = cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)

    # Flip the image to match OpenCV's display format
    array = np.rot90(array, k=1)  # Rotate 90° counterclockwise
    array = np.flip(array, axis=1)  # Flip horizontally

    # Write to virtual webcam (V4L2 loopback)
    out.write(array)

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
                print(f"Vehicle spawned at: {sp.location}")
                break
    else:
        print("Failed to find a suitable spawn point.")
        return

    waypoints = get_lane_waypoints(world, distance=2.0)
    pid_controller = PIDController(kp=0.1, ki=0.01, kd=0.05, integral_limit=5)

    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute("image_size_x", str(width))
    camera_bp.set_attribute("image_size_y", str(height))
    camera_bp.set_attribute("fov", "110")

    camera_transform = carla.Transform(carla.Location(x=0, y=0, z=5), carla.Rotation(pitch=-90))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    steering = 0.0
    camera.listen(lambda image: process_img(image, vehicle, steering))

    clock = time.time()
    start_time = time.time()

    time_list = []
    steering_angle_list = []

    try:
        while time.time() - start_time < 30:
            steering = compute_steering_angle(vehicle, waypoints, pid_controller)
            vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer=steering))

            time_list.append(time.time() - start_time)
            steering_angle_list.append(np.degrees(steering))

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("Simulation manually stopped.")

    finally:
        print("Destroying actors...")
        out.release()
        camera.destroy()
        vehicle.destroy()
        print("Simulation stopped.")

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


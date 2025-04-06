import carla
import cv2
import numpy as np
import random
import time

# Initialize Video Writer for V4L2 loopback
width, height = 800, 600
out = cv2.VideoWriter('/dev/video10', cv2.VideoWriter_fourcc(*'MJPG'), 30, (800, 600))

def process_img(image):
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

    # Spawn the vehicle
    for sp in spawn_points:
        if sp.location is not None:
            vehicle = world.try_spawn_actor(vehicle_bp, sp)
            if vehicle is not None:
                print(f"Vehicle spawned at: {sp.location}")
                break
    else:
        print("Failed to find a suitable spawn point.")
        return

    # Set up the camera
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute("image_size_x", str(width))
    camera_bp.set_attribute("image_size_y", str(height))
    camera_bp.set_attribute("fov", "110")

    camera_transform = carla.Transform(carla.Location(x=0, y=0, z=5), carla.Rotation(pitch=-90))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    # Listen to the camera feed and process it
    camera.listen(lambda image: process_img(image))

    # Simple simulation loop to move the vehicle
    try:
        while True:
            # Apply a simple forward control to move the vehicle
            vehicle.apply_control(carla.VehicleControl(throttle=0.3, steer=0.0))
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("Simulation manually stopped.")

    finally:
        print("Destroying actors...")
        out.release()
        camera.destroy()
        vehicle.destroy()
        print("Simulation stopped.")

if __name__ == '__main__':
    main()


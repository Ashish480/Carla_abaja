#!/usr/bin/env python

import glob
import os
import sys
import argparse
import pygame
import random
import numpy as np
import time
import carla
import matplotlib.pyplot as plt

def find_straight_road(world, minimum_length=46):
    map = world.get_map()
    spawn_points = map.get_spawn_points()
    suitable_points = []

    for spawn in spawn_points:
        waypoint = map.get_waypoint(spawn.location)
        length = 0
        initial_yaw = waypoint.transform.rotation.yaw
        while waypoint and length < minimum_length:
            next_waypoints = waypoint.next(2.0)
            if len(next_waypoints) == 1:
                next_waypoint = next_waypoints[0]
                if abs(next_waypoint.transform.rotation.yaw - initial_yaw) < 5:
                    length += 2.0
                    waypoint = next_waypoint
                else:
                    break
            else:
                break
        if length >= minimum_length:
            suitable_points.append(spawn)

    return suitable_points

def print_imu_data(imu):
    """Prints IMU sensor values in real-time"""
    accel = imu.accelerometer
    gyro = imu.gyroscope
    compass = imu.compass
    print(f"IMU Data - Accel: ({accel.x:.2f}, {accel.y:.2f}, {accel.z:.2f}) "
          f"Gyro: ({gyro.x:.2f}, {gyro.y:.2f}, {gyro.z:.2f}) "
          f"Compass: {compass:.2f}")

def print_wheel_speed_data(wheel_speed):
    """Prints wheel speed sensor values in real-time"""
    print(f"Wheel Speeds: FL: {wheel_speed.front_left:.2f} m/s, "
          f"FR: {wheel_speed.front_right:.2f} m/s, "
          f"RL: {wheel_speed.rear_left:.2f} m/s, "
          f"RR: {wheel_speed.rear_right:.2f} m/s")

def process_img(image, display, vehicle):
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

    velocity = vehicle.get_velocity()
    speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
    font = pygame.font.Font(None, 36)
    text_surface = font.render(f'Speed: {speed:.2f} km/h', True, (255, 255, 255))
    surface.blit(text_surface, (10, 20))
    display.blit(surface, (0, 0))

def main():
    argparser = argparse.ArgumentParser(description="CARLA Control Client")
    argparser.add_argument('--host', metavar='H', default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument('-p', '--port', metavar='P', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    args = argparser.parse_args()

    actor_list = []
    speed_data = []
    time_data = []

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(10.0)
        world = client.get_world()

        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        vehicle_bp.set_attribute('role_name', 'hero')

        spawn_points = find_straight_road(world)
        if not spawn_points:
            print("No suitable straight road found.")
            return

        spawn_point = random.choice(spawn_points)
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        actor_list.append(vehicle)
        print("Created %s at %s" % (vehicle.type_id, spawn_point.location))

        pygame.init()
        display = pygame.display.set_mode((800, 600), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption("CARLA Vehicle Camera")

        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute("image_size_x", "800")
        camera_bp.set_attribute("image_size_y", "600")
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(camera)

        imu_bp = blueprint_library.find('sensor.other.imu')
        imu_sensor = world.spawn_actor(imu_bp, carla.Transform(), attach_to=vehicle)
        actor_list.append(imu_sensor)
        imu_sensor.listen(lambda imu: print_imu_data(imu))

        def print_wheel_speed(vehicle):
            velocity = vehicle.get_velocity()
            speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            wheel_radius = 0.3  # Approximate value for a car wheel
            wheel_speed_rpm = (speed / (2 * np.pi * wheel_radius)) * 60
            print(f"Vehicle Speed: {speed:.2f} m/s, Approx. Wheel Speed: {wheel_speed_rpm:.2f} RPM")
        

        camera.listen(lambda image: process_img(image, display, vehicle))

        time.sleep(2)

        start_time = time.time()

        def drive_vehicle(vehicle, target_speed_kmph=30.0, simulation_time=12.0):
            start_sim_time = time.time()
            while True:
                current_time = time.time() - start_sim_time
                if current_time >= simulation_time:
                    vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                    break
                else:
                    vehicle.apply_control(carla.VehicleControl(throttle=0.5, brake=0.0))
                # Print wheel speed every loop iteration
                print_wheel_speed(vehicle)

                pygame.event.pump()
                pygame.display.flip()
                time.sleep(0.1)  # Adjust delay for real-time updates

        drive_vehicle(vehicle)

    finally:
        print('Destroying actors')
        for actor in actor_list:
            actor.destroy()
        pygame.quit()
        print('Done.')

if __name__ == '__main__':
    main()


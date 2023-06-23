# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles. The agent also responds to traffic lights,
traffic signs, and has different possible configurations. """

import random
import numpy as np
import carla
from basic_agent import BasicAgent
from local_planner import RoadOption
from behavior_types import Cautious, Aggressive, Normal
from srunner.scenariomanager.carla_data_provider import CarlaDataProvider

import copy
import math

from misc import get_speed, positive, is_within_distance, compute_distance, draw_waypoints, is_within_distance_test

import shapely

class RotatedRectangle(object):
    """
    This class contains method to draw rectangle and find intersection point.
    """

    def __init__(self, c_x, c_y, width, height, angle):
        self.c_x = c_x
        self.c_y = c_y
        self.w = width  # pylint: disable=invalid-name
        self.h = height  # pylint: disable=invalid-name
        self.angle = angle

    def get_contour(self):
        """
        create contour
        """
        w = self.w
        h = self.h
        c = shapely.geometry.box(-w / 2.0, -h / 2.0, w / 2.0, h / 2.0)
        rc = shapely.affinity.rotate(c, self.angle)
        return shapely.affinity.translate(rc, self.c_x, self.c_y)

    def intersection(self, other):
        """
        Obtain a intersection point between two contour.
        """
        return self.get_contour().intersection(other.get_contour())

class BehaviorAgent(BasicAgent):
    """
    BehaviorAgent implements an agent that navigates scenes to reach a given
    target destination, by computing the shortest possible path to it.
    This agent can correctly follow traffic signs, speed limitations,
    traffic lights, while also taking into account nearby vehicles. Lane changing
    decisions can be taken by analyzing the surrounding environment such as tailgating avoidance.
    Adding to these are possible behaviors, the agent can also keep safety distance
    from a car in front of it by tracking the instantaneous time to collision
    and keeping it in a certain range. Finally, different sets of behaviors
    are encoded in the agent, from cautious to a more aggressive ones.
    """

    def __init__(self, vehicle, behavior='normal', opt_dict={}, map_inst=None, grp_inst=None):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param behavior: type of agent to apply
        """

        super().__init__(vehicle, opt_dict=opt_dict, map_inst=map_inst, grp_inst=grp_inst)
        self._look_ahead_steps = 0

        # Vehicle information
        self._speed = 0
        self._speed_limit = 0
        self._direction = None
        self._incoming_direction = None
        self._incoming_waypoint = None
        self._min_speed = 5
        self._behavior = None
        self._sampling_resolution = 4.5
        self._vehicle_heading = None
        
        
        # Parameters for agent behavior
        self.try_overtake = False
        self._obstacle_to_overtake = None
        self.overtaking = False
        self.overtaking_bicycle = False
        self._ending_overtake = False
        self.overtake_list = []
        self.old_queue = None
        self._prev_dist_obstacle = None
        self._count_dist_obstacle = 0
        self.original_lane = None
        self._narrowing = False
        self._stop_counter = 0
        self.wait_at_stop = False
        self._stop_managed = None
        self._bicycles = ['vehicle.bh.crossbike', 'vehicle.diamondback.century', 'vehicle.gazelle.omafiets']

        # Parameters for agent behavior
        if behavior == 'cautious':
            self._behavior = Cautious()

        elif behavior == 'normal':
            self._behavior = Normal()

        elif behavior == 'aggressive':
            self._behavior = Aggressive()
            
    def dist(self, v, w): 
        return v.get_location().distance(w.get_location()) - v.bounding_box.extent.x - w.bounding_box.extent.x
        # return v.get_location().distance(w.get_location())
       

    def _update_information(self):
        """
        This method updates the information regarding the ego
        vehicle based on the surrounding world.
        """
        self._speed = get_speed(self._vehicle)
        print("Speed: ", self._speed)
        self._speed_limit = self._vehicle.get_speed_limit()
        self._local_planner.set_speed(self._speed_limit)
        self._direction = self._local_planner.target_road_option
        self._vehicle_heading = self._vehicle.get_transform().rotation.yaw
        if self._direction is None:
            self._direction = RoadOption.LANEFOLLOW

        self._look_ahead_steps = int((self._speed) / 10)

        self._incoming_waypoint, self._incoming_direction = self._local_planner.get_incoming_waypoint_and_direction(
            steps=self._look_ahead_steps)
        if self._incoming_direction is None:
            self._incoming_direction = RoadOption.LANEFOLLOW
        
    def traffic_light_manager(self):
        """
        This method is in charge of behaviors for red lights.
        """
        actor_list = self._world.get_actors()
        lights_list = actor_list.filter("*traffic_light*")
        affected, _ = self._affected_by_traffic_light(lights_list)

        return affected

    def _tailgating(self, waypoint, vehicle_list):
        """
        This method is in charge of tailgating behaviors.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :param vehicle_list: list of all the nearby vehicles
        """

        left_turn = waypoint.left_lane_marking.lane_change
        right_turn = waypoint.right_lane_marking.lane_change

        left_wpt = waypoint.get_left_lane()
        right_wpt = waypoint.get_right_lane()

        behind_vehicle_state, behind_vehicle, _ = self._vehicle_obstacle_detected(vehicle_list, max(
            self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, low_angle_th=160)
        if behind_vehicle_state and self._speed < get_speed(behind_vehicle):
            if (right_turn == carla.LaneChange.Right or right_turn ==
                    carla.LaneChange.Both) and waypoint.lane_id * right_wpt.lane_id > 0 and right_wpt.lane_type == carla.LaneType.Driving:
                new_vehicle_state, _, _ = self._vehicle_obstacle_detected(vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=1)
                if not new_vehicle_state:
                    print("Tailgating, moving to the right!")
                    end_waypoint = self._local_planner.target_waypoint
                    self._behavior.tailgate_counter = 200
                    self.set_destination(end_waypoint.transform.location,
                                         right_wpt.transform.location)
            elif left_turn == carla.LaneChange.Left and waypoint.lane_id * left_wpt.lane_id > 0 and left_wpt.lane_type == carla.LaneType.Driving:
                new_vehicle_state, _, _ = self._vehicle_obstacle_detected(vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=-1)
                if not new_vehicle_state:
                    print("Tailgating, moving to the left!")
                    end_waypoint = self._local_planner.target_waypoint
                    self._behavior.tailgate_counter = 200
                    self.set_destination(end_waypoint.transform.location,
                                         left_wpt.transform.location)

    def collision_and_car_avoid_manager(self, waypoint):
        """
        This module is in charge of warning in case of a collision
        and managing possible tailgating chances.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a vehicle nearby, False if not
            :return vehicle: nearby vehicle
            :return distance: distance to nearby vehicle
        """
        special_vehicle_list = ["vehicle.dodge.charger_police_2020", "vehicle.diamondback.century", "vehicle.ford.crown", "vehicle.mercedes.coupe_2020","vehicle.gazelle.omafiets"]
        vehicle_list = list(self._world.get_actors().filter("*vehicle*"))
        object_list = list(self._world.get_actors().filter("*static*"))
        vehicle_list.extend(object_list)
        def dist(v): return v.get_location().distance(waypoint.transform.location)
        vehicle_list = [v for v in vehicle_list if dist(v) < 45 and v.id != self._vehicle.id]
        objects_to_ignore = ["static.prop.dirtdebris01"]
        vehicle_list = [v for v in vehicle_list if v.type_id not in objects_to_ignore]
        vehicle_list = sorted(vehicle_list, key=dist)
        if len(vehicle_list) == 0:
            return False, None, None

        if self._direction == RoadOption.CHANGELANELEFT:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=1)
        elif self._direction == RoadOption.LEFT or self._direction == RoadOption.RIGHT:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), low_angle_th=45, up_angle_th=315, lane_offset=-1)
        else:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                    vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=30, lane_offset=0)
            # for parked vehicles lane_offset=1 is needed
            vehicle_state_special, vehicle_special, distance_special = self._vehicle_obstacle_detected(
                    vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=30, lane_offset=1)
            if vehicle_state_special and vehicle_special.type_id in special_vehicle_list:
                vehicle_state = vehicle_state_special
                vehicle = vehicle_special
                distance = distance_special

            # Check for tailgating
            if not vehicle_state and self._direction == RoadOption.LANEFOLLOW \
                    and not waypoint.is_junction and self._speed > 10 \
                    and self._behavior.tailgate_counter == 0:
                self._tailgating(waypoint, vehicle_list)

        return vehicle_state, vehicle, distance

    def check_obstacles_to_overtake(self, waypoint):
        self.overtake_list = []
        def dist(v): return v.get_location().distance(waypoint.transform.location)
        vehicle_list = list(self._world.get_actors().filter("*vehicle*"))
        object_list = list(self._world.get_actors().filter("*static*"))
        vehicle_list.extend(object_list)
        
        vehicle_list = [v for v in vehicle_list if is_within_distance(v.get_transform(), self._vehicle.get_transform(), 45, [0, 30]) and v.id != self._vehicle.id]
        #vehicle_list = [v for v in vehicle_list if (self._map.get_waypoint(v.get_location()).lane_id == waypoint.lane_id or self._map.get_waypoint(v.get_location()).lane_id == 2)]
        vehicle_list = [v for v in vehicle_list if self._map.get_waypoint(v.get_location()).lane_id == waypoint.lane_id] 

        # print("BEFORE --- VEHICLE LIST for check overtake: ", end="\n")
        # for v in vehicle_list:
        #     print(v.type_id, end="- ")
        #     print(self._map.get_waypoint(v.get_location()).lane_id, end="- ")
        #     print(v.get_location().x - self._vehicle.get_location().x, end="- ")
        #     print(dist(v), end=", ")
        # print()
        
        vehicle_list = sorted(vehicle_list, key=dist)
        prec_location = None
        for v in vehicle_list:
            if prec_location is None:
                prec_location= v.get_location()
                self.overtake_list.append(v)
            else:
                distance = v.get_location().distance(prec_location)
                prec_location = v.get_location()
                #print("DISTANCE between {} and {}: {}".format(self.overtake_list[-1].type_id, v.type_id, str(distance)))
                if distance < 20:
                    self.overtake_list.append(v)
                else:
                    break
        if len(self.overtake_list) > 0:
            return dist(self.overtake_list[-1])
        return None
        # print("OVERTAKE LIST: ", end="\n")
        # for v in self.overtake_list:
        #     print(v.type_id, end=", ")
        
        # print("DISTANCE by last object: " + str(dist(self.overtake_list[-1])))

    def _check_vehicles_objects_in_direction(self, waypoint, distance=45, offset=None, angles=(0, 180), check_other_lane=False, objects=True, return_list=False):
        vehicle_list = list(self._world.get_actors().filter("*vehicle*"))
        if objects:
            object_list = list(self._world.get_actors().filter("*static*"))
            vehicle_list.extend(object_list)
       
        vehicle_list = [v for v in vehicle_list if self.dist(v, self._vehicle) < distance and v.id != self._vehicle.id]
        #print("VEHICLE LIST: ", end="\n")

        ego_wpt = self._map.get_waypoint(self._vehicle.get_location())
        
        if (ego_wpt.is_junction or self._incoming_waypoint.is_junction) and (self._incoming_direction in [RoadOption.LEFT, RoadOption.RIGHT]):
            for v in vehicle_list:
                print(v.type_id, end="- ")
                print(self._map.get_waypoint(v.get_location()).lane_id, end="- ")
                print(self.dist(v, self._vehicle), end=", ")
                print(is_within_distance_test(v.get_transform(), self._vehicle.get_transform(), distance, (0, 180)), end=", ")
            # print()


        vehicle_list = [v for v in vehicle_list if is_within_distance(v.get_transform(), ego_wpt.transform, distance, angles) and v.id != self._vehicle.id]
        
        for v in vehicle_list:
            wpt = self._map.get_waypoint(v.get_location())
            draw_waypoints(self._world, [wpt], color=carla.Color(0, 0, 255))
        """print("VEHICLE LIST: ", end="\n")
        for v in vehicle_list:
            print(v.type_id, end="- ")
            print(self._map.get_waypoint(v.get_location()).lane_id, end="- ")
            print(self.dist(v, self._vehicle), end=", ")
            print(is_within_distance_test(v.get_transform(), self._vehicle.get_transform(), distance, (0, 180)), end=", ")
        print()"""

        if offset is None:
            if waypoint.lane_id < 0:
                lane_offset = -(waypoint.lane_id - 1)
            else:
                lane_offset = -(waypoint.lane_id + 1)
        else:
            lane_offset = 0

        if check_other_lane:
            vehicle_list = [v for v in vehicle_list if self._map.get_waypoint(v.get_location()).lane_id == waypoint.lane_id + lane_offset]

        vehicle_list = sorted(vehicle_list, key=lambda v: self.dist(v, self._vehicle))
        # for v in vehicle_list:
        #     print(v, end=" ")
        #     print(self._map.get_waypoint(v.get_location()).lane_id)

        if return_list:
            return vehicle_list
        if len(vehicle_list) == 0:
            return False, None, None
        else:
            return True, vehicle_list[0], self.dist(vehicle_list[0], self._vehicle)
    
    def overtake_manager_old(self, waypoint, distance=45, offset=None):
        """
        This module is in charge of warning in case of a collision
        and managing possible tailgating chances.

            :param waypoint: current waypoint of the agent
            :param distance: distance to check for vehicles
            :param offset: offset to check for vehicles
            :return vehicle_state: True if there is a vehicle nearby, False if not
            :return vehicle: nearby vehicle
            :return distance: distance to nearby vehicle
        """
        return self._check_vehicles_objects_in_direction(waypoint, distance, offset, (0, 90), check_other_lane=True)
    
    def check_beside(self, waypoint, distance=15):
        """
        This module is in charge of warning of obstacles beside the vehicle.
        
            :param waypoint: current waypoint of the agent
            :param distance: distance to check for vehicles
            :return vehicle_state: True if there is a vehicle nearby, False if not
            :return vehicle: nearby vehicle
            :return distance: distance to nearby vehicle
            """
        return self._check_vehicles_objects_in_direction(waypoint, distance, None, (65, 115), check_other_lane=False)

    def check_front_overtaking(self, waypoint, distance=20):
        """
        This module is in charge of warning of obstacles in front of the vehicle.
        
            :param waypoint: current waypoint of the agent
            :param distance: distance to check for vehicles
            :return vehicle_state: True if there is a vehicle nearby, False if not
            :return vehicle: nearby vehicle
            :return distance: distance to nearby vehicle
            """
        return self._check_vehicles_objects_in_direction(waypoint, distance, 0, (170, 180), check_other_lane=True, objects=False)
    
    def check_vehicles_intersection(self, waypoint, distance=65):
        """
        This module is in charge of warning of obstacles in front of the vehicle.
        
            :param waypoint: current waypoint of the agent
            :param distance: distance to check for vehicles
            :return vehicle_state: True if there is a vehicle nearby, False if not
            :return vehicle: nearby vehicle
            :return distance: distance to nearby vehicle
            """
        
        vehicle_list = self._check_vehicles_objects_in_direction(waypoint, distance, 0, (0, 90), check_other_lane=False, objects=False, return_list=True)
        #state_left, actor_left, dist_left = self._check_vehicles_objects_in_direction(waypoint, distance, 0, (270, 360), check_other_lane=False, objects=False)
        #print("STATE LEFT: " + str(state_left))
        # print("STATE RIGHT: " + str(state_right))
        #input()

        #return state_left, state_right, actor_left, actor_right, dist_left, dist_right
        # return state_right, actor_right, dist_right
        return vehicle_list

    def detect_lane_obstacle(self, actor, extension_factor=3.2, extension_factor_adv=2.2, margin=1.02):
        """
        This function identifies if an obstacle is present in front of the reference actor
        """
        world = CarlaDataProvider.get_world()
        world_actors = world.get_actors().filter('vehicle.*')
        world_actors = [world_actor for world_actor in world_actors if get_speed(world_actor) > 0.1]
        actor_bbox = actor.bounding_box
        actor_transform = actor.get_transform()
        actor_location = actor_transform.location
        actor_vector = actor_transform.rotation.get_forward_vector()
        actor_vector = np.array([actor_vector.x, actor_vector.y])
        actor_vector = actor_vector / np.linalg.norm(actor_vector)
        actor_vector = actor_vector * (extension_factor - 1) * actor_bbox.extent.x
        actor_location = actor_location + carla.Location(actor_vector[0], actor_vector[1])
        actor_yaw = actor_transform.rotation.yaw

        is_hazard = False
        for adversary in world_actors:
            print("velocità actor: ", get_speed(adversary))
            if adversary.id != actor.id and \
                    actor_transform.location.distance(adversary.get_location()) < 50:
                adversary_bbox = adversary.bounding_box
                adversary_transform = adversary.get_transform()
                adversary_loc = adversary_transform.location
                adversary_yaw = adversary_transform.rotation.yaw
                """overlap_adversary = RotatedRectangle(
                    adversary_loc.x, adversary_loc.y, 2.4 * margin * adversary_bbox.extent.x * extension_factor_adv,
                    1.7 * margin * adversary_bbox.extent.y, adversary_yaw
                )
                overlap_actor = RotatedRectangle(
                    actor_location.x, actor_location.y, 2.2 * margin * actor_bbox.extent.x,
                    2.3 * margin * actor_bbox.extent.y * extension_factor, actor_yaw
                )"""
                overlap_adversary = RotatedRectangle(
                    adversary_loc.x, adversary_loc.y, 2 * margin * adversary_bbox.extent.x * extension_factor_adv,
                    2 * margin * adversary_bbox.extent.y, adversary_yaw
                )
                overlap_actor = RotatedRectangle(
                    actor_location.x, actor_location.y, 2 * margin * actor_bbox.extent.x * extension_factor,
                    2 * margin * actor_bbox.extent.y , actor_yaw
                )
                overlap_area = overlap_adversary.intersection(overlap_actor).area
                if overlap_area > 0:
                    is_hazard = True
                    break

        return is_hazard

            
    def pedestrian_avoid_manager(self, waypoint):
        """
        This module is in charge of warning in case of a collision
        with any pedestrian.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a walker nearby, False if not
            :return vehicle: nearby walker
            :return distance: distance to nearby walker
        """
        walker_list = self._world.get_actors().filter("*walker.pedestrian*")
        def dist(w): return w.get_location().distance(waypoint.transform.location)
        walker_list = [w for w in walker_list if dist(w) < 20]
        walker_list = sorted(walker_list, key=dist)

        # if self._direction == RoadOption.CHANGELANELEFT:
        #     walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
        #         self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=-1)
        # elif self._direction == RoadOption.CHANGELANERIGHT:
        #     walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
        #         self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=1)
        # else:
        #     walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
        #         self._behavior.min_proximity_threshold, self._speed_limit), up_angle_th=60)
        
        walker_list = [v for v in walker_list if is_within_distance(v.get_transform(), waypoint.transform, 20, [0, 60])]
        if walker_list == []:
            return False, None, None
        return True, walker_list[0], self.dist(walker_list[0], self._vehicle)

    def car_following_manager(self, vehicle, distance, debug=False):
        """
        Module in charge of car-following behaviors when there's
        someone in front of us.

            :param vehicle: car to follow
            :param distance: distance from vehicle
            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """

        vehicle_speed = get_speed(vehicle)
        delta_v = max(1, (self._speed - vehicle_speed) / 3.6)
        ttc = distance / delta_v if delta_v != 0 else distance / np.nextafter(0., 1.)

        # Under safety time distance, slow down.
        if self._behavior.safety_time > ttc > 0.0:
            target_speed = min([
                positive(vehicle_speed - self._behavior.speed_decrease),
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # Actual safety distance area, try to follow the speed of the vehicle in front.
        elif self._behavior.safety_time * 2 > ttc >= self._behavior.safety_time:
            target_speed = min([
                max(self._min_speed, vehicle_speed),
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        # Normal behavior.
        else:
            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist])
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)

        return control

    def change_path(self, start_waypoint, total_distance, follow_direction=True, save_and_pop_queue=False):
        if save_and_pop_queue:
            self.old_queue = self._local_planner._waypoints_queue
        distance = 0
        plan = [(start_waypoint.get_left_lane(), RoadOption.LANEFOLLOW)]
        while distance < total_distance:
            if follow_direction:
                next_wps = plan[-1][0].next(1)
            else:
                next_wps = plan[-1][0].previous(1)
            if not next_wps:
                print("Waypoint finished")
                break
            if save_and_pop_queue:
                self.old_queue.popleft()
            next_wp = next_wps[0]
            distance += next_wp.transform.location.distance(plan[-1][0].transform.location)
            plan.append((next_wp, RoadOption.LANEFOLLOW))
        
        return plan


    def check_for_lane_narrowing(self, waypoint):
        """
        This module is in charge of checking if the lane is narrowing.

            :param waypoint: current waypoint of the agent
            :return state: True if the lane is narrowing, False if not
        """
        object_list = list(self._world.get_actors().filter("static.prop.constructioncone"))
        warning_list = list(self._world.get_actors().filter("static.prop.trafficwarning"))    
        def dist(v): return v.get_location().distance(waypoint.transform.location)
        object_list = [v for v in object_list if dist(v) < 30 and is_within_distance(v.get_transform(), waypoint.transform, 30, [0, 90])]
        warning_list = [v for v in warning_list if dist(v) < 30 and is_within_distance(v.get_transform(), waypoint.transform, 30, [0, 90])]
        if len(warning_list) > 0 or self.overtaking:
            object_list = []
        return len(object_list) > 0

    def check_for_stop_sign(self, waypoint):
        """
        This module is in charge of checking if there's a stop sign.
        
            :param waypoint: current waypoint of the agent
            :return state: True if there's a stop sign, False if not
        """
        object_list = list(self._world.get_actors().filter("*stop"))
        def dist(v): return v.get_location().distance(waypoint.transform.location)
        object_list = sorted(object_list, key=dist)
        for v in object_list:
            wpt = self._map.get_waypoint(v.get_location())
            draw_waypoints(self._world, [wpt], 1)
        #print("STOP IS WITHIN DISTANCE test: ", [is_within_distance_test(v.get_transform(), waypoint.transform, 70, [0, 180]) for v in object_list])

        object_list = [v for v in object_list if is_within_distance(v.get_transform(), waypoint.transform, 20, [0, 25])]
        
        ### TODO check if the stop sign is in the same road as the ego vehicle
        # object_list = [v for v in object_list if self._map.get_waypoint(v.get_location()).road_id == waypoint.road_id]
        # if len(object_list) > 0:
        #     input()
        return (len(object_list) > 0, object_list[0] if len(object_list) > 0 else None)

    def closest_intersection(self, waypoint):
        def dist(v): return v.distance(waypoint.transform.location)
        closest_distance = float('inf')

        intersections = [wpt[0] for wpt in self._local_planner._waypoints_queue if wpt[0].is_junction]
        
        for intersection in intersections:
            intersection_location = intersection.transform.location
            intersection_distance = dist(intersection_location)

            if intersection_distance < closest_distance: 
                closest_distance = intersection_distance 

        return closest_distance


    def run_step(self, debug=True):
        """
        Execute one step of navigation.

            :param debug: boolean for debugging
            :return control: carla.VehicleControl
        """
        self._update_information()

        control = None
        if self._behavior.tailgate_counter > 0:
            self._behavior.tailgate_counter -= 1

        ego_vehicle_loc = self._vehicle.get_location()
        ego_vehicle_wp = self._map.get_waypoint(ego_vehicle_loc)

        #print("INCOMING IS IN JUNCTION: ", self._incoming_waypoint.is_junction)
        #print("EGO IS IN JUNCTION: ", ego_vehicle_wp.is_junction)

        stop_state, stop_sign = self.check_for_stop_sign(ego_vehicle_wp)

        #if stop_state and not self._incoming_waypoint.is_junction:
        # if stop_state:  
        #     print("ho visto lo stop") 
        #     print("DISTANCE TO STOP: ", self.dist(self._vehicle, stop_sign))
        #     # if self._stop_managed is None:
        #     #     print("controllo: ", stop_sign.id != self._stop_managed)
        #     print("Is within distance test: ", is_within_distance_test(stop_sign.get_transform(), self._vehicle.get_transform(), 15, [0, 180]))
        #     #input()
               
        if stop_state and not self.wait_at_stop and stop_sign.id != self._stop_managed and not ego_vehicle_wp.is_junction:
        # if stop_state and not self.wait_at_stop and stop_sign.id != self._stop_managed:
            #print("faccio soft stop")
            #self.soft_stop()
            if self._incoming_waypoint.is_junction:
                # print("il prossimo wpt è una junction")
                #input()
                # and self.dist(self._vehicle, stop_sign) < 9:
                # input()
                #if 9 < self.dist(self._vehicle, stop_sign) < 14:
                self._stop_counter = 100
                self.wait_at_stop = True
                self._stop_managed = stop_sign.id
                return self.emergency_stop()
        
        
        #if stop_state and self._vehicle.get_velocity().length() < 1.0:
        #if self.dist(self._vehicle, stop_sign) < 9:
        elif self.wait_at_stop and self._stop_counter > 0:
            # print("sono entrato nel secondo if")
            # print("stop counter: ", self._stop_counter)
            self._stop_counter -= 1
            return self.emergency_stop()
        else:
            self.wait_at_stop = False
            

        # 1: Red lights and stops behavior
        if self.traffic_light_manager():
            return self.emergency_stop()

        # 1.1: Pedestrian avoidance behaviors
        walker_state, walker, w_distance = self.pedestrian_avoid_manager(ego_vehicle_wp)

        if walker_state:
            # print("ho trovato un pedone")
            # Distance is computed from the center of the two cars,
            # we use bounding boxes to calculate the actual distance
            distance = w_distance - max(
                walker.bounding_box.extent.y, walker.bounding_box.extent.x) - max(
                    self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)
            #print("DISTANCE TO PEDESTRIAN: ", distance)
            #print("SELF SPEED: ", self._speed)
            #input()
            # Emergency brake if the car is very close.
            if distance - 1 < self._behavior.braking_distance:
                return self.emergency_stop()
            else:
                return self.decelerate(speed=20)
        
        #print("EGO WP IS JUNCTION: ", ego_vehicle_wp.is_junction)
        #print("INCOMING WP IS JUNCTION: ", self._incoming_waypoint.is_junction)
        # if ego_vehicle_wp.is_junction or self._incoming_waypoint.is_junction:
        #     input()
        
        # 2: Normal behavior 
        if not self._incoming_waypoint.is_junction and not ego_vehicle_wp.is_junction:
            # 2.1: Lane narrowing
            narrowing_state = self.check_for_lane_narrowing(ego_vehicle_wp)

            if narrowing_state:
                self._local_planner.set_lateral_offset(-1.25)
                self._narrowing = True
                #to prevent outlane penalty
                return self.decelerate(speed=30)
            else:
                self._local_planner.set_lateral_offset(0)
                self._narrowing = False
                target_speed = min([
                    self._behavior.max_speed,
                    self._speed_limit - self._behavior.speed_lim_dist])
                self._local_planner.set_speed(target_speed)
                control = self._local_planner.run_step(debug=debug)
                
            # 2.2: Ending overtake behavior
            if self._ending_overtake:
                # print("sto terminando sorpasso")
                if not len(self._local_planner._waypoints_queue) > 1:
                    self._ending_overtake = False
                    self.overtaking = False
                    route_trace_p = list(map(lambda x: x[0], self._waypoints_queue_copy))
                    route_trace = []
                    for i in range(self._global_planner._find_closest_in_list(ego_vehicle_wp, route_trace_p), len(self._waypoints_queue_copy)):
                        route_trace.append(self._waypoints_queue_copy[i])
                    self._local_planner.set_global_plan(route_trace, True)
                if self.overtaking_bicycle:
                    # input("sto terminando sorpasso bicicletta")
                    self._local_planner.set_lateral_offset(0)
                    self._ending_overtake = False
                    self.overtaking = False
                    self.overtaking_bicycle = False
                target_speed = min([self._behavior.max_speed, self._speed_limit])
                self._local_planner.set_speed(target_speed)
                control = self._local_planner.run_step(debug=debug)
                return control
            # 2.3: Overtaking behavior
            elif self.overtaking:
                #print("Overtaking...", [x.type_id for x in self.overtake_list])
                if not len(self._local_planner._waypoints_queue) > 1: # or self.overtaking_bicycle:
                    # state, actor, distance = self.overtake_manager_old(ego_vehicle_wp, distance=75, check_lane=True)
                    state, actor, distance = self.check_beside(ego_vehicle_wp)
                    if ego_vehicle_wp.lane_id != self.original_lane: # and not self.overtaking_bicycle:
                        state_front, actor_front, distance_front = self.check_front_overtaking(ego_vehicle_wp)
                    else:
                        state_front = None
                    # print("overtake manager inside overtaking: ", state, actor, distance)
                    # print("dimensione ego vehicle: ", self._vehicle.bounding_box.extent.x*2)
                    #if not state and not actor in self.overtake_list:
                    if (actor is not None and actor.id == self.overtake_list[-1].id) or (state_front is not None and state_front):
                        # input("overtake finished")
                        # self._local_planner.set_lateral_offset(0)
                        # print("overtake finished")
                        if self.lane_change("left", self._vehicle_heading, 0, 1.50, 0.3):
                            self._ending_overtake = True
                    else:
                        self.lane_change("left", self._vehicle_heading, 0.85, 0, 0)
                if self.overtaking_bicycle:   
                    self._local_planner.set_lateral_offset(1.4)
                    state, actor, distance = self.check_beside(ego_vehicle_wp)
                    if actor is not None and actor.id == self.overtake_list[-1].id :
                        self._ending_overtake = True
                target_speed = max([self._behavior.max_speed, self._speed_limit]) + 10
                self._local_planner.set_speed(target_speed)
                control = self._local_planner.run_step(debug=debug)
                return control
            
            # 2.4: Car following behavior
            actor_state, actor, distance = self.collision_and_car_avoid_manager(ego_vehicle_wp)
            # self.collision_and_car_avoid_manager_bb(ego_vehicle_wp)
            #print("IF")
            
            
            if actor_state:
                print("Collision and car avoid manager: ", actor_state, actor, distance)
                # Distance is computed from the center of the two cars,
                # we use bounding boxes to calculate the actual distance
                # print("Actor: ", actor)
                # print("Distance to car: ", distance, "m")
                distance = distance - max(
                    actor.bounding_box.extent.y, actor.bounding_box.extent.x) - max(
                        self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)
            
                #if not self._narrowing and not ego_vehicle_wp.is_junction:
                if not self._narrowing:
                    # ego_forward_vector = ego_vehicle_wp.transform.get_forward_vector()
                    # actor_forward_vector = self._map.get_waypoint(actor.get_location()).transform.get_forward_vector()
                    ego_forward_vector = self.get_forward_vector(self._vehicle)
                    actor_forward_vector = self.get_forward_vector(actor)
                    #print("ANGOLO: ", self.compute_angle(ego_forward_vector, actor_forward_vector))
                    print("Actor speed: ", get_speed(actor))
                    if 80 < self.compute_angle(ego_forward_vector, actor_forward_vector) < 100 and get_speed(actor) > 0:
                        return self.emergency_stop()
                    elif distance - 1 < self._behavior.braking_distance and not self.overtaking and get_speed(actor) < 5:
                        if ego_vehicle_wp.left_lane_marking.type == carla.LaneMarkingType.Broken or ego_vehicle_wp.left_lane_marking.type == carla.LaneMarkingType.SolidBroken:
                            # print("velocità, actor: ", actor.get_velocity().length(), actor.type_id)
                            print("Trying to overtake...")
                            #input()
                            distance_to_last_obj = self.check_obstacles_to_overtake(ego_vehicle_wp)
                            #print("distanza all'ultimo oggetto: ", distance_to_last_obj)
                            if len(self.overtake_list) <= 2:
                                state, actor, actor_dist = self.overtake_manager_old(ego_vehicle_wp, distance=65)
                                print("distanza dall'attore: ", actor_dist)
                                if actor is not None:
                                   print("actor id: ", actor.id)
                                #input()
                            else:
                                state, actor, _ = self.overtake_manager_old(ego_vehicle_wp, distance=max(80, distance_to_last_obj*3))
                            #print("Intersection closer:", self.closest_intersection(ego_vehicle_wp) > distance_to_last_obj*3)
                            if not state and self.closest_intersection(ego_vehicle_wp) > distance_to_last_obj*3:
                                if self.overtake_list[0].type_id in self._bicycles:
                                    # input("Overtaking bicycle")
                                    self._local_planner.set_lateral_offset(1.4)
                                    self.overtaking = True
                                    self.overtaking_bicycle = True
                                    self.original_lane = ego_vehicle_wp.lane_id
                                    target_speed = max([self._behavior.max_speed, self._speed_limit]) + 10
                                    #print("speed: ", target_speed)
                                    self._local_planner.set_speed(target_speed)
                                    control = self._local_planner.run_step(debug=debug)
                                    return control
                                else:
                                    self._waypoints_queue_copy = self._local_planner._waypoints_queue.copy()
                                    if self.lane_change("left", self._vehicle_heading, 0, 2, 1.7):
                                        self.overtaking = True
                                        self.original_lane = ego_vehicle_wp.lane_id
                                        target_speed = max([self._behavior.max_speed, self._speed_limit]) + 10
                                        #print("speed: ", target_speed)
                                        self._local_planner.set_speed(target_speed)
                                        control = self._local_planner.run_step(debug=debug)
                                        return control
                        else:
                            pass
                            #print("Cannot change lane")
                        return self.emergency_stop()
                    else:
                        #print("Car following", actor)
                        control = self.car_following_manager(actor, distance)
            else:
                #print("sono entrato nel nuovo else")
                target_speed = min([
                    self._behavior.max_speed,
                    self._speed_limit - self._behavior.speed_lim_dist])
                self._local_planner.set_speed(target_speed)
                control = self._local_planner.run_step(debug=debug)
        
        # 3: Intersection behavior
        elif (ego_vehicle_wp.is_junction or self._incoming_waypoint.is_junction) and (self._incoming_direction in [RoadOption.LEFT, RoadOption.RIGHT]):
            
            ### NEW CODE ###
            state = self.detect_lane_obstacle(self._vehicle)
            if state:
                print("Collision detected")
                return self.emergency_stop()
            else:
                print("No collision detected")
                return self.decelerate(speed=50)
            ### END NEW CODE ###

            
            # vehicles_at_intersection = self.check_vehicles_intersection(ego_vehicle_wp)
            
            # intersection_clear = True
            # for vehicle in vehicles_at_intersection:
            #     # check if a detected vehicle is at least 3 meters before a junction
            #     if self._map.get_waypoint(vehicle.get_location()).next(15)[-1].is_junction:
            #         input("is junction")
            #         dist = self._vehicle.get_location().distance(vehicle.get_location())
            #         vehicle_speed = get_speed(vehicle)
            #         delta_v = max(1, (self._speed - vehicle_speed) / 3.6)
            #         ttc = dist / delta_v if delta_v != 0 else dist / np.nextafter(0., 1.)
            #         print("ttc: ", ttc)
            #         if ttc < self._behavior.safety_time and vehicle_speed > 0:
            #         #if actor_state and get_speed(actor) > 1.0:
            #         #if dist_right < self._behavior.braking_distance:
            #             print("ho trovato un ostacolo nell'incrocio")
            #             intersection_clear = False
            #             # return self.emergency_stop()
                    
            # if intersection_clear:
            #     print("sono nell'incrocio e non ho trovato niente")
            #     #return self.decelerate(speed=20)
            #     target_speed = min([
            #         self._behavior.max_speed,
            #         self._speed_limit - self._behavior.speed_lim_dist])
            #     """self._local_planner.set_speed(target_speed)
            #     control = self._local_planner.run_step(debug=debug)"""
            #     #target_speed = max([self._behavior.max_speed, self._speed_limit]) + 10
            #     self._local_planner.set_speed(target_speed)
            #     control = self._local_planner.run_step(debug=debug)
            # else:
            #     return self.emergency_stop()

            ### OLD CODE ###
            # if actor_state_right and actor_right is not None and self._map.get_waypoint(actor_right.get_location()).is_junction:
            #     actor_right_speed = get_speed(actor_right)
            #     delta_v = max(1, (self._speed - actor_right_speed) / 3.6)
            #     ttc = actor_right_dist / delta_v if delta_v != 0 else actor_right_dist / np.nextafter(0., 1.)
            #     print("ttc: ", ttc)
            #     if ttc < self._behavior.safety_time and actor_right_speed > 0:
            #     #if actor_state and get_speed(actor) > 1.0:
            #     #if dist_right < self._behavior.braking_distance:
            #     #print("ho trovato un ostacolo nell'incrocio")
            #         return self.emergency_stop()
            #     else:
            #         target_speed = max([self._behavior.max_speed, self._speed_limit]) + 10
            #         self._local_planner.set_speed(target_speed)
            #         control = self._local_planner.run_step(debug=debug)
            #     # else:
            #     #     control = self.car_following_manager(actor_right, dist_right)
            #     """elif actor_state_left and actor_left is not None and self._map.get_waypoint(actor_left.get_location()).is_junction:
            #     actor_left_speed = get_speed(actor_left)
            #     delta_v = max(1, (self._speed - actor_left_speed) / 3.6)
            #     ttc = dist_left / delta_v if delta_v != 0 else dist_left / np.nextafter(0., 1.)
            #     print("ttc: ", ttc)
            #     if ttc < self._behavior.safety_time and actor_left_speed > 0:
            #     #if actor_state and get_speed(actor) > 1.0:
            #     #if dist_right < self._behavior.braking_distance:
            #     #print("ho trovato un ostacolo nell'incrocio")
            #         return self.emergency_stop()
            #     else:
            #         target_speed = max([self._behavior.max_speed, self._speed_limit]) + 10
            #         self._local_planner.set_speed(target_speed)
            #         control = self._local_planner.run_step(debug=debug)"""
            # else:
            #     #print("sono nell'incrocio e non ho trovato niente")
            #     #return self.decelerate(speed=20)
            #     target_speed = min([
            #         self._behavior.max_speed,
            #         self._speed_limit - self._behavior.speed_lim_dist])
            #     """self._local_planner.set_speed(target_speed)
            #     control = self._local_planner.run_step(debug=debug)"""
            #     #target_speed = max([self._behavior.max_speed, self._speed_limit]) + 10
            #     self._local_planner.set_speed(target_speed)
            #     control = self._local_planner.run_step(debug=debug)
        # 3.1: Intersection behavior with LANEFOLLOW RoadOption
        else:
            #print ("LANE FOLLOWING")
            actor_state, actor, distance = self.collision_and_car_avoid_manager(ego_vehicle_wp)
            if actor_state:
                if distance < self._behavior.braking_distance:
                    #print("ho trovato un ostacolo davanti molto vicino")
                    return self.emergency_stop()
                else:
                    control = self.car_following_manager(actor, distance)        
            else:
                #print("nessun ostacolo davanti")
                target_speed = min([
                    self._behavior.max_speed,
                    self._speed_limit - self._behavior.speed_lim_dist])
                self._local_planner.set_speed(target_speed)
                control = self._local_planner.run_step(debug=debug)

        return control
                     

    def emergency_stop(self):
        """
        Overwrites the throttle a brake values of a control to perform an emergency stop.
        The steering is kept the same to avoid going out of the lane when stopping during turns

            :param speed (carl.VehicleControl): control to be modified
        """
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = self._max_brake
        control.hand_brake = False
        return control

    def soft_stop(self):
        """
        This method is in charge of braking the vehicle
            :return control: carla.VehicleControl
        """
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = 0.2
        control.hand_brake = False
        return control

    def decelerate(self, speed=20):
        """
        This method is in charge of decelerating the vehicle when there's a turn ahead.
            :return control: carla.VehicleControl
        """
        control = carla.VehicleControl()
        self._local_planner.set_speed(speed)
        control = self._local_planner.run_step()
        return control
    
    def compute_angle(self, v1, v2):
        dot = v1.x*v2.x + v1.y*v2.y
        v1_magnitude = math.sqrt(v1.x**2 + v1.y**2)
        v2_magnitude = math.sqrt(v2.x**2 + v2.y**2)
        cosine = np.clip(dot / (v1_magnitude*v2_magnitude), -1, 1)
        angle_rad = math.acos(cosine)
        angle_deg = math.degrees(angle_rad)
        return angle_deg

    def get_forward_vector(self, vehicle):
        yaw_rad = vehicle.get_transform().rotation.yaw * 3.14159 / 180.0
        forward_vector = carla.Vector3D(x=-1*math.sin(yaw_rad), y=math.cos(yaw_rad), z=0)
        return forward_vector
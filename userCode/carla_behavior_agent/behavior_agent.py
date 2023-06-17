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
        self._speed_limit = self._vehicle.get_speed_limit()
        self._local_planner.set_speed(self._speed_limit)
        self._direction = self._local_planner.target_road_option
        self._vehicle_heading = self._vehicle.get_transform().rotation.yaw
        if self._direction is None:
            self._direction = RoadOption.LANEFOLLOW

        self._look_ahead_steps = int((self._speed_limit) / 10)

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
            print("LEFT OR RIGHT")
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), low_angle_th=45, up_angle_th=315, lane_offset=-1)
        else:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                    vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=30, lane_offset=0)
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
                # print("DISTANCE between {} and {}: {}".format(self.overtake_list[-1].type_id, v.type_id, str(distance)))
                if distance < 15:
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

    def _check_vehicles_objects_in_direction(self, waypoint, distance=45, offset=None, angles=(0, 180), check_other_lane=False, objects=True):
        vehicle_list = list(self._world.get_actors().filter("*vehicle*"))
        if objects:
            object_list = list(self._world.get_actors().filter("*static*"))
            vehicle_list.extend(object_list)
       
        vehicle_list = [v for v in vehicle_list if self.dist(v, self._vehicle) < distance and v.id != self._vehicle.id]
        # print("VEHICLE LIST: ", end="\n")
        # for v in vehicle_list:
        #     print(v.type_id, end="- ")
        #     print(self._map.get_waypoint(v.get_location()).lane_id, end="- ")
        #     print(self.dist(v, self._vehicle), end=", ")
        #     print(is_within_distance_test(v.get_transform(), self._vehicle.get_transform(), distance, (0, 180)), end=", ")
        # print()

        ego_wpt = self._map.get_waypoint(self._vehicle.get_location())

        vehicle_list = [v for v in vehicle_list if is_within_distance(v.get_transform(), ego_wpt.transform, distance, angles) and v.id != self._vehicle.id]
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
        return self._check_vehicles_objects_in_direction(waypoint, distance, None, (75, 105), check_other_lane=False)

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
    
    def check_vehicles_intersection(self, waypoint, distance=20):
        """
        This module is in charge of warning of obstacles in front of the vehicle.
        
            :param waypoint: current waypoint of the agent
            :param distance: distance to check for vehicles
            :return vehicle_state: True if there is a vehicle nearby, False if not
            :return vehicle: nearby vehicle
            :return distance: distance to nearby vehicle
            """
        state_right, _, _ = self._check_vehicles_objects_in_direction(waypoint, distance, 0, (45, 90), check_other_lane=False, objects=False)
        state_left, _, _ = self._check_vehicles_objects_in_direction(waypoint, distance, 0, (270, 315), check_other_lane=False, objects=False)
        print("STATE LEFT: " + str(state_left))
        print("STATE RIGHT: " + str(state_right))
        return state_left, state_right
            
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
        walker_list = [w for w in walker_list if dist(w) < 10]
        
        if len(walker_list) == 0:
            return False, None, None

        if self._direction == RoadOption.CHANGELANELEFT:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=90, lane_offset=1)
        else:
            walker_state, walker, distance = self._vehicle_obstacle_detected(walker_list, max(
                self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=60)

        return walker_state, walker, distance

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
        elif 2 * self._behavior.safety_time > ttc >= self._behavior.safety_time:
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
        object_list = [v for v in object_list if is_within_distance(v.get_transform(), waypoint.transform, 20, [0, 30])]
        print("STOP SIGN, SIGN ROAD ID: ", [self._map.get_waypoint(v.get_location()).road_id for v in object_list])
        print("STOP SIGN, EGO ROAD ID: ", waypoint.road_id)
        print("STOP IS WITHIN DISTANCE test: ", [is_within_distance_test(v.get_transform(), waypoint.transform, 20, [0, 30]) for v in object_list])
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

        stop_state, stop_sign = self.check_for_stop_sign(ego_vehicle_wp)

        if stop_state and not self._incoming_waypoint.is_junction:
            print("STOP SIGN")
            print("DISTANCE TO STOP: ", self.dist(self._vehicle, stop_sign))
            if self._stop_managed is None:
                print("controllo: ", stop_sign.id != self._stop_managed)
            print("VERO O FALSO: ", not self.wait_at_stop, stop_sign.id != self._stop_managed, self.dist(self._vehicle, stop_sign) < 6)
            
        
        # if stop_state and not self.wait_at_stop and stop_sign.id != self._stop_managed:
        if stop_state and not self.wait_at_stop and stop_sign.id != self._stop_managed and self.dist(self._vehicle, stop_sign) < 6 and not ego_vehicle_wp.is_junction:
            print("sono entrato nell'if dello stop")
            self._stop_counter = 100
            self.wait_at_stop = True
            self._stop_managed = stop_sign.id

        elif self.wait_at_stop and self._stop_counter > 0:
            print("sono entrato nel secondo if")
            print(self._stop_counter)
            self._stop_counter -= 1
            return self.emergency_stop()
        else:
            self.wait_at_stop = False
            
            
            #return self.emergency_stop()

        # 1: Red lights and stops behavior
        if self.traffic_light_manager():
            return self.emergency_stop()

        # 2.1: Pedestrian avoidance behaviors
        walker_state, walker, w_distance = self.pedestrian_avoid_manager(ego_vehicle_wp)

        if walker_state:
            print("ho trovato un pedone")
            # Distance is computed from the center of the two cars,
            # we use bounding boxes to calculate the actual distance
            distance = w_distance - max(
                walker.bounding_box.extent.y, walker.bounding_box.extent.x) - max(
                    self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)

            # Emergency brake if the car is very close.
            if distance < self._behavior.braking_distance:
                return self.emergency_stop()
        
        #print("EGO WP IS JUNCTION: ", ego_vehicle_wp.is_junction)
        #print("INCOMING WP IS JUNCTION: ", self._incoming_waypoint.is_junction)
        # if ego_vehicle_wp.is_junction or self._incoming_waypoint.is_junction:
        #     input()
        
        #2.3 Lane narrowing
        if not self._incoming_waypoint.is_junction and not ego_vehicle_wp.is_junction:
            
            narrowing_state = self.check_for_lane_narrowing(ego_vehicle_wp)

            if narrowing_state:
                self._local_planner.set_lateral_offset(-1.5)
                self._narrowing = True
                target_speed = min([
                    self._behavior.max_speed,
                    self._speed_limit - self._behavior.speed_lim_dist])
                self._local_planner.set_speed(target_speed)
                control = self._local_planner.run_step(debug=debug)
            else:
                self._local_planner.set_lateral_offset(0)
                self._narrowing = False
                target_speed = min([
                    self._behavior.max_speed,
                    self._speed_limit - self._behavior.speed_lim_dist])
                self._local_planner.set_speed(target_speed)
                control = self._local_planner.run_step(debug=debug)
                

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
                target_speed = min([self._behavior.max_speed, self._speed_limit])
                self._local_planner.set_speed(target_speed)
                control = self._local_planner.run_step(debug=debug)
                return control
            elif self.overtaking:
                print("Overtaking...", [x.type_id for x in self.overtake_list])
                if not len(self._local_planner._waypoints_queue) > 1:
                    # state, actor, distance = self.overtake_manager_old(ego_vehicle_wp, distance=75, check_lane=True)
                    state, actor, distance = self.check_beside(ego_vehicle_wp)
                    if ego_vehicle_wp.lane_id != self.original_lane:
                        state_front, actor_front, distance_front = self.check_front_overtaking(ego_vehicle_wp)
                    # print("overtake manager inside overtaking: ", state, actor, distance)
                    # print("dimensione ego vehicle: ", self._vehicle.bounding_box.extent.x*2)
                    #if not state and not actor in self.overtake_list:
                    if (actor is not None and actor.id == self.overtake_list[-1].id) or state_front:
                        # print("overtake finished")
                        if self.lane_change("left", self._vehicle_heading, 0, 1.50, 0.3):
                            self._ending_overtake = True
                    else:
                        self.lane_change("left", self._vehicle_heading, 0.85, 0, 0)

                target_speed = max([self._behavior.max_speed, self._speed_limit])
                self._local_planner.set_speed(target_speed)
                control = self._local_planner.run_step(debug=debug)
                return control
            
            # 2.2: Car following behaviors
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
            
                if not self._narrowing and not ego_vehicle_wp.is_junction:
                    if distance - 1  < self._behavior.braking_distance and not self.overtaking and actor.get_velocity().length() < 3.0:
                        if ego_vehicle_wp.left_lane_marking.type == carla.LaneMarkingType.Broken or ego_vehicle_wp.left_lane_marking.type == carla.LaneMarkingType.SolidBroken:
                            # print("velocità, actor: ", actor.get_velocity().length(), actor.type_id)
                            print("Trying to overtake...")
                            distance_to_last_obj = self.check_obstacles_to_overtake(ego_vehicle_wp)
                            if len(self.overtake_list) <= 2:
                                state, actor, _ = self.overtake_manager_old(ego_vehicle_wp, distance=65)
                            else:
                                state, actor, _ = self.overtake_manager_old(ego_vehicle_wp, distance=max(80, distance_to_last_obj*3))
                            print("Intersection closer:", self.closest_intersection(ego_vehicle_wp) > distance_to_last_obj*3)
                            if not state and self.closest_intersection(ego_vehicle_wp) > distance_to_last_obj*3:
                                    # print("change lane")
                                    self._waypoints_queue_copy = self._local_planner._waypoints_queue.copy()
                                    if self.lane_change("left", self._vehicle_heading, 0, 2, 1.5):
                                        self.overtaking = True
                                        self.original_lane = ego_vehicle_wp.lane_id
                                        target_speed = max([self._behavior.max_speed, self._speed_limit])
                                        self._local_planner.set_speed(target_speed)
                                        control = self._local_planner.run_step(debug=debug)
                                        return control
                        else:
                            print("Cannot change lane")
                        return self.emergency_stop()
                    else:
                        # print("Car following", actor)
                        control = self.car_following_manager(actor, distance)

        
        # 3: Intersection behavior
        elif ego_vehicle_wp.is_junction or (self._incoming_waypoint.is_junction and (self._incoming_direction in [RoadOption.LEFT, RoadOption.RIGHT])):
            print("Intersection with direction", self._incoming_direction)
            # draw incoming waypoint
            draw_waypoints(self._world, [self._incoming_waypoint], color=carla.Color(0, 0, 255))
            actor_state, actor, distance = self.collision_and_car_avoid_manager(ego_vehicle_wp)
            #actor_state_left, actor_state_right = self.check_vehicles_intersection(ego_vehicle_wp)
            # input()
            if actor_state:
                print("Collision and car avoid manager: ", actor_state, actor, distance)
                #input()
                if distance < self._behavior.braking_distance:
                    print("ho trovato un ostacolo davanti molto vicino")
                    return self.emergency_stop()
                else:
                    control = self.car_following_manager(actor, distance)
            else:
                # target_speed = min([
                #     self._behavior.max_speed,
                #     self._speed_limit - 5])
                target_speed = 30
                self._local_planner.set_speed(target_speed)
                control = self._local_planner.run_step(debug=debug)
        # 4: Normal behavior
        else:
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

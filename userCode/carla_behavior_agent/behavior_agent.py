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
        
        # Parameters for agent behavior
        self.try_overtake = False
        self._obstacle_to_overtake = None
        self.overtaking = False
        self.overtake_list = []
        #self.end_overtake = False
        self.old_queue = None
        self._prev_dist_obstacle = None
        self._count_dist_obstacle = 0
        self.original_lane = None
        self._narrowing = False

        # Parameters for agent behavior
        if behavior == 'cautious':
            self._behavior = Cautious()

        elif behavior == 'normal':
            self._behavior = Normal()

        elif behavior == 'aggressive':
            self._behavior = Aggressive()

    def _update_information(self):
        """
        This method updates the information regarding the ego
        vehicle based on the surrounding world.
        """
        self._speed = get_speed(self._vehicle)
        self._speed_limit = self._vehicle.get_speed_limit()
        self._local_planner.set_speed(self._speed_limit)
        self._direction = self._local_planner.target_road_option
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

    def collision_and_car_avoid_manager_bb(self, waypoint):
        bb_list = self._world.get_level_bbs(actor_type=carla.CityObjectLabel.Vehicles)
        # distance from waypoint to bb
        def dist(bb): return compute_distance(waypoint.transform.location, bb.location)
        bb_list = [bb for bb in bb_list if dist(bb) < 45]
        # sort by distance
        bb_list = sorted(bb_list, key=dist)
        print("BEFORE --- BB LIST: ", end="\n")
        for bb in bb_list:
            print(bb.type_id, end="- ")
            print(dist(bb), end=", ")
        print()
        # get bb of the vehicles in front of the ego vehicle
        bb_list = [bb for bb in bb_list if bb.location.x - self._vehicle.get_location().x > 0]
        print("AFTER --- BB LIST: ", end="\n")
        for bb in bb_list:
            print(bb.type_id, end="- ")
            print(dist(bb), end=", ")
        print()
        return False

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

        vehicle_list = list(self._world.get_actors().filter("*vehicle*"))
        object_list = list(self._world.get_actors().filter("*static*"))
        vehicle_list.extend(object_list)
        def dist(v): return v.get_location().distance(waypoint.transform.location)
        vehicle_list = [v for v in vehicle_list if dist(v) < 45 and v.id != self._vehicle.id]
        vehicle_list = sorted(vehicle_list, key=dist)
        print("VEHICLE LIST: ", end="\n")
        for i, v in enumerate(vehicle_list):
            if 'police' in v.type_id:
                # v.destroy()
                # vehicle_list.pop(i)
                print("POLICE CAR")
                # wpt = self._map.get_waypoint(v.get_location())
                # print(wpt.lane_id)
                # print(wpt.lane_type)
                # print(dist(v))
                # print("Angle: ", is_within_distance_test(v.get_transform(), waypoint.transform, 45, [0, 180]))
            else:
                # print(v.type_id, end="- ")
                wpt = self._map.get_waypoint(v.get_location())
                # print("Lane id: ", wpt.lane_id, end="- ")
                # print("Lane type: ", wpt.lane_type, end="- ")
                # print("Distance: ", dist(v), end="- ")
                # print("Angle: ", is_within_distance_test(v.get_transform(), waypoint.transform, 45, [0, 180]))
                # print("Velocity: ", v.get_velocity(), end="- ")
                # print("VERO O FALSO: ", v.get_velocity() == carla.Vector3D(0, 0, 0))
        print()

        # print("Direction: ", self._direction)

        if self._direction == RoadOption.CHANGELANELEFT:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limi1t / 2), up_angle_th=180, lane_offset=-1)
        elif self._direction == RoadOption.CHANGELANERIGHT:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=1)
        else:
            vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
                vehicle_list, max(
                    self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=30)

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
                print("DISTANCE between {} and {}: {}".format(self.overtake_list[-1].type_id, v.type_id, str(distance)))
                if distance < 11:
                    self.overtake_list.append(v)
                else:
                    break
        
        # print("OVERTAKE LIST: ", end="\n")
        # for v in self.overtake_list:
        #     print(v.type_id, end=", ")
        
        # print("DISTANCE by last object: " + str(dist(self.overtake_list[-1])))

    def overtake_manager_old(self, waypoint, direction=None, distance=45, offset=None):
        """
        This module is in charge of warning in case of a collision
        and managing possible tailgating chances.

            :param location: current location of the agent
            :param waypoint: current waypoint of the agent
            :return vehicle_state: True if there is a vehicle nearby, False if not
            :return vehicle: nearby vehicle
            :return distance: distance to nearby vehicle
        """
        if direction is None:
            direction = self._direction

        vehicle_list = list(self._world.get_actors().filter("*vehicle*"))
        object_list = list(self._world.get_actors().filter("*static*"))
        vehicle_list.extend(object_list)
        def dist(v): return v.get_location().distance(waypoint.transform.location)
        vehicle_list = [v for v in vehicle_list if dist(v) < distance and v.id != self._vehicle.id]
        # for v in vehicle_list:
        #     # get left lane waypoint of the vehicle
        #     left_wpt = self._map.get_waypoint(self._vehicle.get_location()).get_left_lane()
        #     check, angle = is_within_distance_test(v.get_transform(), left_wpt.transform, 45, [160, 180])
        #     print("IS WITHIN DISTANCE ", check, angle, v.type_id)
        left_wpt = self._map.get_waypoint(self._vehicle.get_location()).get_left_lane()
        # if left_wpt is None:
        #     return False, None, None
        vehicle_list = [v for v in vehicle_list if is_within_distance(v.get_transform(), left_wpt.transform, distance, [160, 180]) and v.id != self._vehicle.id]
        # print("BEFORE --- VEHICLE LIST inside overtake manager: ", end="\n")
        # for v in vehicle_list:
        #     print(v.type_id, end="- ")
        #     print(self._map.get_waypoint(v.get_location()).lane_id, end=", ")
        #     print("ANGLE INSIDE OVERTAKE MANAGER: ", is_within_distance_test(v.get_transform(), left_wpt.transform, distance, [0, 180]))
        # print()
        #looking opposite lane (?)
        if offset is None:
            if waypoint.lane_id < 0:
                lane_offset = -(waypoint.lane_id - 1)
            else:
                lane_offset = -(waypoint.lane_id + 1)
        else:
            lane_offset = 0
        vehicle_list = [v for v in vehicle_list if self._map.get_waypoint(v.get_location()).lane_id == waypoint.lane_id + lane_offset]
        print("EGO LANE ID:", waypoint.lane_id)
        print("AFTER --- VEHICLE LIST after filtering: ", end="\n")
        for v in vehicle_list:
            print(v.type_id, end="- ")
            print(self._map.get_waypoint(v.get_location()).lane_id, end=", ")
        print()
        for i, v in enumerate(vehicle_list):
            if "static.prop.trafficwarning" in v.type_id:
                continue
            loc = v.get_location()
            wp = self._map.get_waypoint(loc)
            draw_waypoints(self._world, [wp], color=carla.Color(0, 0, 255))
            # if "static.prop." in v.type_id:
            #     vehicle_list.pop(i)
            #     v.destroy()
        # sort by distance
        # print("ID + OFFSET " + str(waypoint.lane_id + lane_offset))
        vehicle_list = sorted(vehicle_list, key=dist)
        for v in vehicle_list:
            print(v, end=" ")
            print(self._map.get_waypoint(v.get_location()).lane_id)
            
        if len(vehicle_list) == 0:
            return False, None, None
        
        # if direction == RoadOption.CHANGELANELEFT:
        #     vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
        #         vehicle_list, max(
        #             self._behavior.min_proximity_threshold, self._speed_limit / 2 ) , up_angle_th=180, lane_offset=-1, obstacle_to_overtake=obstacle_to_overtake)
        # elif direction == RoadOption.CHANGELANERIGHT:
        #     vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
        #         vehicle_list, max(
        #             self._behavior.min_proximity_threshold, self._speed_limit / 2), up_angle_th=180, lane_offset=0, obstacle_to_overtake=obstacle_to_overtake)
        # else:
        #     print("Vehicle list: ", end=" ")
        #     for v in vehicle_list:
        #         print(v.type_id, end=" ")
        #     print("Distance: ", max(self._behavior.min_proximity_threshold, self._speed_limit / 2))
        #     vehicle_state, vehicle, distance = self._vehicle_obstacle_detected(
        #         vehicle_list, max(
        #             self._behavior.min_proximity_threshold, self._speed_limit / 3), up_angle_th=30, obstacle_to_overtake=obstacle_to_overtake, lane_offset=lane_offset)

        #     # Check for tailgating
        #     if not vehicle_state and direction == RoadOption.LANEFOLLOW \
        #             and not waypoint.is_junction and self._speed > 10 \
        #             and self._behavior.tailgate_counter == 0:
        #         self._tailgating(waypoint, vehicle_list)
        # print("STATE " + str(vehicle_state) + " " + str(vehicle))
        else:
            return True, vehicle_list[0], dist(vehicle_list[0])
            
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
        ########################################AGGIUNGERE CONTROLLO SU LANE ID!!!
        object_list = list(self._world.get_actors().filter("static.prop.constructioncone"))
        bicycle_namelist = ['vehicle.bh.crossbike', 'vehicle.diamondback.century', 'vehicle.gazelle.omafiets']
        bicycle_list = []
        for name in bicycle_namelist:
            bicycle_list.extend(self._world.get_actors().filter(name))      
        def dist(v): return v.get_location().distance(waypoint.transform.location)
        object_list = [v for v in object_list if dist(v) < 30]
        bicycle_list = [b for b in bicycle_list if dist(b) < 30]        
        #bicycle_list = [b for b in bicycle_list if dist(b) < 30 and self._map.get_waypoint(b.get_location()).lane_id > 1]
        print("OBJECT LIST: ", end="\n")
        for v in object_list:
            print(v.type_id, end=", ")
        print()
        print("BICYCLE LIST: ", end="\n")
        for b in bicycle_list:
            print(b.type_id, end=", ")
            print("Lane id: ", self._map.get_waypoint(b.get_location()).lane_id, end=", ")
        print()
        return (object_list, bicycle_list)


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
        #self.collision_and_car_avoid_manager(ego_vehicle_wp)
        def dist(v): return v.get_location().distance(ego_vehicle_wp.transform.location)
        if self.try_overtake:
            # print("Veichle lane id:" + str(ego_vehicle_wp.lane_id))
            draw_waypoints(world=self._world, waypoints=[ego_vehicle_wp], color=carla.Color(0, 255, 0))
            self.check_obstacles_to_overtake(ego_vehicle_wp)
            """if self.overtake_list[-1].type_id in ['vehicle.diamondback.century', 'vehicle.bh.crossbike']:
                print("Overtake bike with lane id ", self._map.get_waypoint(self.overtake_list[-1].get_location()).lane_id)
                self._local_planner.set_lateral_offset(1.5)
                target_speed = min([
                    self._behavior.max_speed,
                    self._speed_limit - self._behavior.speed_lim_dist])
                self._local_planner.set_speed(target_speed)
                control = self._local_planner.run_step(debug=debug)
                self.try_overtake = False
                self.overtaking = True
                self.original_lane = ego_vehicle_wp.lane_id
                return control
            else:"""
            distance = 60
            
            state, actor, _ = self.overtake_manager_old(ego_vehicle_wp, RoadOption.LANEFOLLOW, distance=distance)
            
            if not state:
                print("change lane")
                print(ego_vehicle_wp.lane_id)
                
                start_location = self._vehicle.get_location()
                start_waypoint = self._map.get_waypoint(start_location)
            
                plan = self.change_path(start_waypoint, int(dist(self.overtake_list[-1])-3), follow_direction=False, save_and_pop_queue=True)
                self._local_planner.set_global_plan(plan, clean_queue=False, create_new=True)
                self._local_planner.set_global_plan(self.old_queue, clean_queue=False, create_new=False)
                target_speed = min([
                    self._behavior.max_speed,
                    self._speed_limit - self._behavior.speed_lim_dist])
                self._local_planner.set_speed(target_speed)
                control = self._local_planner.run_step(debug=debug)
                self.try_overtake = False
                self.overtaking = True
                self.original_lane = ego_vehicle_wp.lane_id
                return control
            else:
                print("can't change lane", actor)
                # self.try_overtake = False
                return self.emergency_stop()
        
        if self.overtaking:
            print("Overtake list")
            for v in self.overtake_list:
                print(v.type_id, end=", ")
            print()
            print("Distance from last object: " + str(dist(self.overtake_list[-1])) + "m")
            if is_within_distance(self._vehicle.get_transform(), self.overtake_list[-1].get_transform(), 10, [175, 185]) or \
                    is_within_distance(self._vehicle.get_transform(), self.overtake_list[-1].get_transform(), 10, [85, 95]):
                print("Overtake finished")
                #self._local_planner.set_lateral_offset(0)
                self.overtaking = False
                self._obstacle_to_overtake = None
                self.overtake_list = []
            else:
                if ego_vehicle_wp.lane_id != self.original_lane:
                    state, actor, _ = self.overtake_manager_old(ego_vehicle_wp, direction=RoadOption.CHANGELANERIGHT, offset=0, distance=60)
                else:
                    state, actor, _ = self.overtake_manager_old(ego_vehicle_wp, direction=RoadOption.CHANGELANERIGHT, distance=60)
                if state and ego_vehicle_wp.lane_id == self.original_lane:
                    print("emergency overtake")
                    #if not self.overtake_list[-1].type_id in ['vehicle.diamondback.century', 'vehicle.bh.crossbike']:
                    return self.emergency_stop()

            target_speed = min([
                self._behavior.max_speed,
                self._speed_limit - self._behavior.speed_lim_dist]) + 20
            self._local_planner.set_speed(target_speed)
            control = self._local_planner.run_step(debug=debug)
            print("NO emergency overtake")
            return control

        # 1: Red lights and stops behavior
        if self.traffic_light_manager():
            return self.emergency_stop()

        # 2.1: Pedestrian avoidance behaviors
        walker_state, walker, w_distance = self.pedestrian_avoid_manager(ego_vehicle_wp)

        if walker_state:
            print("ho trovato un pedone KTM")
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
            obj_list, bic_list = self.check_for_lane_narrowing(ego_vehicle_wp)

            if len(obj_list) > 0:
                #print("Lane narrowing right")
                self._local_planner.set_lateral_offset(-1.5)
                self._narrowing = True
                target_speed = min([
                    self._behavior.max_speed,
                    self._speed_limit - self._behavior.speed_lim_dist])
                self._local_planner.set_speed(target_speed)
                control = self._local_planner.run_step(debug=debug)
                #elif len(bic_list) > 0:
                # print("Lane narrowing left")
                # offset = self._vehicle.bounding_box.extent.y + 1
                # self._local_planner.set_lateral_offset(offset=offset)
                # self._narrowing = True
                # target_speed = min([
                #     self._behavior.max_speed,
                #     self._speed_limit - self._behavior.speed_lim_dist])
                # self._local_planner.set_speed(target_speed)
                # control = self._local_planner.run_step(debug=debug)
            else:
                self._local_planner.set_lateral_offset(0)
                self._narrowing = False
                target_speed = min([
                    self._behavior.max_speed,
                    self._speed_limit - self._behavior.speed_lim_dist])
                self._local_planner.set_speed(target_speed)
                control = self._local_planner.run_step(debug=debug)

            # 2.2: Car following behaviors
            actor_state, actor, distance = self.collision_and_car_avoid_manager(ego_vehicle_wp)
            # self.collision_and_car_avoid_manager_bb(ego_vehicle_wp)
            #print("IF")
            #print("Collision and car avoid manager: ", actor_state, actor, distance)
            
            if actor_state:
                # Distance is computed from the center of the two cars,
                # we use bounding boxes to calculate the actual distance
                # print("Actor: ", actor)
                # print("Distance to car: ", distance, "m")
                distance = distance - max(
                    actor.bounding_box.extent.y, actor.bounding_box.extent.x) - max(
                        self._vehicle.bounding_box.extent.y, self._vehicle.bounding_box.extent.x)
                # print("Distance to car (bounding box): ", distance, "m")
                # Emergency brake if the car is very close.
                # if actor.type_id == "static.prop.trafficwarning":
                #     print("Traffic warning")
                #     if distance < self._behavior.braking_distance:
                #         print("Emergency stop (traffic warning)")
                #         self.try_overtake = True
                #         return self.emergency_stop()
                # else:
                
                """if 'police' in actor.type_id and distance - 1  < self._behavior.braking_distance and not self.overtaking:
                    print("FERMA! Ho trovato la macchina della polizia!")
                    print("POLICE VELOCITY: ", actor.get_velocity())
                    self.try_overtake = True
                    self._obstacle_to_overtake = actor
                    return self.emergency_stop()"""
                
                # print("NO POLICE")
            
                if not self._narrowing and not ego_vehicle_wp.is_junction:
                    if distance - 1  < self._behavior.braking_distance and not self.overtaking and actor.get_velocity() == carla.Vector3D(0, 0, 0):
                        # print("FERMA! Ho trovato il traffic warning")
                        self.try_overtake = True
                        self._obstacle_to_overtake = actor
                        return self.emergency_stop()
                    else:
                        print("Car following")
                        control = self.car_following_manager(actor, distance)

        
        # 3: Intersection behavior
        elif ego_vehicle_wp.is_junction or (self._incoming_waypoint.is_junction and (self._incoming_direction in [RoadOption.LEFT, RoadOption.RIGHT])):
            print("Intersection with direction", self._incoming_direction)
            actor_state, actor, distance = self.collision_and_car_avoid_manager(ego_vehicle_wp)
            print("ELIF")
            print("Collision and car avoid manager: ", actor_state, actor, distance)
            if actor_state:
                return self.emergency_stop()
            else:
                target_speed = min([
                    self._behavior.max_speed,
                    self._speed_limit - 5])
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

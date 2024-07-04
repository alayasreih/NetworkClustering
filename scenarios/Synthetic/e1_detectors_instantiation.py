import sumolib
import pandas as pd
import random
import os
import json


def parse_network_lanes(network_path):

    network = sumolib.net.readNet(network_path, withInternal=False)
    all_edges = network.getEdges(withInternal=False)

    all_lanes = []
    for edge in all_edges:
        edge_id = edge.getID()
        edge_type = edge.getType()
        lanes = edge.getLanes()
        for lane in lanes:
            lane_id = lane.getID()
            length = lane.getLength()
            allowed = lane._allowed   # getPermissions() didn't work !!
            all_lanes.append({'edge_id': edge_id, 'edge_type': edge_type, 'lane_id': lane_id, 'length': length,
                              'allowed': ', '.join(allowed)})
    df_all_lanes = pd.DataFrame(all_lanes)

    # filter motorised lanes (by allowed vehicle classes)
    irrelevant_classes = ['pedestrian', 'bicycle', 'pedestrian, bicycle', 'rail']
    df_lanes = df_all_lanes[~df_all_lanes.allowed.isin(irrelevant_classes)]

    return df_lanes


def general_placement(df_lanes, distance_to_downstream, road_categories_covered, coverage_percentage):

    """
    This function

    :param: network_path: path to the scenario directory (string)
    :param: distance_to_downstream: (scalar)
    :param: road_categories_covered: detectors measurements' frequency (list)
    :param: coverage_percentage: (scalar)
    """

    # filter lanes by road category
    if not road_categories_covered == ['all']:
        df_lanes = df_lanes[df_lanes.edge_type.isin(road_categories_covered)]

    # filter lanes by length (minimum length should be 10 meters more than the distance_to_downstream) => assumption
    min_length = 10 + distance_to_downstream
    df_lanes = df_lanes.loc[df_lanes.length >= min_length]

    # sample edges based on coverage percentage
    if not coverage_percentage == 100:
        all_edges = list(df_lanes.edge_id.unique())
        sample_size = round((coverage_percentage/100) * len(all_edges))
        edges_covered = random.sample(all_edges, sample_size)
        df_lanes = df_lanes[df_lanes.edge_id.isin(edges_covered)]

    # determine detectors positions and save the results as {lane_id: detector_position}
    lane_id = list(df_lanes.lane_id)
    detector_pos = list(round(length - distance_to_downstream) for length in list(df_lanes.length))
    lanes_dict = {k: v for k, v in zip(lane_id, detector_pos)}

    return lanes_dict


def specific_placement(df_lanes, placement_attr):

    """
    This function

    :param: placement_attr: {'road_category': ({'coverage_percentage': value}, {'distance_to_downstream': value})}(dict)

    """

    # filter lanes by road category
    road_categories_covered = placement_attr.keys()
    df_lanes = df_lanes[df_lanes.edge_type.isin(road_categories_covered)]

    lanes_dict = {}  # to save the results as {lane_id: detector_position}
    for road_category, attr in placement_attr.items():
        temp = df_lanes.loc[df_lanes.edge_type == road_category]

        # filter lanes by length (minimum length should be 10 meters more than the distance_to_downstream) => assumption
        min_length = 10 + attr['distance_to_downstream']
        temp = temp.loc[temp.length >= min_length]

        # sample edges based on coverage percentage
        if not attr['coverage_percentage'] == 100:
            all_edges = list(temp.edge_id.unique())
            sample_size = round((attr['coverage_percentage'] / 100) * len(all_edges))
            edges_covered = random.sample(all_edges, sample_size)
            temp = temp[temp.edge_id.isin(edges_covered)]

        # determine detectors positions and update
        lane_id = list(temp.lane_id)
        detector_pos = list(round(length - attr['distance_to_downstream']) for length in list(temp.length))
        lanes_dict.update({k: v for k, v in zip(lane_id, detector_pos)})

    return lanes_dict


def generate_detectors_def(scenario_dir, lanes_dict, frequency, output_file_name):

    """
    This function uses lanes id's and detectors position to generate detectors definitions in an additional-file

    :param: scenario_dir: path to the scenario directory (string)
    :param: lanes_dict: {lane_id: detector_position} (dictionary)
    :param: frequency: detectors measurements' frequency (scalar)
    """

    detectors_path = os.path.join(scenario_dir, 'detectors.add.xml').replace(os.sep, '/')
    detectors_output_path = os.path.join('output', output_file_name).replace(os.sep, '/')

    with open(detectors_path, 'w') as file:
        file.write('<additional>\n')

        for lane_id, detector_pos in lanes_dict.items():
            file.write(f'\t<e1Detector id="E1_{lane_id}" lane="{lane_id}" '
                       f'pos="{detector_pos}" freq="{frequency}" '
                       f'friendlyPos="true" file="{detectors_output_path}"/>\n')

        file.write('</additional>')
    file.close()

    return
# ---------------------------------------------------------------------------------------------------------------------


def instantiating_e1_detectors(**detectparam):

    # check network file
    network_path = os.path.join(detectparam['scenario_dir'], detectparam['network_file']).replace(os.sep, '/')
    if not os.path.isfile(network_path):
        raise IOError('could not find SUMO network file!')

    # return a dictionary {lane_id: detector_position} based on the defined requirements
    lanes_df = parse_network_lanes(network_path)
    if detectparam['detectors_placement_method'] == 'general':
        placement_attr = detectparam['general_placement_attributes']
        lanes_covered = general_placement(lanes_df, placement_attr['distance_to_downstream'],
                                          placement_attr['road_categories'], placement_attr['coverage_percentage'])

    else:
        placement_attr = detectparam['specific_placement_attributes']
        lanes_covered = specific_placement(lanes_df, placement_attr)

    # generate detectors definition
    generate_detectors_def(detectparam['scenario_dir'], lanes_covered, detectparam['frequency'], detectparam['output_file_name'])

    # save parameters
    #param_path = os.path.join(detectparam['scenario_dir'], 'detectors_instantiating_param.json').replace(os.sep, '/')
    #with open(param_path, 'w') as fp:
        #json.dump(detectparam, fp, indent=4)

    return


if __name__ == "__main__":

    general_detect_placement_attr = {'road_categories': ['all'],
                                     'coverage_percentage': 80,
                                     'distance_to_downstream': 20}

    specific_detect_placement_attr = {'highway.primary': {'coverage_percentage': 80, 'distance_to_downstream': 30},
                                      'highway.secondary': {'coverage_percentage': 60, 'distance_to_downstream': 25}}

    detectparam = {'scenario_dir': 'Synthetic',
                   'network_file': 'network.net.xml',
                   'detectors_placement_method': 'general',
                   'general_placement_attributes': general_detect_placement_attr,
                   'specific_placement_attributes': specific_detect_placement_attr,
                   'frequency': 300,
                   'output_file_name': 'detectors_output.xml'}

    instantiating_e1_detectors(**detectparam)

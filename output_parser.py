import os
import pickle
import pandas as pd
import xml.etree.ElementTree as ET
from abc import abstractmethod


class OutputParser:
    def __init__(self, file_path, output_dir):
        self.file_path = file_path
        self.output_dir = output_dir

    def save_pickle(self, obj, file_name):
        file_path = os.path.join(self.output_dir, file_name).replace(os.sep, '/')
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)

    @abstractmethod
    def process_and_save(self):
        pass


class EdgeOutputParser(OutputParser):
    def process_and_save(self):
        tree = ET.parse(self.file_path)
        root = tree.getroot()

        records = []
        for interval in root.iter('interval'):
            timestep = interval.get('begin')
            for edge in interval.findall('edge'):
                edge_id = edge.get('id')
                speed = edge.get('speed')
                density = edge.get('density')
                lane_density = edge.get('laneDensity')
                occupancy = edge.get('occupancy')
                sampled_seconds = edge.get('sampledSeconds')
                records.append((timestep, edge_id, sampled_seconds, speed, occupancy, density, lane_density))

        output_edges = pd.DataFrame(records,
                                    columns=['timestep', 'edge_id', 'sampled_seconds', 'speed', 'occupancy',
                                             'density', 'lane_density']).fillna(0).astype(
            {'timestep': float, 'sampled_seconds': float, 'speed': float, 'occupancy': float, 'density': float})

        output_edges['flow'] = output_edges['speed'] * 3.6 * output_edges['density']

        self.save_pickle(output_edges, 'output_edges.pkl')


class LaneOutputParser(OutputParser):
    def process_and_save(self):
        tree = ET.parse(self.file_path)
        root = tree.getroot()

        records = []
        for interval in root.iter('interval'):
            timestep = interval.get('begin')
            for edge in interval.findall('edge'):
                edge_id = edge.get('id')
                for lane in edge.findall('lane'):
                    lane_id = lane.get('id')
                    speed = lane.get('speed')
                    density = lane.get('laneDensity')
                    occupancy = lane.get('occupancy')
                    sampled_seconds = lane.get('sampledSeconds')
                    records.append((timestep, edge_id, lane_id, sampled_seconds, speed, occupancy, density))

        output_lanes = pd.DataFrame(records,
                                    columns=['timestep', 'edge_id', 'lane_id', 'sampled_seconds', 'speed', 'occupancy',
                                             'density']).fillna(0).astype(
            {'timestep': float, 'sampled_seconds': float, 'speed': float, 'occupancy': float, 'density': float})

        output_lanes['flow'] = output_lanes['speed'] * 3.6 * output_lanes['density']

        self.save_pickle(output_lanes, 'output_lanes.pkl')


class VehRouteOutputParser(OutputParser):
    def process_and_save(self):
        tree = ET.parse(self.file_path)
        root = tree.getroot()

        vehroute = []
        for veh in root.iter('vehicle'):
            veh_id = veh.get('id')
            depart = float(veh.get('depart'))
            arrival = float(veh.get('arrival')) if not veh.get('arrival') is None else None
            route = veh.find('route')
            edges = route.get('edges').split()
            exit_times = [float(t) for t in route.get('exitTimes').split()]
            entry_times = [depart] + exit_times[:-1]
            travel_times = [exit - entry for entry, exit in zip(entry_times, exit_times)]

            route = pd.DataFrame(
                {'edge_id': edges, 'entry_time': entry_times, 'exit_time': exit_times, 'travel_time': travel_times}
            )
            route['veh_id'] = veh_id
            route['veh_depart'] = depart
            route['veh_arrival'] = arrival
            vehroute.append(route)

        vehroute = pd.concat(vehroute)

        self.save_pickle(vehroute, 'output_vehroute.pkl')


class FCDOutputParser(OutputParser):
    def process_and_save(self):
        tree = ET.parse(self.file_path)
        root = tree.getroot()

        vehfcd = []
        for timestep in root.iter('timestep'):
            time = timestep.get('time')
            for veh in timestep.findall('vehicle'):
                veh_id = veh.get('id')
                speed = veh.get('speed')
                lane = veh.get('lane')
                pos = veh.get('pos')
                vehfcd.append((veh_id, time, lane, pos, speed))

        vehfcd = pd.DataFrame(vehfcd, columns=['veh_id', 'time', 'lane_id', 'position', 'speed'])
        vehfcd.astype({'time': float, 'position': float, 'speed': float})
        vehfcd.sort_values(by=['veh_id', 'time'], ignore_index=True, inplace=True)

        self.save_pickle(vehfcd, 'output_fcd.pkl')


class DetectorsOutputParser(OutputParser):
    def process_and_save(self):
        tree = ET.parse(self.file_path)
        root = tree.getroot()

        records = []
        for interval in root.iter('interval'):
            timestep = interval.get('begin')
            detector_id = interval.get('id')
            no_veh = interval.get('nVehContrib')
            flow = interval.get('flow')
            occupancy = interval.get('occupancy')
            speed = interval.get('speed')
            harmonic_mean_speed = interval.get('harmonicMeanSpeed')
            mean_veh_length = interval.get('length')
            records.append(
                (timestep, detector_id, no_veh, flow, occupancy, speed, harmonic_mean_speed, mean_veh_length))

        output_detectors = pd.DataFrame(records,
                                        columns=['timestep', 'detector_id', 'nVehContrib', 'flow', 'occupancy', 'speed',
                                                 'harmonic_mean_speed', 'mean_veh_length']).astype(
            {'timestep': float, 'nVehContrib': float, 'flow': float, 'occupancy': float, 'speed': float,
             'harmonic_mean_speed': float, 'mean_veh_length': float})

        output_detectors['density'] = (output_detectors['occupancy'] / output_detectors['mean_veh_length']) * 10

        self.save_pickle(output_detectors, 'output_detectors.pkl')


# __________________________________________________________________________________________________________________


if __name__ == '__main__':

    # define scenario
    scr = 'Zurich'       # Ingolstadt- detectors_output.xml   # Zurich- dets_measurement.xml  # Synthetic

    # define simulation output paths
    lanes_output = r'scenarios\{}\output\lane_output.xml'.format(scr)
    edges_output = r'scenarios\{}\output\edge_output.xml'.format(scr)
    vehroute_output = r'scenarios\{}\output\vehroute_output.xml'.format(scr)
    detectors_output = r'scenarios\{}\output\dets_measurement.xml'.format(scr)
    fcd_output = r'scenarios\{}\output\fcd_output.xml'.format(scr)

    # define simulation output directory
    output_dir = r'data\{}\output'.format(scr)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # parse simulation output
    parsers = [
        EdgeOutputParser(edges_output, output_dir),
        LaneOutputParser(lanes_output, output_dir),
        VehRouteOutputParser(vehroute_output, output_dir),
        DetectorsOutputParser(detectors_output, output_dir),
        # FCDOutputParser(fcd_output, output_dir)
    ]

    for parser in parsers:
        parser.process_and_save()

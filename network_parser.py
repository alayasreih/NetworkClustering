import os
import pickle
import sumolib
import pandas as pd
import networkx as nx
from shapely.geometry import Point, LineString
from abc import abstractmethod


class NetworkDataProcessor:
    def __init__(self, network_path, network_dir):
        self.network = sumolib.net.readNet(network_path)
        self.network_dir = network_dir

    def save_pickle(self, obj, file_name):
        file_path = os.path.join(self.network_dir, file_name).replace(os.sep, '/')
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)

    @abstractmethod
    def process_and_save(self):
        pass


class LanesProcessor(NetworkDataProcessor):
    def process_and_save(self):
        edges = self.network.getEdges()

        list_lanes = []
        for edge in edges:
            edge_id = edge.getID()
            edge_function = edge.getFunction()
            edge_type = edge.getType()
            lanes = edge.getLanes()
            for lane in lanes:
                id = lane.getID()
                length = lane.getLength()
                width = lane.getWidth()
                allowed = lane._allowed
                list_lanes.append((edge_id, edge_function, edge_type, id, length, width, ', '.join(allowed)))

        df_lanes = pd.DataFrame(list_lanes,
                                columns=['edge_id', 'edge_function', 'edge_type', 'lane_id', 'length', 'width',
                                         'allowed'
                                         ]
                                )

        self.save_pickle(df_lanes, 'lanes.pkl')


class NodesProcessor(NetworkDataProcessor):
    def process_and_save(self):
        nodes = self.network.getNodes()

        list_nodes = []
        for node in nodes:
            node_id = node.getID()
            x, y = node.getCoord()
            try : lon, lat = self.network.convertXY2LonLat(x, y)
            except: lon, lat = x, y
            geometry = Point(lon, lat)
            connected_edges = []
            internal_edges = []
            incoming = node.getIncoming()
            for edge in incoming:
                id = edge.getID()
                if edge.getFunction() == 'internal':
                    internal_edges.append(id)
                else:
                    connected_edges.append(id)
            outgoing = node.getOutgoing()
            for edge in outgoing:
                id = edge.getID()
                if edge.getFunction() == 'internal':
                    internal_edges.append(id)
                else:
                    connected_edges.append(id)
            list_nodes.append((node_id, geometry, list(set(connected_edges)), list(set(internal_edges))))

        df_nodes = pd.DataFrame(list_nodes, columns=['node_id', 'geometry', 'connected_edges', 'internal_edges'])

        self.save_pickle(df_nodes, 'nodes.pkl')


class EdgesProcessor(NetworkDataProcessor):
    def process_and_save(self):
        edges = self.network.getEdges()

        list_edges = []
        for edge in edges:
            id = edge.getID()
            function = edge.getFunction()
            type = edge.getType()
            length = edge.getLength()
            shape = edge.getShape()
            try: shape_global = [self.network.convertXY2LonLat(x, y) for (x, y) in shape]
            except: shape_global = shape
            geometry = LineString(shape_global)
            try: from_node, to_node = edge.getFromNode().getID(), edge.getToNode().getID()
            except: from_node, to_node = None, None
            list_edges.append((id, length, function, type, geometry, from_node, to_node))

        df_edges = pd.DataFrame(list_edges,
                                columns=['edge_id', 'length', 'function', 'type', 'geometry', 'from_node', 'to_node'])

        self.save_pickle(df_edges, 'edges.pkl')
        return list_edges


class ConnectionsProcessor(NetworkDataProcessor):
    def process_and_save(self):
        connections = [connection for node in self.network.getNodes() for connection in node.getConnections()]

        list_connections = []
        for connection in connections:
            from_edge = connection.getFrom().getID()
            to_edge = connection.getTo().getID()
            connecting_lane = connection.getViaLaneID()
            list_connections.append((from_edge, to_edge, connecting_lane))

        df_connections = pd.DataFrame(list_connections, columns=['from_edge', 'to_edge', 'connecting_lane'])

        self.save_pickle(df_connections, 'connections.pkl')

        return list_connections


class DetectorsProcessor(NetworkDataProcessor):
    def __init__(self, network_path, network_dir, detectors_path):
        super().__init__(network_path, network_dir)
        self.detectors_path = detectors_path

    def process_and_save(self):
        detectors = sumolib.sensors.inductive_loop.read(self.detectors_path)

        list_detectors = []
        for detector in detectors:
            detector_id = detector.id
            lane_id = detector.lane
            lane_length = self.network.getLane(lane_id).getLength()
            pos = detector.pos
            pos_adjusted = lane_length + pos if pos < 0 else pos
            x, y = sumolib.geomhelper.positionAtShapeOffset(self.network.getLane(lane_id).getShape(), pos_adjusted)
            try: lon, lat = self.network.convertXY2LonLat(x, y)
            except: lon, lat = x, y
            geometry = Point(lon, lat)
            list_detectors.append((detector_id, lane_id, lane_length, pos, geometry))

        df_detectors = pd.DataFrame(list_detectors,
                                    columns=['detector_id', 'lane_id', 'lane_length', 'pos', 'geometry'])

        self.save_pickle(df_detectors, 'detectors.pkl')


class NetworkGraphBuilder:
    def __init__(self, edges, connections, network_dir):
        self.edges = edges
        self.connections = connections
        self.network_dir = network_dir

    def build_networkx_graph(self):
        G = nx.Graph()

        for edge in self.edges:
            G.add_node(edge[0], weight=edge[1])

        for connection in self.connections:
            if connection[0] in G and connection[1] in G:
                G.add_edge(connection[0], connection[1], label=connection[2])

        graph_path = os.path.join(self.network_dir, 'network_graph.pkl')
        with open(graph_path, 'wb') as file:
            pickle.dump(G, file)


# __________________________________________________________________________________________________________________


if __name__ == '__main__':

    # define scenario
    scr = 'Zurich'                          # Ingolstadt                      # Zurich               # Synthetic
    network_file = 'CZHAST.net.xml'         # ingolstadt_24h.net.xml.gz       # CZHAST.net.xml       # network.net.xml
    detectors_file = 'CZHASTrealloops.xml'  # FINAL_detector_location.add.xml # CZHASTrealloops.xml  # detectors.add.xml

    # define files paths
    network = r'scenarios/{}/{}'.format(scr, network_file)
    detectors = r'scenarios/{}/{}'.format(scr, detectors_file)

    # define network directory
    network_dir = r'data/{}/network'.format(scr)
    if not os.path.isdir(network_dir):
        os.makedirs(network_dir)

    # parse network data
    processors = [
        NodesProcessor(network, network_dir),
        LanesProcessor(network, network_dir),
        EdgesProcessor(network, network_dir),
        ConnectionsProcessor(network, network_dir),
        DetectorsProcessor(network, network_dir, detectors)
    ]

    list_edges = None
    list_connections = None

    for processor in processors:
        if isinstance(processor, EdgesProcessor):
            list_edges = processor.process_and_save()
        elif isinstance(processor, ConnectionsProcessor):
            list_connections = processor.process_and_save()
        else:
            processor.process_and_save()

    # build networkx graph
    graph_builder = NetworkGraphBuilder(list_edges, list_connections, network_dir)
    graph_builder.build_networkx_graph()

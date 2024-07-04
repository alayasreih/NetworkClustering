import os
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd


class InputData:
    def __init__(self, dir):
        self.dir = dir

    @staticmethod
    def load_pickle(file_path):
        with open(file_path, 'rb') as file:
            obj = pickle.load(file)
        return obj

    def save_pickle(self, obj, file_name):
        file_path = os.path.join(self.dir, file_name)
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
        return


class DataStandardiser(InputData):
    def __init__(self, dir):
        super().__init__(dir)

    @staticmethod
    def get_bbox_edges(edges, bbox):
        gdf = gpd.GeoDataFrame(edges, geometry='geometry')
        return gdf[gdf.intersects(bbox)].edge_id.unique()

    @staticmethod
    def get_motorised(lanes):
        irrelevant = ('pedestrian', 'bicycle', 'rail')
        return [lane.lane_id for _, lane in lanes.iterrows() if not set(lane.allowed.split(',')).issubset(irrelevant)]

    @staticmethod
    def get_motorway(lanes):
        irrelevant = ['highway.motorway', 'highway.motorway_link']
        return [lane.lane_id for _, lane in lanes.iterrows() if lane.edge_type in irrelevant]

    @staticmethod
    def get_residential(lanes):
        irrelevant = [
            'highway.living_street', 'highway.residential', 'highway.path', 'highway.service', 'highway.track',
            'railway.rail', 'highway.footway', 'highway.steps', 'highway.cycleway', 'cycleway.lane|highway.service',
            'cycleway.lane|highway.residential', 'cycleway.lane|cycleway.track|highway.residential',
            'cycleway.track|highway.service', 'cycleway.track|highway.residential',
            'cycleway.opposite_lane|highway.residential', 'cycleway.opposite_track|highway.residential'
        ]
        return [lane.lane_id for _, lane in lanes.iterrows() if lane.edge_type in irrelevant]

    @staticmethod
    def get_unused(output_lanes):
        return [edge for edge, df in output_lanes.groupby('edge_id') if
                (df.speed == 0).all() and (df.density == 0).all()]

    @staticmethod
    def get_unused_detct(output_detectors):
        return [detct for detct, df in output_detectors.groupby('detector_id') if
                (df.speed == 0).all() and (df.density == 0).all()]

    def standardise_input(self, edges, lanes, output_lanes, bbox=None, filter_motorway=False, filter_residential=False,
                          file_name='standard_input.pkl'):

        # restrict the scr to a specific area if bbox is provided
        if not bbox is None:
            scr_edges = self.get_bbox_edges(edges, bbox)
            lanes = lanes.loc[lanes.edge_id.isin(scr_edges)]
            edges = edges.loc[edges.edge_id.isin(scr_edges)]
            output_lanes = output_lanes.loc[output_lanes.edge_id.isin(scr_edges)]

        print(f'Number of edges: {output_lanes.edge_id.nunique()}')
        print(f'Number of lanes: {output_lanes.lane_id.nunique()}\n')

        # filter out non-motorised lanes
        motorised_lanes = self.get_motorised(lanes)
        output_lanes = output_lanes.loc[output_lanes.lane_id.isin(motorised_lanes)]

        print(f'Number of motorised lanes: {len(motorised_lanes)}')
        print(f'Number of edges after filtration: {output_lanes.edge_id.nunique()}')
        print(f'Number of lanes after filtration: {output_lanes.lane_id.nunique()}\n')

        # filter out motorway lanes - optional
        if filter_motorway:
            motorway_lanes = self.get_motorway(lanes)
            output_lanes = output_lanes.loc[~output_lanes.lane_id.isin(motorway_lanes)]

            print(f'Number of motorway lanes: {len(motorway_lanes)}')
            print(f'Number of edges after filtration: {output_lanes.edge_id.nunique()}')
            print(f'Number of lanes after filtration: {output_lanes.lane_id.nunique()}\n')

        # filter out residential lanes - optional
        if filter_residential:
            residential_lanes = self.get_residential(lanes)
            output_lanes = output_lanes.loc[~output_lanes.lane_id.isin(residential_lanes)]

            print(f'Number of residential lanes: {len(residential_lanes)}')
            print(f'Number of edges after filtration: {output_lanes.edge_id.nunique()}')
            print(f'Number of lanes after filtration: {output_lanes.lane_id.nunique()}\n')

        # filter out unused edges
        unused_edges = self.get_unused(output_lanes)
        output_lanes = output_lanes.loc[~output_lanes.edge_id.isin(unused_edges)]

        print(f'Number of unused edges: {len(unused_edges)}')
        print(f'Number of edges after filtering out unused edges: {output_lanes.edge_id.nunique()}')
        print(f'Number of lanes after filtering out unused edges: {output_lanes.lane_id.nunique()}\n')

        standard_input = pd.merge(output_lanes, edges[['edge_id', 'length', 'geometry', 'type']], on='edge_id',
                                  how='left')
        standard_input = standard_input[
            ['timestep', 'edge_id', 'length', 'geometry', 'lane_id', 'sampled_seconds', 'speed', 'occupancy', 'density',
             'flow']]

        self.save_pickle(standard_input, file_name)

    def standardise_detectors_input(self, detectors, edges, lanes, output_detectors, output_lanes, bbox=None,
                                    meso=False, filter_motorway=False, filter_residential=False,
                                    files_names=(
                                    'detectors_standard_input.pkl', 'detectors_input.csv', 'detectors_loc.csv')
                                    ):

        # restrict to specific area if bbox is provided
        if bbox is not None:
            scr_edges = self.get_bbox_edges(edges, bbox)
            lanes = lanes.loc[lanes.edge_id.isin(scr_edges)]
            detectors = detectors.loc[detectors.lane_id.isin(lanes.lane_id.unique())]

        print(f'Initial no. detectors: {detectors.detector_id.nunique()}')

        if meso:
            output_lanes = output_lanes.loc[output_lanes.lane_id.isin(lanes.lane_id.unique())]
            # filter out motorway lanes - optional
            if filter_motorway:
                motorway_lanes = self.get_motorway(lanes)
                output_lanes = output_lanes.loc[~output_lanes.lane_id.isin(motorway_lanes)]
            # filter out residential lanes - optional
            if filter_residential:
                residential_lanes = self.get_residential(lanes)
                output_lanes = output_lanes.loc[~output_lanes.lane_id.isin(residential_lanes)]
            # filter out unused edges
            unused_edges = self.get_unused(output_lanes)
            output_lanes = output_lanes.loc[~output_lanes.edge_id.isin(unused_edges)]

            standard_input = pd.merge(
                output_lanes, detectors, on='lane_id', how='left'
            ).dropna(
                subset=['detector_id'], ignore_index=True
            )
            standard_input['pos'] = standard_input['lane_length'] / 2
        else:
            output_detectors = output_detectors.loc[output_detectors.detector_id.isin(detectors.detector_id.unique())]
            # filter out motorway lanes - optional
            if filter_motorway:
                motorway_detectors = detectors.loc[detectors.lane_id.isin(self.get_motorway(lanes))].detector_id.unique()
                output_detectors = output_detectors.loc[~output_detectors.detector_id.isin(motorway_detectors)]
            # filter out residential lanes - optional
            if filter_residential:
                residential_detectors = detectors.loc[detectors.lane_id.isin(self.get_residential(lanes))].detector_id.unique()
                output_detectors = output_detectors.loc[~output_detectors.detector_id.isin(residential_detectors)]
            # filter out unused detectors
            unused_detectors = self.get_unused_detct(output_detectors)
            output_detectors = output_detectors.loc[~output_detectors.detector_id.isin(unused_detectors)]

            standard_input = pd.merge(
                output_detectors, detectors, on='detector_id', how='left'
            )

        print(f'Final no. detectors: {standard_input.detector_id.nunique()}')
        self.save_pickle(
            standard_input[
                ['timestep', 'detector_id', 'geometry', 'lane_id', 'lane_length', 'pos',
                 'speed', 'occupancy', 'density', 'flow']
            ], files_names[0]
        )

        standard_input.rename(
            columns={'detector_id': 'detid', 'lane_length': 'length', 'occupancy': 'occ', 'timestep': 'interval'},
            inplace=True
        )
        standard_input['lat'] = standard_input['geometry'].apply(lambda point: point.x)
        standard_input['long'] = standard_input['geometry'].apply(lambda point: point.y)
        standard_input = standard_input[
            ['detid', 'flow', 'speed', 'occ', 'lat', 'long', 'density', 'interval', 'length', 'pos']]

        standard_input[['detid', 'lat', 'long']].drop_duplicates().to_csv(
            os.path.join(self.dir, files_names[1]), index=False
        )
        standard_input[['detid', 'flow', 'speed', 'occ', 'density', 'interval', 'length', 'pos']].to_csv(
            os.path.join(self.dir, files_names[2]), index=False
        )


class DataProcessor(InputData):
    def __init__(self, dir, standard_input):
        super().__init__(dir)
        self.standard_input = standard_input

    @staticmethod
    def scale_df(df):
        return (df - df.min().min()) / (df.max().max() - df.min().min())

    @staticmethod
    def scale_columns(df):
        return (df - df.min()) / (df.max() - df.min())

    @staticmethod
    def weighted_avg(x, weights):
        return np.average(x, weights=weights) if np.sum(weights) > 0 else 0

    def generate_ML_input(self, agg_interval=300, feature='density', traffic_feature_weight=1, geo_feature_weight=2):
        agg_funcs = {'density': 'mean',
                     'flow': 'mean',
                     'speed': lambda x: self.weighted_avg(x, self.standard_input.loc[x.index, 'sampled_seconds'])
                     }

        edges_average = (self.standard_input.groupby(['timestep', 'edge_id'])
                         .agg({feature: agg_funcs[feature]})
                         .reset_index()
                         .fillna(0)
                         )

        traffic_vectors = (edges_average.assign(interval=(edges_average['timestep'] // agg_interval) * agg_interval)
                           .groupby(['interval', 'edge_id'])[feature]
                           .mean()
                           .reset_index()
                           .pivot(index='edge_id', columns='interval', values=feature)
                           .fillna(0)
                           )

        scaled_traffic_vectors = self.scale_df(traffic_vectors) * traffic_feature_weight

        geo_vectors = (self.standard_input[['edge_id', 'geometry']].drop_duplicates(subset=['edge_id'])
                       .assign(
            x=lambda df: df['geometry'].apply(lambda line: line.interpolate(0.5, normalized=True).coords.xy[0][0]),
            y=lambda df: df['geometry'].apply(lambda line: line.interpolate(0.5, normalized=True).coords.xy[1][0]),
        )
                       .drop(columns='geometry')
                       .set_index('edge_id')
                       )

        scaled_geo_vectors = self.scale_columns(geo_vectors) * geo_feature_weight

        method_input = scaled_geo_vectors.join(scaled_traffic_vectors)

        self.save_pickle(method_input,
                         f'ml_input_{agg_interval}_{feature}_{traffic_feature_weight}_{geo_feature_weight}.pkl')

        return method_input

    def generate_snake_input(self, agg_interval=300, feature='density'):
        agg_funcs = {'density': 'mean',
                     'flow': 'mean',
                     'speed': lambda x: self.weighted_avg(x, self.standard_input.loc[x.index, 'sampled_seconds'])
                     }

        edges_average = (self.standard_input.groupby(['timestep', 'edge_id'])
                         .agg({feature: agg_funcs[feature]})
                         .reset_index()
                         .fillna(0)
                         )

        intervals_average = (edges_average.assign(interval=(edges_average['timestep'] // agg_interval) * agg_interval)
                             .groupby(['interval', 'edge_id'])[feature]
                             .mean()
                             .reset_index()
                             .fillna(0)
                             )

        method_input = (intervals_average.groupby('interval')
                        .apply(lambda df: df.set_index('edge_id')[feature].to_dict())
                        .to_dict()
                        )

        return method_input


if __name__ == '__main__':

    # define scenario
    scr = 'Zurich'                      # Ingolstadt                      # Zurich               # Synthetic
    try:
        bbox = gpd.read_file(r'scenarios\{}\bbox\bbox.shp'.format(scr)).to_crs(crs='EPSG:4326').geometry[0]
    except:
        bbox = None

    # load files
    edges = DataStandardiser.load_pickle(r'data\{}\network\edges.pkl'.format(scr))
    lanes = DataStandardiser.load_pickle(r'data\{}\network\lanes.pkl'.format(scr))
    detectors = DataStandardiser.load_pickle(r'data\{}\network\detectors.pkl'.format(scr))

    output_lanes = DataStandardiser.load_pickle(r'data\{}\output\output_lanes.pkl'.format(scr))
    try:
        output_detectors = DataStandardiser.load_pickle(r'data\{}\output\output_detectors.pkl'.format(scr))
    except:
        output_detectors = None

    # define directory
    input_dir = r'data\{}\input'.format(scr)
    if not os.path.isdir(input_dir):
        os.makedirs(input_dir)

    standardiser = DataStandardiser(input_dir)

    standardiser.standardise_input(edges,
                                   lanes,
                                   output_lanes=output_lanes,
                                   bbox=bbox,
                                   filter_motorway=False,
                                   filter_residential=False,
                                   file_name='standard_input.pkl')

    standardiser.standardise_detectors_input(detectors,
                                             edges,
                                             lanes,
                                             output_detectors=output_detectors,
                                             output_lanes=output_lanes,
                                             bbox=bbox,
                                             meso=True,
                                             filter_motorway=False,
                                             filter_residential=False,
                                             files_names=(
                                                 'detectors_standard_input.pkl',
                                                 'detectors_input.csv',
                                                 'detectors_loc.csv'
                                             )
                                             )

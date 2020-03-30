#!/usr/bin/env python3
import argparse
import geopandas as gpd
import pykitti

from shapely.geometry import LineString, Point


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--kitti_root',
        help='directory containing kitti dataset',
        default='data/kitti'
    )
    parser.add_argument(
        '--outdir',
        help='directory to output shapefile',
        default='data/shapes'
    )
    parser.add_argument(
        '--date',
        help='date to load (YYYY_MM_DD format)',
        required=True
    )
    parser.add_argument(
        '--drive',
        help='drive to load',
        type=int,
        required=True
    )
    return parser.parse_args()


def main():
    args = parse_args()

    dataset = pykitti.raw(
        args.kitti_root,
        args.date,
        f'{args.drive:04}'
    )

    geometry = []
    for i in range(1, len(dataset.oxts)):
        oxts_curr = dataset.oxts[i].packet
        oxts_prev = dataset.oxts[i - 1].packet
        geometry.append(
            LineString([
                (oxts_prev.lon, oxts_prev.lat),
                (oxts_curr.lon, oxts_curr.lat)
            ])
        )

    df = gpd.GeoDataFrame(geometry=geometry)
    df.to_file(
        driver='ESRI Shapefile',
        filename=args.outdir
    )


if __name__ == '__main__':
    main()

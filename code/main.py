from dotenv import load_dotenv
import os
import json

from PlanetDownloader import PlanetDownloader
from settings import daterange, cloud_cover_lte, item_types

load_dotenv()

def get_feature(type):
    return {
        "type":"Feature",
        "properties": {
            "year": 2016,
            "month": 1
        },
        "geometry": {
            "type": type,
            "coordinates": []
        }
    }

def calculate_center(coords):
    return [
        coords[0] + ((coords[2] - coords[0]) / 2),
        coords[1] + ((coords[3] - coords[1]) / 2),
    ]

def _create_geojson(features):
    geo_obj = {
        "type": "FeatureCollection",
        "features": features
    }
    return geo_obj

def convert_to_geojson(data):
    features = []
    for id, entry in enumerate(data):
        feature = get_feature('Polygon')
        feature["geometry"]["coordinates"] = \
            entry["coordinates"]
        feature["properties"]["year"] = int(entry["id"][0:4])
        feature["properties"]["month"] = int(entry["id"][4:6])
        features.append(feature)
    geo_obj = _create_geojson(features)
    return geo_obj


if __name__=="__main__":
    AOIs = {
        "Negribreen, Svalbard": [18.747975,78.549513,19.896046,78.677482],
        "Jakobshavns Isbrae": [-50.446716,69.070741,-49.453827,69.303492],
        "Sverdrup Glacier": [-86.454699,74.407512,-79.313586,76.273417],
        "Giesecke Braer": [-55.577168,73.482949,-55.088276,73.624472],
        "Helheim Glacier": [-38.427557,66.247253,-37.615942,66.520674],
    }

    pd = PlanetDownloader(
        os.getenv('API_KEY'),
        cloud_cover_lte,
        item_types
    )

    for loc, aoi in AOIs.items():
        data = []
        for year in daterange['years']:
            for month in daterange['months']:
                month = str(month).zfill(2)
                start_date_time = f"{year}-{month}-01T00:00:00Z"
                end_date_time = f"{year}-{month}-30T00:00:00Z"

                extent = aoi
                data.extend(pd.search_ids(
                    extent,
                    start_date_time,
                    end_date_time
                ))
        geo_obj = convert_to_geojson(data)

        with open(f'../data/{loc}.geojson', 'w') as outfile:
            json.dump(geo_obj, outfile)

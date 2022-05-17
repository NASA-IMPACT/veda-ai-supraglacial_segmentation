import json
import mercantile
import numpy as np
import requests

from ratelimit import limits, sleep_and_retry


BODY = {
    "filter": {
        "type": "AndFilter",
        "config": [
            {
                "type": "GeometryFilter",
                "field_name": "geometry",
                "config": {
                    "type": "Polygon",
                    "coordinates": []
                }
            },
            {
                "type": "DateRangeFilter",
                "field_name": "acquired",
                "config": {
                    "gte": "",
                    "lte": ""
                }
            },
            {
                "type": "RangeFilter",
                "field_name": "cloud_cover",
                "config": {
                    "lte": 0.5
                }
            }
        ]
    },
    "item_types": ["PSScene3Band"]
}

ZOOM_LEVEL = 14 # approx 2.4m per pixel

URL = 'https://api.planet.com/compute/ops/orders/v2'

SEARCH_URL = 'https://api.planet.com/data/v1/quick-search?_sort=acquired+desc&_page_size=250'


class PlanetDownloader:

    def __init__(self, api_key, cloud_cover_lte, item_types):
        """
            Initializer
        Args:
            api_key (str): api key for planet data access
        """
        self.api_key = api_key
        self.cloud_cover_lte = cloud_cover_lte
        self.item_types = item_types

    @sleep_and_retry
    @limits(calls=3, period=1)  # 4 requests per 1 sec
    def _get_next_orders(self, new_url):
        """
            Get next results from the next link in paginated results
        Args:
            new_url (str): Link to the next url

        Returns:
            dict: the json response from the api
        """
        parsed_content = { "features": [], "_links": {} }
        response = requests.get(new_url, auth=(self.api_key, ''))
        # response.raise_for_status()
        parsed_content = json.loads(response.text)
        return parsed_content

    @sleep_and_retry
    @limits(calls=3, period=1)  # 4 requests per 1 sec
    def search_ids(self, extent, start_date_time, end_date_time):
        """
            Search scene ids in planet
        Args:
            extent (list): list of coordinates
            start_date_time (str): start time in the format "yyyymmddThhddssZ"
            end_date_time (str): end time in the format "yyyymmddThhddssZ"
        Returns:
            list: list of ids and tile sizes
        """
        all = []
        body = BODY
        body['filter']['config'][0]['config']['coordinates'] = \
            self.prepare_coordinates(extent)
        body['filter']['config'][1]['config'] = {
            'gte': start_date_time,
            'lte': end_date_time
        }

        if self.cloud_cover_lte:
            body['filter']['config'][2]['config']['lte'] = self.cloud_cover_lte

        if self.item_types:
            body['item_types'] = self.item_types

        response = requests.post(
            SEARCH_URL, auth=(self.api_key, ''), json=body
        )
        # if not 200 raise error
        response.raise_for_status()
        parsed_content = json.loads(response.text)
        # print(parsed_content["_links"])
        all.extend(parsed_content['features'])

        while parsed_content["_links"].get("_next"):
            new_url = parsed_content["_links"].get("_next") + "&_sort=acquired+desc&_page_size=250"
            parsed_content = self._get_next_orders(new_url)            
            all.extend(parsed_content['features'])

        return self.extract_data(all)

    def extract_data(self, parsed_data):
        """
            Extract coordinates and tile information from the response
        Args:
            parsed_data (list): list of items returned by planet
        Returns:
            list: list of item ids and x, y tiles.
        """
        extracted_data = []
        for feature in parsed_data:
            current_data = {'images': []}
            current_data['id'] = feature['id']
            reverted_coordinates = self.revert_coordinates(
                feature['geometry']['coordinates']
            )
            current_data['coordinates'] = feature['geometry']['coordinates']
            feature['geometry']['coordinates']
            current_data['tiles'] = self.tile_indices(reverted_coordinates)
            extracted_data.append(current_data)
        return extracted_data

    def revert_coordinates(self, coordinates):
        """
            Revert the coordinates from an extended notation to flat coordinate
            notation
        Args:
            coordinates (list): list: [
                [left, down],
                [right, down],
                [right, up],
                [left, up],
                [left, down]
            ]
        Returns:
            list: [left, down, right, top]
        """
        coordinates = np.asarray(coordinates)
        lats = coordinates[:, :, 1]
        lons = coordinates[:, :, 0]
        return [lons.min(), lats.min(), lons.max(), lats.max()]

    def tile_indices(self, coordinates):
        """
            Extract tile indices based on coordinates
        Args:
            coordinates (list): [left, down, right, top]
        Returns:
            list: [[start_x, end_x], [start_y, end_y]]
        """
        start_x, start_y, _ = mercantile.tile(
            coordinates[0],
            coordinates[3],
            ZOOM_LEVEL
        )
        end_x, end_y, _ = mercantile.tile(
            coordinates[2],
            coordinates[1],
            ZOOM_LEVEL
        )
        return [[start_x, end_x], [start_y, end_y]]

    def prepare_coordinates(self, extent):
        """
            Revert the coordinates from flat notation to extended coordinate
            notation
        Args:
            extent (list): [left, down, right, up]
        Returns:
            list: [
                [left, down],
                [right, down],
                [right, up],
                [left, up],
                [left, down]
            ]
        """
        return [
            [
                [extent[0], extent[1]],
                [extent[2], extent[1]],
                [extent[2], extent[3]],
                [extent[0], extent[3]],
                [extent[0], extent[1]]
            ]
        ]

    

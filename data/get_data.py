"""
    data/get_data.py
    ----------------
    Script to abstract getting data from loader, this scrip import the loader
    and uses it to retrieve the data and also shuffle it without in a different
    function
"""
from __future__ import annotations

from data.loader import DataLoader, shuffle_and_split
from typing import List, Optional, Union

from logs.logger import Logger


files = [
    "uav_beijing/train.csv", "uav_boston/train.csv", "uav_london/train.csv",
    "uav_moscow/train.csv", "uav_tokyo/train.csv"
]


def get_city_data(cities: Union[str, List]="all", logger: Optional[Logger]=None):
    city_list = [city.strip().lower() for city in cities.split(",")]
    supported = {"beijing", "boston", "london", "moscow", "tokyo"}
    invalid = set(city_list) - supported
    
    if invalid: sys.exit(1)
    
    files = [f"uav_{city}/train.csv" for city in city_list]
    loader = DataLoader(logger=logger)
    
    return loader.load(files)


def get_shuffled_city_data(
    cities: Union[str, List]="all", logger: Optional[Logger]=None, 
    validation_ratio: float=0.10
):
    dtr, dts = shuffle_and_split(
        get_city_data(cities, logger), val_ratio=validation_ratio
    )
    return dtr, dts

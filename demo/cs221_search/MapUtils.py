import heapq
import json
from collections import defaultdict
from dataclasses import dataclass
from math import asin, cos, radians, sin, sqrt
from typing import Dict, Hashable, List, Optional, Set, Tuple

from SearchProblemUtils import (
    Heuristic,
    SearchProblem,
    State,
    SearchAlgorithm,
)

import osmium
from osmium import osm

########################################################################################
# Map Abstraction Overview & Useful Data Structures
#   > `GeoLocation` :: forms the atomic units of our abstraction; each `GeoLocation`
#                      object is uniquely specified as a pair of coordinates denoting
#                      latitude/longitude (in degrees).
#
#   > `CityMap` is the core structure defining the following:
#       + `locations` [str -> GeoLocation]: A dictionary mapping a unique label to a
#                                           specific GeoLocation.
#
#       + `tags` [str -> List[str]]: A dictionary mapping a location label (same keys
#                                    as above) to a list of meaningful "tags"
#                                    (e.g., amenity=park or landmark=hoover_tower).
#                                    These tags are parsed from OpenStreetMaps or
#                                    defined manually as "landmarks" in
#                                    `data/stanford-landmarks.json`.
#
#       + `distances` [str -> [str -> float]]: A nested dictionary mapping pairs of
#                                              locations to distances (e.g.,
#                                              `distances[label1][label2] = 21.3`).

# Constants
RADIUS_EARTH = 6371000  # Radius of earth in meters (~ equivalent to 3956 miles).
UNIT_DELTA = 0.00001    # Denotes the change in latitude/longitude (in degrees) that
                        # equates to distance of ~1m.


@dataclass(frozen=True)
class GeoLocation:
    """A latitude/longitude of a physical location on Earth."""
    latitude: float
    longitude: float

    def __repr__(self):
        return f"{self.latitude},{self.longitude}"


class CityMap:
    """
    A city map consists of a set of *labeled* locations with associated tags, and
    connections between them.
    """
    def __init__(self) -> None:
        # Location label -> actual geolocation (latitude/longitude)
        self.geoLocations: Dict[str, GeoLocation] = {}

        # Location label -> list of tags (e.g., amenity=park)
        self.tags: Dict[str, List[str]] = defaultdict(list)

        # Location label -> adjacent location label -> distance between the two
        self.distances: Dict[str, Dict[str, float]] = defaultdict(dict)

    def addLocation(self, label: str, location: GeoLocation, tags: List[str]) -> None:
        """Add a location (denoted by `label`) to map with the provided set of tags."""
        assert label not in self.geoLocations, f"Location {label} already processed!"
        self.geoLocations[label] = location
        self.tags[label] = [makeTag("label", label)] + tags

    def addConnection(
        self, source: str, target: str, distance: Optional[float] = None
    ) -> None:
        """Adds a connection between source <--> target to `self.distances`."""
        if distance is None:
            distance = computeDistance(
                self.geoLocations[source], self.geoLocations[target]
            )
        self.distances[source][target] = distance
        self.distances[target][source] = distance


def addLandmarks(
    cityMap: CityMap, landmarkPath: str, toleranceMeters: float = 250.0
) -> None:
    """
    Add landmarks from `path` to `cityMap`. A landmark (e.g., Gates Building) is
    associated with a `GeoLocation`.

    Landmarks are explicitly defined via the `landmarkPath` file, which borrows
    latitude/longitude for various spots on Stanford Campus from Google Maps; these
    may not *exactly* line up with existing locations in the CityMap, so instead we map
    a given landmark onto the closest existing location (subject to a max tolerance).
    """
    with open(landmarkPath) as f:
        landmarks = json.load(f)

    for item in landmarks:
        latitudeString, longitudeString = item["geo"].split(",")
        geo = GeoLocation(float(latitudeString), float(longitudeString))

        bestDistance, bestLabel = min(
            (computeDistance(geo, existingGeo), existingLabel)
            for existingLabel, existingGeo in cityMap.geoLocations.items()
        )

        if bestDistance < toleranceMeters:
            for key in ["landmark", "amenity"]:
                if key in item:
                    cityMap.tags[bestLabel].append(makeTag(key, item[key]))


########################################################################################
# Utility Functions


def makeTag(key: str, value: str) -> str:
    """Locations have string-valued tags which are created from (key, value) pairs."""
    return f"{key}={value}"


def locationFromTag(tag: str, cityMap: CityMap) -> Optional[str]:
    possibleLocations = sorted(
        [location for location, tags in cityMap.tags.items() if tag in tags]
    )
    return possibleLocations[0] if len(possibleLocations) > 0 else None


def computeDistance(geo1: GeoLocation, geo2: GeoLocation) -> float:
    """
    Compute the distance (straight line) between two geolocations using the
    Haversine formula. Returns distance in meters.
    """
    lon1, lat1 = radians(geo1.longitude), radians(geo1.latitude)
    lon2, lat2 = radians(geo2.longitude), radians(geo2.latitude)

    deltaLon, deltaLat = lon2 - lon1, lat2 - lat1
    haversine = (sin(deltaLat / 2) ** 2) + (cos(lat1) * cos(lat2)) * (
        sin(deltaLon / 2) ** 2
    )

    return 2 * RADIUS_EARTH * asin(sqrt(haversine))


def checkValid(
    path: List[str],
    cityMap: CityMap,
    startLocation: str,
    endTag: str,
    waypointTags: List[str],
) -> bool:
    """Check if a given solution/path is valid subject to the given CityMap instance."""
    if path[0] != startLocation:
        print(f"Invalid path: does not start with {startLocation}")
        return False

    if endTag not in cityMap.tags[path[-1]]:
        print("Invalid path: final location does not contain {endTag}")
        return False

    for i in range(len(path) - 1):
        if path[i + 1] not in cityMap.distances[path[i]]:
            print(f"Invalid path: {path[i]} is not connected to {path[i + 1]}")
            return False

    doneTags = set(tag for location in path for tag in cityMap.tags[location])
    diffTags = set(waypointTags).difference(doneTags)
    if len(diffTags) > 0:
        print(f"Invalid path: does not contain waypoints {diffTags}")
        return False

    return True


def getTotalCost(path: List[str], cityMap: CityMap) -> float:
    """Return the total distance of the given path (assuming it's valid)."""
    cost = 0.0
    for i in range(len(path) - 1):
        cost += cityMap.distances[path[i]][path[i + 1]]
    return cost

def extractPath(startLocation: str, search: SearchAlgorithm) -> List[str]:
    """
    Assumes that `solve()` has already been called on the `searchAlgorithm`.

    We extract a sequence of locations from `search.path` (see MapUtils.py to better
    understand exactly how this list gets populated).
    """
    return [startLocation] + search.actions

def printPath(
    path: List[str],
    waypointTags: List[str],
    cityMap: CityMap,
    outPath: Optional[str] = "path.json",
):
    doneWaypointTags = set()
    for location in path:
        for tag in cityMap.tags[location]:
            if tag in waypointTags:
                doneWaypointTags.add(tag)
        tagsStr = " ".join(cityMap.tags[location])
        doneTagsStr = " ".join(sorted(doneWaypointTags))
        print(f"Location {location} tags:[{tagsStr}]; done:[{doneTagsStr}]")
    print(f"Total distance: {getTotalCost(path, cityMap)}")

    # (Optional) Write path to file, for use with `visualize.py`
    if outPath is not None:
        with open(outPath, "w") as f:
            data = {"waypointTags": waypointTags, "path": path}
            json.dump(data, f, indent=2)


########################################################################################
# Data Processing Functions


def makeGridLabel(x: int, y: int) -> str:
    """Function to create location label from grid coordinates."""
    return f"{x},{y}"


def createGridMap(width: int, height: int) -> CityMap:
    """Create a simple width x height grid map of locations."""
    cityMap = CityMap()

    for x, lat in enumerate([x * UNIT_DELTA for x in range(width)]):
        for y, lon in enumerate([y * UNIT_DELTA for y in range(height)]):
            cityMap.addLocation(
                makeGridLabel(x, y),
                GeoLocation(lat, lon),
                tags=[makeTag("x", x), makeTag("y", y)],
            )
            if x > 0:
                cityMap.addConnection(
                    makeGridLabel(x - 1, y), makeGridLabel(x, y), distance=1
                )
            if y > 0:
                cityMap.addConnection(
                    makeGridLabel(x, y - 1), makeGridLabel(x, y), distance=1
                )

    return cityMap


def createGridMapWithCustomTags(width: int, height: int, tags: Dict[Tuple[int, int], List[str]]) -> CityMap:
    """Create a width x height grid map with per-cell custom tags."""
    cityMap = CityMap()

    for x, lat in enumerate([x * UNIT_DELTA for x in range(width)]):
        for y, lon in enumerate([y * UNIT_DELTA for y in range(height)]):
            cityMap.addLocation(
                makeGridLabel(x, y),
                GeoLocation(lat, lon),
                tags=[makeTag("x", x), makeTag("y", y)] + tags[(x, y)],
            )
            if x > 0:
                cityMap.addConnection(
                    makeGridLabel(x - 1, y), makeGridLabel(x, y), distance=1
                )
            if y > 0:
                cityMap.addConnection(
                    makeGridLabel(x, y - 1), makeGridLabel(x, y), distance=1
                )

    return cityMap


def readMap(osmPath: str) -> CityMap:
    """
    Create a CityMap given a path to an OSM `.pbf` file.
    """
    class MapCreationHandler(osmium.SimpleHandler):
        def __init__(self) -> None:
            super().__init__()
            self.nodes: Dict[str, GeoLocation] = {}
            self.tags: Dict[str, List[str]] = defaultdict(list)
            self.edges: Set[Tuple[str, str]] = set()

        def node(self, n: osm.Node) -> None:
            self.tags[str(n.id)] = [makeTag(tag.k, tag.v) for tag in n.tags]

        def way(self, w: osm.Way) -> None:
            pathType = w.tags.get("highway", None)
            if pathType is None or pathType in {
                "motorway", "motorway_link", "trunk", "trunk_link",
            }:
                return
            elif (
                w.tags.get("pedestrian", "n/a") == "no"
                or w.tags.get("foot", "n/a") == "no"
            ):
                return

            wayNodes = w.nodes
            for sourceIdx in range(len(wayNodes) - 1):
                s, t = wayNodes[sourceIdx], wayNodes[sourceIdx + 1]
                sLabel, tLabel = str(s.ref), str(t.ref)
                sLoc = GeoLocation(s.location.lat, s.location.lon)
                tLoc = GeoLocation(t.location.lat, t.location.lon)

                assert sLoc != tLoc, "Source and Target are the same location!"

                self.nodes[sLabel], self.nodes[tLabel] = sLoc, tLoc
                self.edges.add((sLabel, tLabel))

    mapCreator = MapCreationHandler()
    mapCreator.apply_file(osmPath, locations=True)

    cityMap = CityMap()
    for nodeLabel in mapCreator.nodes:
        cityMap.addLocation(
            nodeLabel, mapCreator.nodes[nodeLabel], tags=mapCreator.tags[nodeLabel]
        )

    for src, tgt in mapCreator.edges:
        cityMap.addConnection(src, tgt)

    return cityMap


def createStanfordMap() -> CityMap:
    cityMap = readMap("data/stanford.pbf")
    addLandmarks(cityMap, "data/stanford-landmarks.json")
    return cityMap


def createCustomMap(map_file: str, landmarks_file: str) -> CityMap:
    """
    Create a CityMap from an OSM `.pbf` file and a landmarks `.json` file.
    """
    cityMap = readMap(map_file)
    addLandmarks(cityMap, landmarks_file)
    return cityMap

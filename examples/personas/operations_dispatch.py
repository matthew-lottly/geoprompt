from __future__ import annotations

from geoprompt.geocoding import get_travel_mode, route_directions_narrative, stop_sequence_optimize


if __name__ == "__main__":
    stops = [(0.0, 0.0), (1.0, 0.5), (2.0, 1.0), (1.5, 2.5)]
    mode = get_travel_mode("emergency")
    order = stop_sequence_optimize(stops)
    ordered = [stops[i] for i in order]
    directions = route_directions_narrative(ordered)
    print("Travel mode:", mode)
    print("Stop order:", order)
    print("Directions:")
    for step in directions:
        print("-", step["instruction"], step["distance"], step["distance_unit"])

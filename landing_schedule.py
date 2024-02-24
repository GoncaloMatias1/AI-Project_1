import heapq
from airplane import Airplane

def generate_airplane_stream(num_airplanes):
    return [Airplane(airplane_id=i) for i in range(1, num_airplanes + 1)]

def schedule_landings(airplane_stream, num_runways=3):
    airplane_stream.sort(key=lambda x: (-x.emergency, x.priority), reverse=True)
    runway_availability = [0] * num_runways
    landing_schedule = []

    for airplane in airplane_stream:
        earliest_runway_index = min(range(num_runways), key=lambda i: runway_availability[i])
        landing_time = max(airplane.expected_arrival_time, runway_availability[earliest_runway_index])
        landing_schedule.append((airplane.airplane_id, landing_time))
        runway_availability[earliest_runway_index] = landing_time + 3

    landing_schedule.sort(key=lambda x: x[1])
    return landing_schedule

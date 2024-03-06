import random
import pandas as pd

class Airplane:
    def __init__(self, id, min_fuel, max_fuel, min_arrival_time, max_arrival_time):
        self.id = id
        self.fuel_consumption_rate = random.uniform(5, 20)
        self.expected_landing_time = random.uniform(min_arrival_time, max_arrival_time)
        self.fuel_level = random.uniform(min_fuel, max_fuel)
        
        self.emergency_fuel = (self.fuel_consumption_rate*60)
        self.fuel_level_final = self.fuel_level - (self.emergency_fuel)
        self.remaining_flying_time = self.fuel_level_final / self.fuel_consumption_rate
        self.urgency = self.remaining_flying_time - 60

def generate_airplane_stream(num_airplanes, min_fuel, max_fuel, min_arrival_time, max_arrival_time):
    return [Airplane(i, min_fuel, max_fuel, min_arrival_time, max_arrival_time) for i in range(1, num_airplanes + 1)]

def schedule_landings(airplane_stream):
    sorted_airplanes = sorted(airplane_stream, key=lambda x: (x.urgency >= 0, x.expected_landing_time))
    landing_schedule = []
    landing_strip_availability = [0, 0, 0]
    for airplane in sorted_airplanes:
        chosen_strip, next_available_time = min(enumerate(landing_strip_availability), key=lambda x: x[1])
        if airplane.urgency < 0 or airplane.expected_landing_time <= next_available_time or airplane.remaining_flying_time <= next_available_time or airplane.remaining_flying_time <= airplane.expected_landing_time:
            actual_landing_time = max(next_available_time, airplane.expected_landing_time)
        else:
            actual_landing_time = airplane.expected_landing_time
        actual_landing_time = min(actual_landing_time, airplane.remaining_flying_time)
        landing_strip_availability[chosen_strip] = actual_landing_time + 3
        landing_schedule.append((airplane.id, actual_landing_time, airplane.urgency < 0, chosen_strip + 1))
    return pd.DataFrame(landing_schedule, columns=["Airplane", "Landing Time", "Urgent", "Landing Strip"])

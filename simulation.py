import random
import pandas as pd


class Airplane:
    def __init__(self, id, min_fuel, max_fuel, min_arrival_time, max_arrival_time):
        self.id = id
        self.fuel_consumption_rate = random.uniform(5, 20)
        self.expected_landing_time = random.uniform(min_arrival_time, max_arrival_time)
        self.fuel_level = random.uniform(min_fuel, max_fuel)

        self.emergency_fuel = (self.fuel_consumption_rate * 60)
        self.fuel_level = max(self.fuel_level, self.emergency_fuel)
        self.fuel_level_final = self.fuel_level - self.emergency_fuel
        self.remaining_flying_time = self.fuel_level_final / self.fuel_consumption_rate
        self.urgency = self.remaining_flying_time < 1  # urgency is true if less than one hour of fuel remains

def generate_airplane_stream(num_airplanes, min_fuel, max_fuel, min_arrival_time, max_arrival_time):
    return [Airplane(i, min_fuel, max_fuel, min_arrival_time, max_arrival_time) for i in range(1, num_airplanes + 1)]

def schedule_landings(airplane_stream):
    # Airplanes sorted by their expected landing time and remaining flying time
    sorted_airplanes = sorted(airplane_stream, key=lambda x: x.expected_landing_time)
    landing_schedule = []
    landing_strip_availability = [0, 0, 0]  # Initial availability of each of the 3 landing strips

    for airplane in sorted_airplanes:
        chosen_strip, next_available_time = min(enumerate(landing_strip_availability), key=lambda x: x[1])
        # Determine if this landing is urgent based on emergency fuel
        is_urgent = airplane.fuel_level_final < airplane.emergency_fuel

        # The actual landing time should be the later of expected landing time or next available time
        actual_landing_time = max(airplane.expected_landing_time, next_available_time)

        # If the landing is urgent, the plane needs to land immediately, if possible
        if is_urgent and actual_landing_time > airplane.remaining_flying_time:
            actual_landing_time = airplane.remaining_flying_time  # Force immediate landing

        # Update the landing strip availability, adding 3 minutes for next landing
        landing_strip_availability[chosen_strip] = actual_landing_time + 3 / 60  # Convert 3 minutes to hours for availability

        # Append to landing schedule
        landing_schedule.append((airplane.id, actual_landing_time, is_urgent, chosen_strip + 1))

    # Convert the landing schedule to a DataFrame
    return pd.DataFrame(landing_schedule, columns=["Airplane", "Landing Time", "Urgent", "Landing Strip"])

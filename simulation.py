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
        self.urgency = self.remaining_flying_time < 1

def generate_airplane_stream(num_airplanes, min_fuel, max_fuel, min_arrival_time, max_arrival_time):
    return [Airplane(i, min_fuel, max_fuel, min_arrival_time, max_arrival_time) for i in range(1, num_airplanes + 1)]

def schedule_landings(airplane_stream):
    sorted_airplanes = sorted(airplane_stream, key=lambda x: x.expected_landing_time)
    landing_schedule = []
    landing_strip_availability = [0, 0, 0]  

    for airplane in sorted_airplanes:
        #neste caso se o combustivel restante apenas der para 60 minutos de voo, o avião irá aterrar com menos de 60 minutos de combustivel (não há nada para contrariar este problema)
        if airplane.fuel_level_final == 0:
            airplane.fuel_level_final = airplane.fuel_level


        chosen_strip, next_available_time_with_gap = min(
            [(index, time + 3/60) for index, time in enumerate(landing_strip_availability)], key=lambda x: x[1])
        is_urgent = airplane.fuel_level_final < airplane.emergency_fuel or airplane.remaining_flying_time < airplane.expected_landing_time
        actual_landing_time = max(airplane.expected_landing_time, next_available_time_with_gap)

        if is_urgent and actual_landing_time > airplane.remaining_flying_time:
            actual_landing_time = airplane.remaining_flying_time 

        landing_strip_availability[chosen_strip] = actual_landing_time + 3

        landing_schedule.append((airplane.id, actual_landing_time, is_urgent, chosen_strip + 1))

    return pd.DataFrame(landing_schedule, columns=["Airplane", "Landing Time", "Urgent", "Landing Strip"])

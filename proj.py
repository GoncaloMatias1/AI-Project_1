import random
import pandas as pd
import heapq
from heapq import heappush, heappop

class Airplane:
    def __init__(self, id):
        self.id = id
        self.arriving_fuel_level = random.uniform(1000, 5000)
        self.fuel_consumption_rate = random.uniform(5, 20)
        self.expected_landing_time = random.uniform(10, 120)
        self.priority = 0

    def calculate_priority(self):
        remaining_fuel_at_expected_landing = self.arriving_fuel_level - (self.fuel_consumption_rate * self.expected_landing_time)
        safety_threshold = self.fuel_consumption_rate * 60
        self.priority = safety_threshold - remaining_fuel_at_expected_landing

def generate_airplane_stream(num_airplanes):
    return [Airplane(id=i) for i in range(1, num_airplanes + 1)]

def schedule_landings(airplane_stream):
    for airplane in airplane_stream:
        airplane.calculate_priority()

    priority_queue = [(plane.priority, plane.expected_landing_time, plane) for plane in airplane_stream]
    heapq.heapify(priority_queue)

    landing_schedule = []
    current_time = 0
    runway_availability = [0] * 3

    while priority_queue:
        priority, expected_landing, airplane = heappop(priority_queue)
        runway_index = runway_availability.index(min(runway_availability))
        landing_time = max(current_time, expected_landing, runway_availability[runway_index])
        safety_threshold = airplane.fuel_consumption_rate * 60
        if airplane.arriving_fuel_level - (airplane.fuel_consumption_rate * landing_time) < safety_threshold:
            print(f"Emergencia: Aviao ID {airplane.id} nao tem combustivel suficiente para esperar. Temos de priorizar pouso imediato.")
            landing_time = max(current_time, runway_availability[runway_index])

        landing_schedule.append((airplane.id, landing_time))
        runway_availability[runway_index] = landing_time + 3
        current_time = landing_time

    landing_schedule.sort(key=lambda x: x[1])
    return landing_schedule


airplane_stream = generate_airplane_stream(50)
landing_schedule = schedule_landings(airplane_stream)

df_schedule = pd.DataFrame(landing_schedule, columns=["Airplane ID", "Landing Time (min)"])
print(df_schedule)

df_airplanes = pd.DataFrame([(plane.id, plane.arriving_fuel_level, plane.fuel_consumption_rate, plane.expected_landing_time)
                             for plane in airplane_stream],
                            columns=["Airplane ID", "Fuel", "Consumption Rate", "Expected Landing Time"])
print(df_airplanes)

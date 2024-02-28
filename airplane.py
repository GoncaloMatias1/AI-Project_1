import random
from utils import log_airplane_data
class Airplane:
    def __init__(self, airplane_id):
        self.airplane_id = airplane_id
        self.fuel_level = random.uniform(1000, 5000)
        self.fuel_consumption_rate = random.uniform(5, 20)
        self.expected_landing_time = random.uniform(10, 120)
        self.calculate_priority()

    def calculate_priority(self):
        remaining_fuel_at_arrival = self.fuel_level - (self.fuel_consumption_rate * (self.expected_landing_time / 60))
        safety_threshold = self.fuel_consumption_rate * 60  # Combustível para uma hora

        self.priority = max(safety_threshold - remaining_fuel_at_arrival, 0)

        self.emergency = remaining_fuel_at_arrival < safety_threshold

import random

class Airplane:
    def __init__(self, airplane_id):
        self.airplane_id = airplane_id
        self.fuel_level = random.uniform(1000, 5000)
        self.fuel_consumption_rate = random.uniform(5, 20)
        self.expected_arrival_time = random.uniform(10, 120)
        self.calculate_priority()

    def calculate_priority(self):
        remaining_fuel_at_arrival = self.fuel_level - (self.fuel_consumption_rate * (self.expected_arrival_time / 60))
        safety_threshold = self.fuel_consumption_rate * 60  # Combustível para uma hora
        self.priority = safety_threshold - remaining_fuel_at_arrival

        self.emergency = remaining_fuel_at_arrival < safety_threshold




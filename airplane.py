import random

class Airplane:
    def __init__(self, airplane_id):
        self.airplane_id = airplane_id
        self.fuel_level = random.uniform(1000, 5000)
        self.fuel_consumption_rate = random.uniform(5, 20)
        self.expected_arrival_time = random.uniform(10, 120)
        self.calculate_priority()
        self.emergency = self.priority < 0

    def calculate_priority(self):
        safety_threshold = self.fuel_consumption_rate * 60  # uma hora de voo
        remaining_fuel_at_arrival = self.fuel_level - (self.fuel_consumption_rate * (self.expected_arrival_time / 60))
        self.priority = safety_threshold - remaining_fuel_at_arrival

        # Quanto menor o combustível restante na chegada, maior a prioridade
        # Invertendo a fórmula para dar alta prioridade a valores menores
        self.priority = max(0, 1 - (remaining_fuel_at_arrival / safety_threshold))

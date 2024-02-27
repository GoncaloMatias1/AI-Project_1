import random
from utils import log_airplane_data
class Airplane:
    def __init__(self, airplane_id):
        self.airplane_id = airplane_id
        # Gerar os valores de forma aleatória dentro dos limites especificados
        self.fuel_level = random.uniform(1000, 5000)
        self.fuel_consumption_rate = random.uniform(5, 20)
        self.expected_arrival_time = random.uniform(10, 120)
        self.calculate_priority()

    def calculate_priority(self):
        # Calcula o combustível restante na chegada esperada
        fuel_used = self.fuel_consumption_rate * (self.expected_arrival_time / 60)
        remaining_fuel_at_arrival = self.fuel_level - fuel_used
        safety_threshold = self.fuel_consumption_rate * 60  # Combustível para uma hora

        # A prioridade agora reflete o quanto o combustível restante é menor do que o limiar de segurança
        self.priority = max(safety_threshold - remaining_fuel_at_arrival, 0)

        # Atribui um status de emergência se o combustível restante na chegada esperada for menor que o limiar de segurança
        self.emergency = remaining_fuel_at_arrival < safety_threshold

        log_airplane_data(self)

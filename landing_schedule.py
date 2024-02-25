import random
import math
from airplane import Airplane

def generate_airplane_stream(num_airplanes):
    return [Airplane(airplane_id=i) for i in range(1, num_airplanes + 1)]


def calculate_fitness(airplane_stream, landing_schedule):
    fitness = 0
    for airplane_id, landing_time in landing_schedule:
        airplane = next(plane for plane in airplane_stream if plane.airplane_id == airplane_id)
        time_waiting = max(0, landing_time - airplane.expected_arrival_time) / 60  # Convertendo para horas
        fuel_remaining = airplane.fuel_level - (airplane.fuel_consumption_rate * time_waiting)

        # Garantir que o combustível restante não seja negativo e penalizar o fitness em caso de combustível insuficiente
        if fuel_remaining < 0:
            fitness -= abs(fuel_remaining) * 1000  # Penalidade significativa para combustível negativo
        else:
            fitness += fuel_remaining  # Aumenta o fitness para combustível positivo

    return fitness

def hill_climbing(airplane_stream, num_runways=3, iterations=1000):
    # Gere um cronograma inicial
    current_schedule = schedule_landings(airplane_stream, num_runways)
    current_fitness = calculate_fitness(airplane_stream, current_schedule)

    for _ in range(iterations):
        # Gere um 'vizinho' trocando dois tempos de pouso aleatoriamente
        neighbor_schedule = list(current_schedule)
        i, j = random.sample(range(len(neighbor_schedule)), 2)
        neighbor_schedule[i], neighbor_schedule[j] = neighbor_schedule[j], neighbor_schedule[i]

        # Avalie o novo cronograma
        neighbor_fitness = calculate_fitness(airplane_stream, neighbor_schedule)

        # Se o 'vizinho' for melhor, mova para ele
        if neighbor_fitness > current_fitness:
            current_schedule, current_fitness = neighbor_schedule, neighbor_fitness

    unable_to_land = [airplane_id for airplane_id, landing_time in current_schedule if (get_fuel_remaining(airplane_stream, airplane_id, landing_time) < 0)]
    return current_schedule, unable_to_land

def schedule_landings(airplane_stream, num_runways=3):
    # Inicializa o cronograma com tempos de pouso baseados no tempo esperado de chegada
    landing_schedule = []
    runway_availability = [0] * num_runways

    # Ordena os aviões pela prioridade
    for airplane in sorted(airplane_stream, key=lambda x: x.priority):
        earliest_runway_index = min(range(num_runways), key=lambda i: runway_availability[i])
        proposed_landing_time = max(airplane.expected_arrival_time, runway_availability[earliest_runway_index])
        time_waiting = (proposed_landing_time - airplane.expected_arrival_time) / 60  # Convertido para horas
        fuel_needed_to_wait = airplane.fuel_consumption_rate * time_waiting

        # Verifica se o avião tem combustível suficiente para esperar
        if airplane.fuel_level >= fuel_needed_to_wait:
            # Agendar pouso se o combustível for suficiente
            landing_schedule.append((airplane.airplane_id, proposed_landing_time))
            runway_availability[earliest_runway_index] = proposed_landing_time + 3  # Assumindo que cada pouso ocupa a pista por 3 minutos
        else:
            # Imprime um aviso se o avião não tem combustível suficiente para esperar
            print(f"Airplane {airplane.airplane_id} cannot land at {proposed_landing_time} minutes due to insufficient fuel.")

    landing_schedule.sort(key=lambda x: x[1])
    return landing_schedule

def simulated_annealing(airplane_stream, num_runways=3, initial_temp=10000, cooling_rate=0.003):
    current_schedule = schedule_landings(airplane_stream, num_runways)
    current_fitness = calculate_fitness(airplane_stream, current_schedule)
    best_schedule = current_schedule
    best_fitness = current_fitness
    temp = initial_temp

    while temp > 1:
        # Gere um novo cronograma de pouso trocando dois aviões de lugar
        new_schedule = list(current_schedule)
        i, j = random.sample(range(len(new_schedule)), 2)
        new_schedule[i], new_schedule[j] = new_schedule[j], new_schedule[i]

        # Calcule o "fitness" do novo cronograma
        new_fitness = calculate_fitness(airplane_stream, new_schedule)

        # Calcule a diferença de "fitness" entre o novo e o atual cronograma
        fitness_diff = new_fitness - current_fitness

        # Se o novo cronograma for melhor ou se um número aleatório for menor que e^(fitness_diff/temp),
        # aceite o novo cronograma
        if fitness_diff > 0 or random.random() < math.exp(fitness_diff / temp):
            current_schedule = new_schedule
            current_fitness = new_fitness

            # Se o novo cronograma tiver o melhor "fitness" até agora, salve-o
            if new_fitness > best_fitness:
                best_schedule = new_schedule
                best_fitness = new_fitness

        # Diminua a temperatura
        temp *= 1 - cooling_rate

    unable_to_land = [airplane_id for airplane_id, landing_time in best_schedule if (get_fuel_remaining(airplane_stream, airplane_id, landing_time) < 0)]
    return best_schedule, unable_to_land

def get_fuel_remaining(airplane_stream, airplane_id, landing_time):
    airplane = next((plane for plane in airplane_stream if plane.airplane_id == airplane_id), None)
    if airplane is None:
        raise ValueError(f"No airplane found with ID: {airplane_id}")

    wait_time = max(0, landing_time - airplane.expected_arrival_time)
    fuel_consumed = airplane.fuel_consumption_rate * wait_time
    fuel_remaining = airplane.fuel_level - fuel_consumed
    return fuel_remaining

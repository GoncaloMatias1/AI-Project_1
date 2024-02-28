import random
import math
from airplane import Airplane

def generate_airplane_stream(num_airplanes):
    return [Airplane(airplane_id=i) for i in range(1, num_airplanes + 1)]


def calculate_fitness(airplane_stream, landing_schedule):
    fitness = 0
    for airplane_id, landing_time in landing_schedule:
        airplane = next(plane for plane in airplane_stream if plane.airplane_id == airplane_id)
        time_waiting = max(0, landing_time - airplane.expected_landing_time) / 60  # Convertendo para horas
        fuel_remaining = airplane.fuel_level - (airplane.fuel_consumption_rate * time_waiting)

        if fuel_remaining < 0:
            fitness -= abs(fuel_remaining) * 1000  
        else:
            fitness += fuel_remaining 

    return fitness

def hill_climbing(airplane_stream, num_runways=3, iterations=1000):
    current_schedule = schedule_landings(airplane_stream, num_runways)
    current_fitness = calculate_fitness(airplane_stream, current_schedule)

    for _ in range(iterations):
        neighbor_schedule = list(current_schedule)
        i, j = random.sample(range(len(neighbor_schedule)), 2)
        neighbor_schedule[i], neighbor_schedule[j] = neighbor_schedule[j], neighbor_schedule[i]
        neighbor_fitness = calculate_fitness(airplane_stream, neighbor_schedule)
        if neighbor_fitness > current_fitness:
            current_schedule, current_fitness = neighbor_schedule, neighbor_fitness

    unable_to_land = [airplane_id for airplane_id, landing_time in current_schedule if (get_fuel_remaining(airplane_stream, airplane_id, landing_time) < 0)]
    return current_schedule, unable_to_land

def schedule_landings(airplane_stream, num_runways=3):
    # Inicializa o cronograma de pouso e a disponibilidade das pistas
    landing_schedule = []
    runway_availability = [0] * num_runways

    # Ordena os aviões por prioridade, considerando primeiro emergências e depois o combustível restante
    sorted_planes = sorted(airplane_stream, key=lambda x: (-x.priority, x.expected_landing_time))

    for airplane in sorted_planes:
        # Encontra a primeira pista disponível e o tempo mais cedo que o avião pode pousar
        earliest_runway_index = min(range(num_runways), key=lambda i: runway_availability[i])
        proposed_landing_time = max(airplane.expected_landing_time, runway_availability[earliest_runway_index])

        # Calcula o combustível restante no tempo proposto de pouso
        fuel_needed_until_proposed_landing = airplane.fuel_consumption_rate * ((proposed_landing_time - airplane.expected_landing_time) / 60)
        fuel_remaining_at_proposed_landing = airplane.fuel_level - fuel_needed_until_proposed_landing

        # Verifica se o avião pode pousar com segurança no tempo proposto
        if fuel_remaining_at_proposed_landing >= airplane.fuel_consumption_rate:  # Garante pelo menos 1 hora de combustível restante
            landing_schedule.append((airplane.airplane_id, proposed_landing_time))
            runway_availability[earliest_runway_index] = proposed_landing_time + 3  # Considera a pista ocupada por 3 minutos após o pouso
        else:
            # Procura um tempo de pouso anterior que permita pouso seguro, se possível
            for earlier_time in range(int(proposed_landing_time - 1), int(airplane.expected_landing_time), -1):
                fuel_needed_until_earlier_landing = airplane.fuel_consumption_rate * ((earlier_time - airplane.expected_landing_time) / 60)
                fuel_remaining_at_earlier_landing = airplane.fuel_level - fuel_needed_until_earlier_landing
                if fuel_remaining_at_earlier_landing >= airplane.fuel_consumption_rate:
                    landing_schedule.append((airplane.airplane_id, earlier_time))
                    runway_availability[earliest_runway_index] = earlier_time + 3
                    break
            else:
                # Caso não encontre um horário seguro para pouso, agendará no tempo proposto originalmente, priorizando a segurança
                landing_schedule.append((airplane.airplane_id, proposed_landing_time))
                runway_availability[earliest_runway_index] = proposed_landing_time + 3

    # Ordena o cronograma de pouso pelo tempo de pouso
    landing_schedule.sort(key=lambda x: x[1])
    return landing_schedule


def simulated_annealing(airplane_stream, num_runways=3, initial_temp=10000, cooling_rate=0.003):
    current_schedule = schedule_landings(airplane_stream, num_runways)
    current_fitness = calculate_fitness(airplane_stream, current_schedule)
    best_schedule = current_schedule
    best_fitness = current_fitness
    temp = initial_temp

    while temp > 1:
        new_schedule = list(current_schedule)
        i, j = random.sample(range(len(new_schedule)), 2)
        new_schedule[i], new_schedule[j] = new_schedule[j], new_schedule[i]

        new_fitness = calculate_fitness(airplane_stream, new_schedule)

        fitness_diff = new_fitness - current_fitness

        if fitness_diff > 0 or random.random() < math.exp(fitness_diff / temp):
            current_schedule = new_schedule
            current_fitness = new_fitness

            if new_fitness > best_fitness:
                best_schedule = new_schedule
                best_fitness = new_fitness

        temp *= 1 - cooling_rate

    unable_to_land = [airplane_id for airplane_id, landing_time in best_schedule if (get_fuel_remaining(airplane_stream, airplane_id, landing_time) < 0)]
    return best_schedule, unable_to_land

def get_fuel_remaining(airplane_stream, airplane_id, landing_time):
    airplane = next((plane for plane in airplane_stream if plane.airplane_id == airplane_id), None)
    if airplane is None:
        raise ValueError(f"No airplane found with ID: {airplane_id}")

    wait_time = max(0, landing_time - airplane.expected_landing_time)
    fuel_consumed = airplane.fuel_consumption_rate * wait_time
    fuel_remaining = airplane.fuel_level - fuel_consumed
    return fuel_remaining

def tabu_search(airplane_stream, num_runways=3, iterations=1000, tabu_size=100):
    # Gera um schedule inicial
    current_schedule = schedule_landings(airplane_stream, num_runways)
    current_fitness = calculate_fitness(airplane_stream, current_schedule)

    best_schedule = current_schedule
    best_fitness = current_fitness

    tabu_list = [current_schedule]

    for _ in range(iterations):
        neighbor_schedule = list(current_schedule)
        i, j = random.sample(range(len(neighbor_schedule)), 2)
        neighbor_schedule[i], neighbor_schedule[j] = neighbor_schedule[j], neighbor_schedule[i]

        neighbor_fitness = calculate_fitness(airplane_stream, neighbor_schedule)

        if neighbor_fitness > current_fitness and neighbor_schedule not in tabu_list:
            current_schedule, current_fitness = neighbor_schedule, neighbor_fitness

            if current_fitness > best_fitness:
                best_schedule, best_fitness = current_schedule, current_fitness

        # Adiciona o horário atual à lista de tabu
        tabu_list.append(current_schedule)

        if len(tabu_list) > tabu_size:
            tabu_list.pop(0)

    unable_to_land = [airplane_id for airplane_id, landing_time in best_schedule if (get_fuel_remaining(airplane_stream, airplane_id, landing_time) < 0)]
    return best_schedule, unable_to_land
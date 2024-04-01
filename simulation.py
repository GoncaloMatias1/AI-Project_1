import random
import pandas as pd
import numpy as np
from collections import deque


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

        self.is_urgent = self.fuel_level_final < self.emergency_fuel or self.remaining_flying_time < 1


def generate_airplane_stream(num_airplanes, min_fuel, max_fuel, min_arrival_time, max_arrival_time):
    return [Airplane(i, min_fuel, max_fuel, min_arrival_time, max_arrival_time) for i in range(1, num_airplanes + 1)]

def schedule_landings(airplane_stream):
    sorted_airplanes = sorted(airplane_stream, key=lambda x: x.expected_landing_time)
    landing_schedule = []
    landing_strip_availability = [0, 0, 0]  
    landing_strip_index = 0

    for airplane in sorted_airplanes:
        if airplane.fuel_level_final == 0:
            airplane.fuel_level_final = airplane.fuel_level

        chosen_strip = landing_strip_index % 3
        next_available_time_with_gap = landing_strip_availability[chosen_strip] + 3/60
        is_urgent = airplane.fuel_level_final < airplane.emergency_fuel or airplane.remaining_flying_time < airplane.expected_landing_time
        actual_landing_time = max(airplane.expected_landing_time, next_available_time_with_gap)

        if is_urgent and actual_landing_time > airplane.remaining_flying_time:
            actual_landing_time = airplane.remaining_flying_time 

        if airplane.fuel_level_final == airplane.emergency_fuel:
            actual_landing_time = max(actual_landing_time, airplane.expected_landing_time)

        landing_strip_availability[chosen_strip] = actual_landing_time + 3
        landing_schedule.append((airplane.id, actual_landing_time, is_urgent, chosen_strip + 1))

        landing_strip_index += 1

    return pd.DataFrame(landing_schedule, columns=["Airplane ID", "Actual Landing Time", "Urgent", "Landing Strip"])

#hill climbing
def evaluate_landing_schedule(landing_schedule_df, airplane_stream):
    for index, row in landing_schedule_df.iterrows():
        airplane = next((ap for ap in airplane_stream if ap.id == row['Airplane ID']), None)
        if airplane:
            difference = abs(airplane.expected_landing_time - row['Actual Landing Time'])
            urgency_penalty = 100 if airplane.is_urgent else 0
            score = 1000 - difference - urgency_penalty
            landing_schedule_df.at[index, 'Score'] = score

    total_score = landing_schedule_df['Score'].sum()
    return total_score

def get_successors(landing_schedule_df, airplane_stream, num_successors=15):
    successors = []
    num_planes = len(landing_schedule_df)
    for _ in range(num_successors):
        i, j = random.sample(range(num_planes), 2)  # Randomly choose two planes to swap
        new_schedule_df = landing_schedule_df.copy()
        new_schedule_df.iloc[i], new_schedule_df.iloc[j] = new_schedule_df.iloc[j].copy(), new_schedule_df.iloc[i].copy()
        # Recalculate the Actual Landing Time and the scores for each plane in the new schedule
        strip_availability_times = deque([0, 0, 0])  # Initialize with 3 strips all available at time 0
        for index, row in new_schedule_df.iterrows():
            airplane = next((ap for ap in airplane_stream if ap.id == row['Airplane ID']), None)
            if airplane:
                current_time = max(strip_availability_times[0], airplane.expected_landing_time)
                new_schedule_df.at[index, 'Actual Landing Time'] = current_time
                difference = abs(airplane.expected_landing_time - current_time)
                urgency_penalty = 100 if airplane.is_urgent else 0
                score = 1000 - difference - urgency_penalty
                new_schedule_df.at[index, 'Score'] = score
                strip_availability_times.popleft()  # Remove the strip that was just used
                strip_availability_times.append(current_time + 3)  # Add the time when the strip will become available again
                strip_availability_times = deque(sorted(strip_availability_times))  # Sort the times to ensure the earliest is always first
        successors.append(new_schedule_df)
    return successors

def generate_initial_schedule(airplane_stream):
    shuffled_stream = random.sample(airplane_stream, len(airplane_stream))
    return schedule_landings(shuffled_stream)

def select_parents(population, fitness_scores, num_parents):
    fitness_scores = np.array(fitness_scores)
    probabilities = 1 / (1 + fitness_scores)
    probabilities /= probabilities.sum()
    selected_indices = np.random.choice(range(len(population)), size=num_parents, replace=False, p=probabilities)
    return [population[i] for i in selected_indices]

def crossover(parents, crossover_rate):
    offspring = []
    for _ in range(len(parents) // 2):
        parent1, parent2 = random.sample(parents, 2)
        if random.random() < crossover_rate:
            crossover_point = random.randint(1, parent1.shape[0] - 2)  # asim evitamos extremos
            child1 = pd.concat([parent1.iloc[:crossover_point], parent2.iloc[crossover_point:]]).reset_index(drop=True)
            child2 = pd.concat([parent2.iloc[:crossover_point], parent1.iloc[crossover_point:]]).reset_index(drop=True)
            offspring.extend([child1, child2])
        else:
            offspring.extend([parent1, parent2])
    return offspring

def mutate(schedule, mutation_rate, airplane_stream):
    for index in range(len(schedule)):
        if random.random() < mutation_rate:
            replacement_plane = random.choice(airplane_stream)
            replacement_index = schedule[schedule['Airplane ID'] == replacement_plane.id].index[0]
            schedule.at[index, 'Actual Landing Time'], schedule.at[replacement_index, 'Actual Landing Time'] = schedule.at[replacement_index, 'Actual Landing Time'], schedule.at[index, 'Actual Landing Time']
    return schedule

#uma pontuação de 0 indica um evento de aterragem ótimo ou sem penalizações

#As pontuações diferentes de zero sugerem penalizações devidas a desvios das condições óptimas, tais como atrasos ou não resposta adequada à urgência devido a pouco combustível.

#O objetivo do AG é minimizar estas pontuações, procurando obter uma pontuação total de 0 no programa, o que indica que não há penalizações em todos os eventos de aterragem e, por conseguinte, um programa ótimo, tendo em conta as restrições e os objectivos definidos.
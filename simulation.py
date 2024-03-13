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

        if airplane.fuel_level_final == airplane.emergency_fuel:
            actual_landing_time = max(actual_landing_time, airplane.expected_landing_time)

        landing_strip_availability[chosen_strip] = actual_landing_time + 3

        landing_schedule.append((airplane.id, actual_landing_time, is_urgent, chosen_strip + 1))

    return pd.DataFrame(landing_schedule, columns=["Airplane ID", "Actual Landing Time", "Urgent", "Landing Strip"])


def evaluate_landing_schedule(landing_schedule_df, airplane_stream):
    for index, row in landing_schedule_df.iterrows():
        airplane = next((ap for ap in airplane_stream if ap.id == row['Airplane ID']), None)
        if airplane:
            difference = abs(airplane.expected_landing_time - row['Actual Landing Time'])
            urgency_penalty = 100 if airplane.is_urgent else 0
            score = difference + urgency_penalty
            landing_schedule_df.at[index, 'Score'] = score

    total_score = landing_schedule_df['Score'].sum()
    return total_score


def get_successors(landing_schedule_df, airplane_stream): #esta funcao faz parte do hill climbing, o que faz é gerar sucessores para o estado atual fazendo pequenas alterações nos tempos de aterragem
    successors = []
    for i in range(len(landing_schedule_df)): #este loop irá iterar sobre todos os aviões de modo a gerar sucessores
        for j in range(i + 1, len(landing_schedule_df)): 
            # esta dataframe é uma cópia da original, de modo a que possamos fazer alterações sem afetar o estado atual
            new_schedule_df = landing_schedule_df.copy()
            # estamos a trocar os tempos de aterragem de dois aviões, de modo a gerar um sucessor
            # o sucessor serve para que possamos comparar o score do estado atual com o score do sucessor
            new_schedule_df.iloc[i], new_schedule_df.iloc[j] = new_schedule_df.iloc[j].copy(), new_schedule_df.iloc[i].copy()
            successors.append(new_schedule_df)
    return successors
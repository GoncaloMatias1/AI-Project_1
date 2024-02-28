import random
import pandas as pd

class Airplane:
    def __init__(self, id):
        self.id = id
        self.arriving_fuel_level = random.uniform(1000, 5000)
        self.fuel_consumption_rate = random.uniform(5, 20)
        self.expected_landing_time = random.uniform(10, 120)
        self.remaining_flying_time = self.arriving_fuel_level / self.fuel_consumption_rate
        self.urgency = self.remaining_flying_time - 60

#função para gerar um stream de aviões com características aleatórias
def generate_airplane_stream(num_airplanes):
    airplane_stream = [Airplane(i) for i in range(1, num_airplanes + 1)]
    return airplane_stream

def print_airplane_info(airplane_stream):
    for airplane in airplane_stream:
        print(f"Airplane {airplane.id} has fuel level {airplane.arriving_fuel_level:.2f} L, "
              f"consumption rate {airplane.fuel_consumption_rate:.2f} L/min, "
              f"expected landing time {airplane.expected_landing_time:.2f} min, "
              f"remaining flying time {airplane.remaining_flying_time:.2f} min, "
              f"urgency {'high' if airplane.urgency < 0 else 'normal'}.")

def schedule_landings(airplane_stream):
    
    #da sort aos aviões por urgência (aqueles com urgência menor que zero precisam de um pouso urgente)    
    sorted_airplanes = sorted(airplane_stream, key=lambda x: (x.urgency >= 0, x.expected_landing_time))
    
    landing_schedule = []
    landing_time = 0  #começar o landing_time em 0
    for airplane in sorted_airplanes:
        #condicoes para determinar se o avião precisa de um pouso urgente
        if airplane.urgency < 0 or airplane.expected_landing_time <= landing_time or airplane.remaining_flying_time <= landing_time:
            actual_landing_time = max(landing_time, airplane.expected_landing_time)
        else:
            #se não precisar de um pouso urgente, o avião pousa no tempo esperado
            actual_landing_time = airplane.expected_landing_time
        #adiciona o avião e o tempo de pouso ao landing_schedule
        landing_schedule.append((airplane.id, actual_landing_time, airplane.urgency < 0))
        #adiciona 3 minutos ao tempo de pouso para o próximo avião
        landing_time = actual_landing_time + 3
    
    return pd.DataFrame(landing_schedule, columns=["Airplane", "Landing Time", "Urgent"])


airplane_stream = generate_airplane_stream(50)
print_airplane_info(airplane_stream)

landing_schedule = schedule_landings(airplane_stream)
print("\nLanding Schedule DataFrame")
print(landing_schedule.sort_values(by="Landing Time"))

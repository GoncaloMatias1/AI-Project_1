from airplane import Airplane

def generate_airplane_stream(num_airplanes):
    return [Airplane(airplane_id=i) for i in range(1, num_airplanes + 1)]

def schedule_landings(airplane_stream, num_runways=3):
    # Atualiza a prioridade com base no combustível restante e no tempo esperado de chegada
    for airplane in airplane_stream:
        # Calcula o tempo de voo restante com base no nível de combustível e taxa de consumo
        airplane.remaining_flight_time = airplane.fuel_level / airplane.fuel_consumption_rate
        # Define a prioridade como a diferença entre o tempo restante de voo e o tempo esperado de chegada
        airplane.priority = airplane.remaining_flight_time - airplane.expected_arrival_time

    # Ordena os aviões pela prioridade (menor primeiro, pois representa mais urgência)
    airplane_stream.sort(key=lambda plane: plane.priority)

    runway_availability = [0] * num_runways
    landing_schedule = []

    for airplane in airplane_stream:
        earliest_runway_index = min(range(num_runways), key=lambda i: runway_availability[i])
        # O tempo de pouso é o máximo entre o tempo de chegada esperado e a disponibilidade da pista
        landing_time = max(airplane.expected_arrival_time, runway_availability[earliest_runway_index])
        landing_schedule.append((airplane.airplane_id, landing_time))
        # Atualiza a disponibilidade da pista para 3 minutos após o pouso
        runway_availability[earliest_runway_index] = landing_time + 3

    # Ordena o cronograma de pouso pelo tempo de pouso
    landing_schedule.sort(key=lambda x: x[1])
    return landing_schedule

import pandas as pd
from landing_schedule import schedule_landings, generate_airplane_stream
from airplane import Airplane

def main():
    # User input section
    num_airplanes = int(input("Enter the number of airplanes for the simulation: "))

    # Aqui você pode definir os limites de combustível ou pedir ao usuário para inseri-los
    # min_fuel, max_fuel = 1000, 5000  # Valores padrão
    # Ou peça ao usuário para inserir
    min_fuel = float(input("Enter minimum fuel level (in liters): "))
    max_fuel = float(input("Enter maximum fuel level (in liters): "))

    # Do mesmo modo, defina para os tempos de chegada
    # min_arrival_time, max_arrival_time = 10, 120  # Valores padrão
    # Ou peça ao usuário para inserir
    min_arrival_time = float(input("Enter minimum expected arrival time (in minutes): "))
    max_arrival_time = float(input("Enter maximum expected arrival time (in minutes): "))

    # Inicialize a stream de aviões com os parâmetros inseridos pelo usuário
    airplane_stream = [Airplane(i) for i in range(num_airplanes)]
    for airplane in airplane_stream:
        airplane.fuel_level = random.uniform(min_fuel, max_fuel)
        airplane.expected_arrival_time = random.uniform(min_arrival_time, max_arrival_time)
        airplane.calculate_priority()

    # Agende os pousos com a stream de aviões configurada
    landing_schedule = schedule_landings(airplane_stream)

    # Exiba os resultados
    print("\nLanding Schedule:")
    df_schedule = pd.DataFrame(landing_schedule, columns=['Airplane ID', 'Landing Time (min)'])
    print(df_schedule.to_string(index=False))

if __name__ == "__main__":
    main()

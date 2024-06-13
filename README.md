# **Artificial Intelligence**

**2024 - 2nd Semester**  
**Course:** Artificial Intelligence  
**Grade:** 19/20

## Project developed by:
- Gonçalo Matias
- Tiago Simões
- Fernando Afonso

| Project Number | Project Name                                          | Description                                                                                               |
|----------------|-------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| 1              | Metaheuristics for Optimization/Decision Problems - Airport Landing | In-depth analysis and implementation of metaheuristic algorithms to optimize airport landing schedules.   |

---

## Project Description - Airport Landing

The aim of this project is to design an algorithm to efficiently manage the landing of a continuous stream of airplanes approaching the airport. Each airplane is characterized by its current fuel level and the expected time of arrival for landing.

### Objectives

1. **Safety First**: Ensure that all airplanes land safely with a minimum of one hour of fuel left upon landing. A plane is considered to have landed safely if it touches down without any fuel-related incidents.
2. **Optimal Scheduling**: If it is not possible to guarantee the safety of a plane, prioritize landing it as fast as possible. Additionally, adhere to the expected landing times to minimize delays.

### Constraints

1. The fuel consumption rate of each airplane is constant.
2. The expected landing times must be respected to avoid disruptions in the overall airport schedule.
3. The algorithm should aim to minimize the risk of crashes while optimizing the overall efficiency of landings.
4. The landing strip stays occupied for 3 minutes after the landing. The airport has a theoretical maximum capacity of 3x20=60 landings per hour.

### Critical Failures

The algorithm is considered unsuccessful if any plane crashes during the landing process.

### Objective Function

Minimize the risk of crashes while optimizing for fuel efficiency and adherence to expected landing times.

### Implementation

We use different metaheuristic algorithms such as hill-climbing, simulated annealing, tabu search, and genetic algorithms to solve the optimization problem. The performance of these algorithms is compared using various metrics.

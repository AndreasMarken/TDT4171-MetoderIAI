import random
from numpy import mean, median

### Exercise 2
wheel = [["bar", "bell", "lemon", "cherry"] for _ in range(3)]
cost_for_spin = 1
number_of_starting_coins = 10
number_of_simulations = 1000

def spin() -> tuple:
    x1, x2, x3 = random.randint(0, 3), random.randint(0, 3), random.randint(0, 3)
    return (wheel[0][x1], wheel[1][x2], wheel[2][x3])

def calculate_payout(spin: tuple) -> int:
    # print(f"The spin was: {spin}")
    if (spin[0] == spin[1] == spin[2]):
        if (spin[0] == "bar"):
            return 20
        elif (spin[0] == "bell"):
            return 15
        elif (spin[0] == "lemon"):
            return 5
        elif (spin[0] == "cherry"):
            return 3
    elif (spin[0] == spin[1] == "cherry"):
        return 2
    elif (spin[0] == "cherry"):
        return 1
    return 0

def play_to_broke(coins: int) -> int:
    iterations = 0
    while coins > 0:
        payout = calculate_payout(spin())
        # if (payout > 0):
        #     print(f"Congrats, you won: {payout} coins")
        coins += payout - cost_for_spin
        iterations += 1
    return iterations

def run_simulation(simulations: int) -> list:
    result = []
    for _ in range(simulations):
        result.append(play_to_broke(number_of_starting_coins))
    return result

## Run the exercise 2 code
if __name__ == "__main__":
    for _ in range(10):
        simulations = run_simulation(number_of_simulations)
        # print(simulations)
        print(f"The mean is: {mean(simulations)}.")
        print(f"The median is: {median(simulations)}.")


### Exercise 3
def probability_of_same_birthday(N: int, number_of_simulations=1000) -> float:
    matching_birthdays = 0
    for _ in range(number_of_simulations):
        birthdays = [random.randint(1, 365) for _ in range (N)]
        if len(birthdays) != len(set(birthdays)):
            matching_birthdays += 1
    
    return matching_birthdays / number_of_simulations

# count = 0
# for N in range(10, 51):
#     probability = probability_of_same_birthday(N)
#     print(f"The probability of at least two people sharing the same birthday in a group of {N} people, is {probability}.")
#     if probability >= 0.50:
#         count += 1
# print(f"The proportion of N where the event happens with at least 50% chance: {count/40}")


def create_group_with_birthday_all_year():
    group = []
    while len(set(group)) != 365:
        group.append(random.randint(1, 365))
    return len(group)

## Run the simulation 100 times:
total = 0
for _ in range(100):
    total += create_group_with_birthday_all_year()
print(f"Peter needs to form a group of {total/100} people at random.")
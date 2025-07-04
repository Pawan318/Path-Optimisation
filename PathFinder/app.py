from flask import Flask, render_template, request
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Fix for macOS
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import io
import base64

app = Flask(__name__)

# Genetic Algorithm Setup
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/run", methods=["POST"])
def run():
    try:
        # Retrieve user inputs
        num_locations = int(request.form["num_locations"])
        num_vehicles = int(request.form["num_vehicles"])
        depot_x = int(request.form["depot_x"])
        depot_y = int(request.form["depot_y"])

        # Debugging outputs
        print("Number of Locations:", num_locations)
        print("Number of Vehicles:", num_vehicles)
        print("Depot Coordinates:", (depot_x, depot_y))

        # Generate random locations
        locations = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(num_locations)]
        depot = (depot_x, depot_y)
        print("Locations:", locations)

        # GA Toolbox setup
        toolbox = base.Toolbox()
        toolbox.register("indices", random.sample, range(num_locations), num_locations)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        # Evaluation function
        def evalVRP(individual):
            total_distance = 0
            distances = []
            for i in range(num_vehicles):
                vehicle_indices = [individual[j] for j in range(i, len(individual), num_vehicles)]
                if any(idx >= num_locations for idx in vehicle_indices):
                    raise ValueError(f"Index out of range in individual: {individual}")
                vehicle_route = [depot] + [locations[idx] for idx in vehicle_indices] + [depot]
                vehicle_distance = sum(
                    np.linalg.norm(np.array(vehicle_route[k+1]) - np.array(vehicle_route[k]))
                    for k in range(len(vehicle_route)-1)
                )
                total_distance += vehicle_distance
                distances.append(vehicle_distance)
            balance_penalty = np.std(distances)
            return total_distance, balance_penalty

        toolbox.register("evaluate", evalVRP)
        toolbox.register("mate", tools.cxPartialyMatched)
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)

        # Run Genetic Algorithm
        pop = toolbox.population(n=300)
        hof = tools.HallOfFame(1)
        algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.2, ngen=300, halloffame=hof, verbose=False)

        # Plot best solution
        best_individual = hof[0]
        plt.figure()
        for (x, y) in locations:
            plt.plot(x, y, 'bo')
        plt.plot(depot[0], depot[1], 'rs')
        distances = []  # Store distances for each vehicle
        for i in range(num_vehicles):
            vehicle_indices = [best_individual[j] for j in range(i, len(best_individual), num_vehicles)]
            vehicle_route = [depot] + [locations[idx] for idx in vehicle_indices] + [depot]
            plt.plot(*zip(*vehicle_route), '-', label=f"Vehicle {i+1}")
            vehicle_distance = sum(
                np.linalg.norm(np.array(vehicle_route[k+1]) - np.array(vehicle_route[k]))
                for k in range(len(vehicle_route)-1)
            )
            distances.append(vehicle_distance)

        total_distance = sum(distances)
        balance_penalty = np.std(distances)

        plt.title("Vehicle Routing Problem Solution")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend()

        # Convert plot to base64 image
        img = io.BytesIO()
        plt.savefig(img, format="png")
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        # Pass summary data
        return render_template(
            "result.html",
            plot_url=plot_url,
            total_distance=total_distance,
            balance_penalty=balance_penalty,
            distances=distances,
            num_vehicles=num_vehicles,
            num_locations=num_locations,
            depot_x=depot_x,
            depot_y=depot_y,
        )
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)

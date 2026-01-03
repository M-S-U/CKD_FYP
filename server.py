import flwr as fl
from flwr.server import ServerConfig

# Define the strategy
strategy = fl.server.strategy.FedAvg(
    min_fit_clients=3,
    min_available_clients=3,
)

# Start the server on your local machine
if __name__ == "__main__":
    print("🚀 FL Server starting on localhost:8080")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=ServerConfig(num_rounds=5),
        strategy=strategy,
    )




import networkx as nx
import matplotlib.pyplot as plt

# Define LSTM model layers as nodes
layers = [
    "Input Layer (60 Time Steps, 1 Feature)",
    "LSTM Layer (50 Units, Return Sequences=True)",
    "Dropout (0.2)",
    "LSTM Layer (50 Units, Return Sequences=False)",
    "Dropout (0.2)",
    "Dense Layer (25 Units)",
    "Dense Layer (1 Unit, Linear Activation)",
    "Output: Predicted Stock Price"
]

# Create graph structure
G = nx.DiGraph()

# Add edges to represent data flow
for i in range(len(layers) - 1):
    G.add_edge(layers[i], layers[i + 1])

# Draw the LSTM architecture
plt.figure(figsize=(10, 6))
pos = nx.spring_layout(G, seed=42)  # Layout for better visualization
nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray",
        node_size=4000, font_size=10, font_weight="bold", arrows=True)
plt.title("LSTM Network Architecture Diagram")
plt.savefig("lstm_network_architecture.png")  # Save the diagram
plt.show()

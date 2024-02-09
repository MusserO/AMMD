from AMMD_functions import *
import itertools
path_avg_flights = "datasets/average_flight_times.csv"

# Now read that file as a networkx file.
G = nx.read_weighted_edgelist(path_avg_flights, delimiter=',',create_using=nx.DiGraph)
largest = max(nx.strongly_connected_components(G), key=len)
G.remove_nodes_from([n for n in G.nodes() if n not in largest])
G = nx.convert_node_labels_to_integers(G, label_attribute="old_id") # relabel so we can use numpy weight matrix.

mapping = {k: G.nodes[k]["old_id"] for k in G.nodes()}
p = dict(nx.shortest_path_length(G, weight="weight"))

D_weights = np.zeros((G.number_of_nodes(), G.number_of_nodes()))
for u, v in p.items():
        for x,y in v.items():
            D_weights[u, x] = y

val, S = BAC3(D_weights, 5)
S_orig = [mapping[v] for v in S]
print(val, S_orig)

for i, j in itertools.product(S, S):
    if i != j:
            print(mapping[i],mapping[j], D_weights[i,j])

            

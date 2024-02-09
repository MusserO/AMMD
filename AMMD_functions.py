import networkx as nx
from networkx.algorithms.flow import dinitz
import matplotlib.pyplot as plt
from collections import defaultdict
from collections import deque
import tsplib95
import bisect
import time
import numpy as np
from datetime import datetime
import os
import pickle
import signal
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix

###### Non-TSP data #####

c_elegans = "datasets/weighted_digraphs/celegansneural_weighted.txt"
moreno_health = "datasets/weighted_digraphs/moreno_health_weighted.txt"
wiki_vote_snap = "datasets/weighted_digraphs/Wiki-Vote.txt"
gnutella_snap = "datasets/weighted_digraphs/Gnutella_snap.txt"
bitcoin = 'datasets/weighted_digraphs/bitcoin.txt'
airport = 'datasets/weighted_digraphs/airport.txt'
openflight = 'datasets/weighted_digraphs/openflight.txt'
cora = 'datasets/weighted_digraphs/cora.txt'

ft70 = 'datasets/ALL_atsp/ft70.atsp'
kro124 = 'datasets/ALL_atsp/kro124p.atsp'
rbg323 = 'datasets/ALL_atsp/rbg323.atsp'

##########################################################################################
############### Experiments ##############################################################
##########################################################################################

### Create synthetic instances ###

def create_scc_scale_free(n):
    G = nx.scale_free_graph(n, seed=0)
    largest_cc = max(nx.weakly_connected_components(G), key=len) # largest weakly connected component.
    G = G.subgraph(largest_cc)
    G = nx.convert_node_labels_to_integers(G) # relabel so we can use numpy weight matrix
    
    # add reversed edges to make it fully connected
    Grev = G.reverse(copy=True)
    G = nx.compose(G,Grev)

    # set weights according to shortest path distances to ensure triangle inequality
    D_weights = np.full((G.number_of_nodes(), G.number_of_nodes()), np.inf)
    edges = np.array([(u, v) for u, v in G.edges()]).astype(int)
    D_weights[edges[:,0], edges[:,1]] = 1
    np.fill_diagonal(D_weights, 0)
    D_weights = metric_closure(D_weights)

    return D_weights

def create_complete_digraph(n, random_weights_max=None, random_weights_min=1):
    if random_weights_max is not None:
        D_weights = np.random.randint(random_weights_min, random_weights_max, size = (n, n))
        D_weights = D_weights.astype(float, copy=False)
    else:
        D_weights = np.random.rand(n,n)
    np.fill_diagonal(D_weights, 0)
    D_weights = metric_closure(D_weights)
    return D_weights

def different_size_graphs(graph_sizes):
    for graph_size in graph_sizes:
        yield create_scc_scale_free(graph_size)

def different_weight_graphs(n_vertices, max_weight_ranges):
    for max_weight_range in max_weight_ranges:
        yield create_complete_digraph(n_vertices, max_weight_range)

### Timing experiments ###

def signal_handler(signum, frame): raise TimeoutError()

def test_running_times_on_graphs(algorithms, graphs, k, graph_parameters, hard_time_limit=None, soft_time_limit=None, results_file_path=None, log_file=None, verbose=True):
    # Test the running times of given algorithms on the given graphs, for fixed k.
    # The graphs are given as a list or generator of D_weights matrices.
    # The execution of an algorithm is halted if it exceeds the hard time limit, 
    # and if it exceeds the soft time limit, the algorithm is not tested on the remaining graphs.
    # Returns a matrix with the running times, where the first column contains the graph parameters.

    if hard_time_limit is not None:
        signal.signal(signal.SIGALRM, signal_handler)
    time_limit_exceeded = np.repeat(False, len(algorithms))
    results_matrix = np.zeros((len(graph_parameters),len(algorithms)+1))
    for i, D_weights in enumerate(graphs):
        print_log(f"Testing {graph_parameters[i]}", log_file, verbose)
        results_matrix[i,0] = graph_parameters[i]
        for j, (algorithm, algorithm_name) in enumerate(algorithms):
            if not time_limit_exceeded[j]:
                if hard_time_limit is None:
                    _, running_time = time_and_test_results(algorithm, algorithm_name, D_weights, k, results_file=log_file, verbose=verbose)
                else:
                    # Run algorithm with a hard time limit
                    signal.alarm(hard_time_limit)
                    try:
                        _, running_time = time_and_test_results(algorithm, algorithm_name, D_weights, k, results_file=log_file, verbose=verbose)
                    except TimeoutError:
                        print_log(f"Time limit of {hard_time_limit} seconds exceeded for {algorithm_name}", log_file, verbose)
                        running_time = np.inf
                    signal.alarm(0)
                if running_time != np.inf:
                    # Convert running time to seconds
                    running_time = running_time.total_seconds()
                if soft_time_limit is not None:
                    time_limit_exceeded[j] = running_time > soft_time_limit
                results_matrix[i,j+1] = running_time
            else:
                results_matrix[i,j+1] = np.inf
        print_log("", log_file, verbose)

    if results_file_path is not None:
        np.savetxt(results_file_path, results_matrix)
    return results_matrix
    
### Performance experiments ###

def Performance_one_Dataset(file_path, krange=None):
    # file_paths is a path to the dataset, it assumes *forward* slashes '.../.../'.
    # Tested algorithms = [BAC1, BAC2, BAC3, optimum_MaxIndSet_binsearch, randomk, largest_dmin_next].
    # krange is the range of k values for which we test, e.g. krange = [2, 4, 8, 16, 32, 64].
    # This automatically write to the corrext filename 'Performance_datasetname.txt'

    D_weights = read_graph_from_file(file_path)

    if krange is None:
        # Create list of increasing powers of two for k until reaching the number of vertices
        n = D_weights.shape[0]
        krange = [2**i for i in range(1, int(np.log2(n-0.01))+1)] + [n]
    
    ks = len(krange)
    results_matrix = np.zeros((ks,6)) # Matrix rows are k values, Matrix columns are the different Algorithms.
    i = 0
    while i < len(krange):
        print(i)
        start = time.time()
        results_matrix[i,0] = BAC1(D_weights, krange[i])
        end = time.time()
        print('BAC1 done in %.2f!' % (end - start))

        start = time.time()
        results_matrix[i,1] = BAC2(D_weights, krange[i])
        end = time.time()
        print('BAC2 done in %.2f!' % (end - start))

        start = time.time()
        results_matrix[i,2] = BAC3(D_weights, krange[i])
        end = time.time()
        print('BAC3 done in %.2f!' % (end - start))

        start = time.time()
        results_matrix[i,3] = optimum_MaxIndSet_binsearch(D_weights, krange[i])
        end = time.time()
        print('opt done in %.2f!' % (end - start))

        start = time.time()
        results_matrix[i,4] = randomk(D_weights, krange[i], 10)
        end = time.time()
        print('randomk done in %.2f!' % (end - start))

        start = time.time()
        results_matrix[i,5] = largest_dmin_next(D_weights, krange[i])
        end = time.time()
        print('largest dmin next done in %.2f!' % (end - start))
        i += 1
    # Add the krange vector as the first column:
    results_matrix = np.insert(results_matrix, 0, krange, axis=1)

    # Automatically writing to correct file name:
    x = file_path.rsplit('/', 1)[-1]
    name = x.split('.')[0]
    np.savetxt('Performance_'+name+'.txt', results_matrix ,fmt='%.0f')
    return results_matrix

def time_and_test_results(function, function_name, D_weights, k, results_file=None, verbose=True):
    start_time = datetime.now()
    results = function(D_weights, k)
    running_time = datetime.now() - start_time
    print_log(f"{function_name} done: {running_time}", results_file, verbose)
    return results, running_time

def test_one_dataset(file_path, algorithms, k, results_file=None, verbose=True):
    results = []
    
    D_weights = read_graph_from_file(file_path, results_file, verbose)

    for algorithm, algorithm_name in algorithms:
        result, _ = time_and_test_results(algorithm, algorithm_name, D_weights, k, results_file=results_file, verbose=verbose)
        results.append(result)
    
    return results

def test_all_datasets(datasets, algorithms, k, results_file=None, verbose=True):
    for dataset in datasets:
        print_log(f"Testing dataset {dataset}", results_file, verbose)
        results = test_one_dataset(dataset, algorithms, k, results_file=results_file, verbose=verbose)
        print_log(results, results_file, verbose)
        print_log('**************', results_file, verbose)

##########################################################################################
############### Data File Reading ########################################################
##########################################################################################
        
def read_graph_from_file(file_path, results_file=None, verbose=True):
    # Load D_weights from pickle file if exists, otherwise read file and save to pickle.
    file_path_without_extension = file_path[:-4] if file_path[-4:] == '.txt' else file_path[:-5]
    if os.path.isfile(file_path_without_extension + '.pkl'):
        D_weights = pickle.load(open(file_path_without_extension + '.pkl', 'rb'))
    else:
        start_time = datetime.now()
        if file_path[-4:] == '.txt':
            # Then we have a weighted text file
            D_weights = read_weighted_graph_lscc(file_path)
        else:
            # Otherwise ATSP file.
            D_weights = read_tsp_graph(file_path)
        pickle.dump(D_weights, open(file_path_without_extension + '.pkl', 'wb'))
        print_log(f'Preprocessing completed: {datetime.now() - start_time}', results_file, verbose)
    return D_weights
    
def graph_to_D_weights(G):
    G = nx.convert_node_labels_to_integers(G) # relabel so we can use numpy weight matrix
    D_weights = np.full((G.number_of_nodes(), G.number_of_nodes()), np.inf)

    edges = np.array(G.edges())
    weights = np.array([G[u][v].get('weight', 1) for u, v in G.edges()])
    D_weights[edges[:, 0], edges[:, 1]] = weights

    np.fill_diagonal(D_weights, 0)
    D_weights = metric_closure(D_weights)

    return D_weights

def read_weighted_graph_lscc(file_path):
    # Reads a 3 column edge list as a weighted DiGraph, takes the largest strongly connected comp (LSCC).
    # Returns D_weights, which is a 2D numpy array containing edge weights of the metric closure of the LSCC.
    
    G = nx.read_weighted_edgelist(file_path, create_using=nx.DiGraph)
    largest_scc = max(nx.strongly_connected_components(G), key=len) # nodes from largest SCC
    G.remove_nodes_from([n for n in G.nodes() if n not in largest_scc]) # removing other nodes.
    return graph_to_D_weights(G)

def read_weighted_graph(file_path):
    # Reads a 3 column edge list as a weighted DiGraph.
    # Returns D_weights, which is a 2D numpy array containing edge weights of the metric closure.
    # Since we do not take the largest SCC, some node pair distances might be infinite!
    # We add an 'inf' value for the corresponding edges.
    
    G = nx.read_weighted_edgelist(file_path, create_using=nx.DiGraph)
    return graph_to_D_weights(G)


def read_tsp_graph(file_path):
    # This reads TSP data from TSPLIB
    # All those graphs are already complete, but we still need to take metric closure to ensure the triangle inequality.
    with open(file_path) as f:
        text = f.read()

    problem = tsplib95.parse(text)
    G = problem.get_graph() # creates a NetworkX graph.
    return graph_to_D_weights(G)

##########################################################################################
################ Main Algorithms #########################################################
##########################################################################################

def BAC1(D_weights, k):
    # D_weights is a 2D numpy array with the directed distances (satisfying triangle ineq.)
    # k is an integer between 1 and n, denoting the size of the set of vertices we want to return.
    # Vanilla version of our algorithm. We iterate over all possible R values, and draw G_aux if d_ij < R/2k.

    if k<=1 or k>D_weights.shape[0]:
        raise ValueError("integer k needs to be 2<=k<=n.")

    # Create a matrices of min and max values between each edge (u,v) and (v,u)
    D_min_weights = np.minimum(D_weights, D_weights.T)
    D_max_weights = np.maximum(D_weights, D_weights.T)

    # A list of the unique positive distances
    unique_d = np.unique(D_min_weights[D_min_weights > 0])

    S = None
    best_score = 0

    greedy_start_node = find_greedy_start_node_index(D_min_weights)
    i = 0
    next_i = 0
    while i < len(unique_d):
        R = unique_d[i]
        if i >= next_i:
            centers = cluster_dmax(D_min_weights, D_max_weights, R, start_node=greedy_start_node)
            if len(centers) < k:
                break
            # Find the index of the next R value that changes the set of centers.
            next_i = find_next_center_distance_index(D_max_weights, unique_d, centers, i)
        G_aux = create_auxiliary_graph(R/(2*k), centers, D_weights) # node ID's still the same as in D.
        sol = extract_solution(G_aux, D_min_weights, k)
        if len(sol) == k:
            best_score, S = max_of_scores(sol, best_score, S, D_min_weights)
        i += 1
    
    return best_score

def BAC2(D_weights, k):
    # D_weights is a 2D numpy array with the directed distances (satisfying triangle ineq.)
    # k is an integer between 1 and n, denoting the size of the set of vertices we want to return.
    # Second variant of our algorithm.
    # Iterates over all possible R values, but binary searches for a *bigger* treshold value than R/2k, when creating G_aux as e in E <-> d_ij < R/2k.
    # We will look for values in between [R/2k, R], *if* we find a solution for R/2k.

    if k<=1 or k>D_weights.shape[0]:
        raise ValueError("integer k needs to be 2<=k<=n.")

    # Create a matrices of min and max values between each edge (u,v) and (v,u)
    D_min_weights = np.minimum(D_weights, D_weights.T)
    D_max_weights = np.maximum(D_weights, D_weights.T)

    # A list of the unique positive distances
    unique_d = np.unique(D_min_weights[D_min_weights > 0])

    S = None
    best_score = 0

    greedy_start_node = find_greedy_start_node_index(D_min_weights)
    i = 0
    while i < len(unique_d):
        R = unique_d[i]
        centers = cluster_dmax(D_min_weights, D_max_weights, R, start_node=greedy_start_node)
        if len(centers) < k:
            break
        # Find the index of the next R value that changes the set of centers.
        next_i = find_next_center_distance_index(D_max_weights, unique_d, centers, i)
        G_aux = create_auxiliary_graph(R/(2*k), centers, D_weights) # node ID's still the same as in D.
        sol = extract_solution(G_aux, D_min_weights, k)
        if len(sol) == k:
            best_score, S = max_of_scores(sol, best_score, S, D_min_weights)
            v1 = R/(2*k)
            a = bisect.bisect_left(unique_d, v1)  # the index of first value >= than v1.
            b = next_i-1
            # Find the largest cutoff R2 that still gives a solution of size >=k 
            while a<=b:
                midpoint = (a + b)//2
                R2 = unique_d[midpoint]
                G_aux = create_auxiliary_graph(R2, centers, D_weights)
                sol = extract_solution(G_aux, D_min_weights, k)
                if len(sol) == k:
                    best_score, S = max_of_scores(sol, best_score, S, D_min_weights)
                    a = midpoint+1
                else:
                    b = midpoint-1
        i = next_i
    
    return best_score

def BAC3(D_weights, k):
    # D_weights is a 2D numpy array with the directed distances (satisfying triangle ineq.)
    # k is an integer between 1 and n, denoting the size of the set of vertices we want to return.
    # Third variant of our algorithm.

    if k<=1 or k>D_weights.shape[0]:
        raise ValueError("integer k needs to be 2<=k<=n.")

    # Create a matrices of min and max values between each edge (u,v) and (v,u)
    D_min_weights = np.minimum(D_weights, D_weights.T)
    D_max_weights = np.maximum(D_weights, D_weights.T)

    # A list of the unique positive distances
    unique_d = np.unique(D_min_weights[D_min_weights > 0])

    a = 0
    b = len(unique_d)-1

    S = None
    best_score = 0

    greedy_start_node = find_greedy_start_node_index(D_min_weights)
    while a<=b:
        midpoint = (a + b)//2
        R = unique_d[midpoint]

        centers = cluster_dmax(D_min_weights, D_max_weights, R, start_node=greedy_start_node)
        G_aux = create_auxiliary_graph(R/(2*k), centers, D_weights) # node ID's still the same as in D.
        sol = extract_solution(G_aux, D_min_weights, k)
        if len(sol) == k:
            best_score, S = max_of_scores(sol, best_score, S, D_min_weights)
            a = midpoint+1
        else:
            b = midpoint-1
    

    b = a-1
    R = unique_d[a-1]
    centers = cluster_dmax(D_min_weights, D_max_weights, R, start_node=greedy_start_node)

    v1 = R/(2*k)
    a = bisect.bisect_left(unique_d, v1)  # the index of first value >= than v1.
    # Find the largest cutoff R2 that still gives a solution of size >=k 
    while a<=b:
        midpoint = (a + b)//2
        R2 = unique_d[midpoint]
        G_aux = create_auxiliary_graph(R2, centers, D_weights)
        sol = extract_solution(G_aux, D_min_weights, k)
        if len(sol) == k:
            best_score, S = max_of_scores(sol, best_score, S, D_min_weights)
            a = midpoint+1
        else:
            b = midpoint-1

    return best_score

##########################################################################################
################ Heuristics #########################################################
##########################################################################################

def randomk(D_weights, k, repeats=10):
    # This algorithm picks a random k-subset as solution.
    # Picks the best result found among several repeats
    teller = 1
    best_val = 0
    l = range(D_weights.shape[0])
    while teller <= repeats:
        teller += 1
        cands = list(np.random.choice(l, k, replace=False))
        cands_val = div_score(D_weights,cands)
        if cands_val >= best_val:
            best_val = cands_val
    return best_val

def find_furthest_node_index(D_min_weights, nodes=None):
    # Returns the node with the largest min distance to all other nodes.
    if nodes is None:
        A = D_min_weights.copy()
    else:
        A = D_min_weights[nodes, :][:, nodes]
    np.fill_diagonal(A, np.inf)
    return np.argmax(np.min(A, axis=1))

def find_greedy_start_node_index(D_min_weights, nodes=None):
    # Finds a good starting node for the greedy algorithm.
    # Returns the first node from the maximum edge.
    #return find_furthest_node_index(D_min_weights, nodes)
    return find_maximum_edge_index(D_min_weights, nodes)[0]

def find_maximum_edge_index(D_min_weights, nodes=None):
    if nodes is None:
        return np.unravel_index(np.argmax(D_min_weights), D_min_weights.shape)
    return np.unravel_index(np.argmax(D_min_weights[nodes, :][:, nodes]), (len(nodes),len(nodes)))

def greedy_select(D_min_weights, k, nodes=None, start_node_index=None):
    # Greedily select k nodes, based on the largest d_min distance.
    if start_node_index is None:
        #v0 = np.random.choice(nodes, 1, replace=False)[0]
        v0_index = find_greedy_start_node_index(D_min_weights, nodes)
    else:
        v0_index = start_node_index
    if nodes is None:
        nodes = np.arange(D_min_weights.shape[0])
    v0 = nodes[v0_index]
    distances = D_min_weights[nodes, v0]
    distances[v0_index] = -1

    sol = [v0]
    while len(sol) < k:
        ind = np.argmax(distances)
        next_v = nodes[ind]
        sol.append(next_v)
        distances = np.minimum(distances, D_min_weights[nodes, next_v])
        distances[ind] = -1
    return sol

def largest_dmin_next(D_weights, k):
    # This algorithm mimics the 2-approximation of symmetric Max-Min Diversification.
    D_min_weights = np.minimum(D_weights, D_weights.T)
    sol = greedy_select(D_min_weights, k)
    return div_score_D_min(D_min_weights, sol)

##########################################################################################
################ Help Functions ##########################################################
##########################################################################################

def print_log(message, log_file=None, verbose=True):
    if verbose:
        print(message)
    if log_file is not None:
        print(message, file=log_file)

def cluster_dmax(D_min_weights, D_max_weights, R, start_node=None):
    # The d_max clustering phase from our paper
    nodes = np.arange(D_min_weights.shape[0])
    if R == 0:
        return nodes

    if start_node is None:
        start_node = find_greedy_start_node_index(D_min_weights)
    c = start_node
    
    centers = [c]
    distances = D_min_weights[nodes, c]
    distances[c] = -1

    while True:
        distances[D_max_weights[nodes, c] < R] = -1
        distances = np.minimum(distances, D_min_weights[nodes, c])
        c = np.argmax(distances)
        if distances[c] == -1:
            break
        centers.append(c)

    return np.array(centers)

def find_cycle_or_antichain_or_path(G):
    if len(G.nodes()) <= 1:
        return list(G.nodes())
    
    # Find the strongly connected components
    Gc = nx.condensation(G, scc=None) # reindexed DAG with nodes = (0,1,2,...).
    member_dict = nx.get_node_attributes(Gc, "members") # scc membership dict, {0:{a,b,c}, 1:{d,e},...}.
    check_cycle_exists = [x for x, v in member_dict.items() if len(v)>1]

    # For antichain and path, keep working with the condensation DAG
    topo_order = list(nx.topological_sort(Gc))
    antichain = maxAntiChain(Gc, topo_order)
    path = longest_shortest_path2(Gc, topo_order)[::2]

    if check_cycle_exists:
        cycle = find_Chordless_cycle(G)[::2]
        if len(cycle) >= len(antichain) and len(cycle) >= len(path):
            return cycle
    if len(antichain) >= len(path):
        # mapping back to original node ID's by picking the first node in each component
        return [list(member_dict[x])[0] for x in antichain]
    return [list(member_dict[x])[0] for x in path]

def extract_solution(G_aux, D_min_weights, k):
    sol = []
    for component in nx.weakly_connected_components(G_aux):
        sol += find_cycle_or_antichain_or_path(G_aux.subgraph(component))
    if len(sol) > k:
        sol = greedy_select(D_min_weights, k, nodes=sol)
    return sol

def create_auxiliary_graph(R, centers, D_weights):
    # After the clustering phase, we create a graph if d_ij < R.
    # k is the size parameter and R the guess of an optimum value.
    G_aux = nx.DiGraph()
    G_aux.add_nodes_from(centers)
    #weight_dict = nx.get_edge_attributes(D_new,'weight')
    A = D_weights[centers, :][:, centers]
    np.fill_diagonal(A, np.inf)
    x, y = np.nonzero(A < R)
    
    #edges_filtered = [e for e in D_new.edges() if D_weights[e] < R/(6*k)]
    G_aux.add_edges_from(zip(centers[x], centers[y])) # Note that these edges are unweighted!

    return G_aux

def get_node_pairs(nodes):
    # Create a 2d array of all pair combinations
    rows, cols = np.triu_indices(len(nodes), k=1)
    node_pairs = np.vstack((nodes[rows], nodes[cols])).T
    return node_pairs

def find_next_center_distance_index(D_max_weights, unique_d, centers, i):
    # Find the index of the next R value that changes the set of centers.
    center_pairs = get_node_pairs(centers)
    min_max_center_distance = np.min(D_max_weights[center_pairs[:, 0], center_pairs[:, 1]])
    return bisect.bisect(unique_d, min_max_center_distance, lo=i+1)

def div_score(D_weights, sol):
    # Computes the min-max oAjective function for a given set sol.
    node_pairs = get_node_pairs(np.array(sol))
    return np.min(np.minimum(D_weights[node_pairs[:, 0], node_pairs[:, 1]], 
                              D_weights[node_pairs[:, 1], node_pairs[:, 0]]))

def div_score_D_min(D_min_weights, sol):
    # Computes the min-max oAjective function for a given set sol.
    # D_min_weights is a matrix of the minimum distances between pairs of nodes.
    node_pairs = get_node_pairs(np.array(sol))
    return np.min(D_min_weights[node_pairs[:, 0], node_pairs[:, 1]])

def max_of_scores(sol, best_score, S, D_min_weights):
    score = div_score_D_min(D_min_weights, sol)
    if score >= best_score:
        best_score = score
        S = sol
    return best_score, S

def optimum_MaxIndSet_binsearch(D_weights, k):
    # Guess the optimum R, and create a Max. Ind. Set problem.
    # This is way faster than brute-force.
    opt_val = 0
    D_min_weights = np.minimum(D_weights, D_weights.T)
    unique_d = np.unique(D_min_weights)
    a = 0
    b = len(unique_d)-1
    while a<=b:
        midpoint = (a + b)//2
        R = unique_d[midpoint]
        
        Gr = nx.Graph()
        Gr.add_nodes_from(range(D_weights.shape[0]))
        x, y = np.nonzero(D_min_weights >= R)
        Gr.add_edges_from(zip(x, y))
        clique = nx.max_weight_clique(Gr, weight=None)[0]

        if len(clique)>=k:
            opt_val = div_score_D_min(D_min_weights,clique)
            # try to look for a higher R value
            a = midpoint+1
            if opt_val > unique_d[midpoint+1]:
                a = bisect.bisect(unique_d, opt_val, lo=midpoint+1)
            opt = clique
        else:
            b = midpoint-1      
    return opt_val

class ExplicitZeroCSRMatrix(csr_matrix):
    """A Compressed Sparse Row matrix with explicit zeros outside of the diagonal."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(args[0], np.ndarray):
            matrix_with_filled_zeros = args[0] + 1 # Add 1 to all elements to avoid zero values
            np.fill_diagonal(matrix_with_filled_zeros, 0) # Set diagonal to zero which excludes it
            matrix_with_filled_zeros = csr_matrix(matrix_with_filled_zeros)
            self.data = matrix_with_filled_zeros.data - 1 # Subtract 1 to restore original values
            self.indices = matrix_with_filled_zeros.indices # Indices and indptr now include zeros off diagonal
            self.indptr = matrix_with_filled_zeros.indptr

def metric_closure(D_weights):
    matrix = D_weights
    # Check if there are zero weight edges outside of the diagonal
    diagonal_mask = np.eye(D_weights.shape[0], dtype=bool)
    if np.any(D_weights[~diagonal_mask] == 0):
         # Convert the input to a CSR matrix with explicit zeros
        matrix = ExplicitZeroCSRMatrix(D_weights)
    # Use the dijkstra algorithm to compute the shortest paths for all pairs of nodes
    return dijkstra(matrix)

##########################################################################################
################ Finding Longest Shortest Path in a Directed Acyclic Graph (DAG) #########
##########################################################################################

# largest finite diameter:
# diameter = max([max(j.values()) for (i,j) in nx.shortest_path_length(Gc)])

def dag_shortest_path_source(Gc, topo_order, source):
    # Single source shortest paths to all other vertices, and returning the largest distance.
    # Runs in O(V+E) time, only works for DAGs.
    parent = {source: None}
    d = {source: 0}

    for u in topo_order:
        if u not in d: continue  # get to the source node
        for v in Gc.successors(u):
            if v not in d or d[v] > d[u] + 1:
                d[v] = d[u] + 1
                parent[v] = u

    # vertex with longest shortest path distance from source.
    max_v = max(d.items(), key = lambda x: (x[1], x[0]))[0]

    # Corresponding path.
    y = max_v
    path = []
    length_p = 0
    while y != None:
        path.append(y)
        y = parent[y]
        length_p += 1
    path.reverse()
    return path, max_v, length_p

def longest_shortest_path2(Gc, topo_order):
    # Iterating SS over all sources.
    l = deque(topo_order)
    n = len(topo_order)
    if n <= 1:
        return topo_order
    d_uv = {}
    for i in range(n-1):
        source = l[0]
        _, max_v, length_p = dag_shortest_path_source(Gc, l, source)
        d_uv[(source,max_v)] = length_p
        l.popleft()

    (u,v) = max(d_uv.items(), key = lambda x: (x[1], x[0]))[0] # the nodepair with largest (finite) distance.
    path, _, _ = dag_shortest_path_source(Gc, topo_order, u) # the corresponding path.

    return path


def dag_sp_secondway(Gc, topo_order):
    # Slower way.
    n = len(topo_order)
    parent_all = [{topo_order[i]: None} for i in range(n)] # list of dicts.
    d_all = [{topo_order[i]: 0} for i in range(n)] # list of dicts.

    tel1 = 1
    for u in topo_order:
        tel2 = 0
        for d in d_all[:tel1]:
            #print(u, d, tel2)
            tel2 += 1
            if u not in d: continue  # get to the source node
            for v in Gc.successors(u):
                if v not in d or d[v] > d[u] + 1:
                    d[v] = d[u] + 1
                    parent_all[tel2-1][v] = u    
        tel1 +=1

    max_keynval = [max(d.items(), key = lambda x: (x[1], x[0])) for d in d_all]
    max_lists_vals = [x[1] for x in max_keynval]
    max_lists_keys = [x[0] for x in max_keynval]
    y = max(max_lists_vals)
    z = max_lists_vals.index(y)
    t = max_lists_keys[z]
    s = topo_order[z]

    # Corresponding path.
    path = []
    while t != None:
        path.append(t)
        t = parent_all[z][t]
    path.reverse()
    return path
    
        

##########################################################################################
################ Extracting a Chordless Cycle from DiGraph ##############################
##########################################################################################

# cycle_edgelist = nx.find_cycle(G, orientation="original")
# cycle = [e[0] for e in cycle_edgelist]

def shortcut_Rightmostneighbor(G, cycle, visited):
    detection = False
    H = G.subgraph(cycle).copy()
    u = cycle[0]
    visited.append(u)
    idx = {k: v for v, k in enumerate(cycle)}
    rightmost_ngb = list(H[u])[0]
    for v in H[u]:
            if idx[v]>idx[rightmost_ngb]:
                rightmost_ngb = v

    if rightmost_ngb in visited:
            detection = True
    next_cycle = cycle[idx[rightmost_ngb]:]
    next_cycle.append(u)
    return next_cycle, detection
    
def extract_chordless_cycle(G, cycle):
    # We extract a chordless cycle from a given cycle.
    detection = False
    next_cycle = cycle
    visited = []
    while not detection:
        next_cycle, detection = shortcut_Rightmostneighbor(G, next_cycle, visited)
                        
    return next_cycle

def chordless_cycle_check(G, c):
    b = False
    H = G.subgraph(c).copy()
    if len(H.edges())==len(c):
        b = True

    return b

def find_Chordless_cycle(G):
    # Find chordless cycle in directed graph G
    cycle_edgelist = nx.find_cycle(G)
    cycle = [e[0] for e in cycle_edgelist] 

    if not chordless_cycle_check(G, cycle):
        c = extract_chordless_cycle(G, cycle)
    else:
        c = cycle
    return c

##########################################################################################
################ Maximum Antichain in a DiGraph #####################
##########################################################################################

def lp_DAG_marked(G, topo_order, marked):
    # Outputs path with most *unmarked* vertices in a DAG g.
    # g is assumed to be a networkx graph, and marked a set of vertices in g.
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Input graph is not a DAG!")

    if not set(marked).issubset(G.nodes()):
        raise ValueError("Marked vertices are not a subset of the graph's vertices!")
    
    dist = {} # largest number of unmarked vertices on a path ending at x.
    pred = {x: x for x in topo_order} # predecessors
    m = {} 
    for v in topo_order:
        if v in marked:
            m[v] = 0
            dist[v] = 0
        else:
            m[v] = 1
            dist[v] = 1
    
    for u in topo_order:
        for v in G.successors(u):
            if dist[v] < dist[u] + m[v]:
                dist[v] = dist[u] + m[v]
                pred[v] = u
    x = None
    y = max(dist, key = lambda x: (dist[x], x))
    lp = []
    while x != y:
        lp.append(y)
        x = y
        y = pred[y]
    lp.reverse()
    return lp

def naive_Initialization(g, topo_order):
    # The path cover simply consits of every singleton node.
    pc = [[v] for v in g.nodes()]
    sparse_g = g.copy()

    edge_counter = defaultdict(int) # counter the number of paths in the cover each edge is part of.
    node_counter = defaultdict(int) # counter the number of paths in the cover each node is part of.
    source_nodes = defaultdict(int) # counting how many paths start at a certain node
    sink_nodes = defaultdict(int) # counting how many paths end at a certain node
    for v in g.nodes():
        node_counter[v] = 1
        source_nodes[v] = 1
        sink_nodes[v] = 1

    return pc, sparse_g, edge_counter, node_counter, source_nodes, sink_nodes
def sparsify_and_GreedyPaths(g, topo_order):
    # Algorithm 4 in 'MPC: The Power of Parameterization'-paper.
    # Simultaneous sparsifies (deleting transitive edges) and finds a greedy path cover.
    # G needs to be a DAG.
    marked = set()
    pc = [] # The cover will be a list of lists.
    n = len(topo_order)
    edge_counter = defaultdict(int) # counter the number of paths in the cover each edge is part of.
    node_counter = defaultdict(int) # counter the number of paths in the cover each node is part of.
    source_nodes = defaultdict(int) # counting how many paths start at a certain node
    sink_nodes = defaultdict(int) # counting how many paths end at a certain node
    sparse_g = g.copy()
    
    while len(marked) != n:
        lp = lp_DAG_marked(sparse_g, topo_order, marked)
        marked.update(lp)
        pc.append(lp)
        # All edges in the lp to the edge counter.
        for a, b in zip(lp, lp[1:]):
            edge_counter[(a,b)] += 1
            node_counter[a] += 1
        node_counter[lp[-1]] += 1
        source_nodes[lp[0]] += 1
        sink_nodes[lp[-1]] += 1
        
        R = set()
        for v in lp:
            for u in list(sparse_g.predecessors(v)):
                if u in R and (u,v) not in edge_counter:
                    sparse_g.remove_edge(u,v)
                    #print(u,v)
                    #print('Above edge deleted!')
                R.add(u) 
    return pc, sparse_g, edge_counter, node_counter, source_nodes, sink_nodes

def create_MaxFlowInstance(n, topo_order, pc, sparse_g, edge_counter, node_counter, source_nodes, sink_nodes):
    # We create the max. flow instance from the folklore reduction, described in 'MPC: The Power of Parameterization'-paper. 
    # G_orig is the initial graph is the original umodified graph (needs to be a DAG), edge_counter_dict and node_counter are used to give an initial path cover (and hence initial flow).
    # NodeIDs are assumed as follows: the v_in nodes are (0,1,...n-1), the v_out nodes are (n,n+1,...2n-1).
    # We assume original G_orig nodes are also (0,1,2,...). The SCC algorithm does this.
    # pc, sparse_g, edge_counter, node_counter, source_nodes, sink_nodes = naive_Initialization(G_orig, topo_order)
    number_of_paths = len(pc)

    # Residual Graph
    G_r = nx.DiGraph()
    G_r.add_nodes_from(range(2*n))
    G_r.add_nodes_from(['s', 't'])

    # Every starting node of a path gets 1 flow from source.
    G_r.add_weighted_edges_from([('s', v, source_nodes[v]) for v in source_nodes])
    #G_r.add_weighted_edges_from([(v, 's', number_of_paths) for v in range(n)])

    # Every ending node of a path exists 1 flow to sink
    G_r.add_weighted_edges_from([(v+n, 't', sink_nodes[v]) for v in sink_nodes])
    #G_r.add_weighted_edges_from([('t',v, number_of_paths) for v in range(n,2*n)])

    # Edges between copies of the same nodes
    G_r.add_weighted_edges_from([(v, v+n, node_counter[v]-1) for v in node_counter if node_counter[v]>1])
    G_r.add_weighted_edges_from([(v+n, v, number_of_paths) for v in range(n)])

    # Edges between v_out and v_in nodes
    G_r.add_weighted_edges_from([(v+n,u, edge_counter[(v,u)]) for (v,u) in edge_counter])
    G_r.add_weighted_edges_from([(u,v+n, number_of_paths) for (v,u) in sparse_g.edges()])

    nx.set_edge_attributes(G_r, nx.get_edge_attributes(G_r, 'weight'), 'capacity')

    return G_r

def create_finalResidual(sparse_g, source_nodes, sink_nodes, edge_counter, node_counter, flow_dict, n):
    # Here we create the residual graph G_res corresponding to an optimal min_flow solution
    # To create the min_flow f_min = f_initial - f_maxflow, we update our flow with the max flow given by flow_dict.
    # Note that flow_dict is a dict of dicts (see nx.maximum_flow output).
    # To find the maximum antichain, we only care about reachability from 's' in G_res, so we dont put any capacities on the edges in G_res.

    # Initialize
    G_f = nx.DiGraph()
    G_f.add_nodes_from(range(2*n))
    G_f.add_nodes_from(['s', 't'])

    # Update flow leaving the source;
    for (u,f) in [(k,v) for k,v in flow_dict['s'].items() if v>0]:
        source_nodes[u] -= f
        node_counter[u] -= f

    # Update outgoing flow of v_in nodes. All are back-edges, so we add flow! only (v, v+n) is forward edge.
    for v in range(n):
        for (u,f) in [(k,y) for k,y in flow_dict[v].items() if y>0 and k != v+n]:
            edge_counter[(u % n,v)] += f
            node_counter[u % n] += f
            node_counter[v] += f
        if v+n in flow_dict[v]:
            # Forward edge
            node_counter[v] -= flow_dict[v][v+n]

    # Update outgoing flow of v_out nodes (also going to sink). All are forward edges, except (v,v-n) is backwards.
    for v in range(n,2*n):
        for (u,f) in [(k,y) for k,y in flow_dict[v].items() if y>0 and k != v-n]:
            node_counter[v % n] -= f
            if u != 't':
                edge_counter[(v % n,u)] -= f
                node_counter[u] -= f
            else:
                sink_nodes[v % n] -= f
        if v-n in flow_dict[v]:
            # Back edge
            node_counter[v % n] -= flow_dict[v][v-n]

    # Residual Graph
    G_f = nx.DiGraph()
    G_f.add_nodes_from(range(2*n))
    G_f.add_nodes_from(['s', 't'])

    # Every starting node of a path gets 1 flow from source.
    G_f.add_edges_from([('s', v) for v in source_nodes if source_nodes[v]>0])
    #G_r.add_weighted_edges_from([(v, 's', number_of_paths) for v in range(n)])

    # Every ending node of a path exists 1 flow to sink
    G_f.add_edges_from([(v+n, 't') for v in sink_nodes if sink_nodes[v]>0])
    #G_r.add_weighted_edges_from([('t',v, number_of_paths) for v in range(n,2*n)])

    # Edges between copies of the same nodes
    G_f.add_edges_from([(v, v+n) for v in node_counter if node_counter[v]>1])
    G_f.add_edges_from([(v+n, v) for v in range(n)])

    # Edges between v_out and v_in nodes
    G_f.add_edges_from([(v+n,u) for (v,u) in edge_counter if edge_counter[(v,u)]>0])
    G_f.add_edges_from([(u,v+n) for (v,u) in sparse_g.edges()])
    
    return G_f

def maxAntiChain(G_orig, topo_order):
    # We compute the maximum antichain, by looking at nodes reachable from 's' in the updated G_res.
    n = len(topo_order)
    pc, sparse_g, edge_counter, node_counter, source_nodes, sink_nodes = naive_Initialization(G_orig, topo_order)
    # pc, sparse_g, edge_counter, node_counter, source_nodes, sink_nodes = sparsify_and_GreedyPaths(G_orig, topo_order)
    G_r = create_MaxFlowInstance(n, topo_order, pc, sparse_g, edge_counter, node_counter, source_nodes, sink_nodes)
    flow_value, flow_dict = nx.maximum_flow(G_r, 's', 't')
    G_f = create_finalResidual(sparse_g, source_nodes, sink_nodes, edge_counter, node_counter, flow_dict, n)
    cands = set(nx.descendants(G_f,'s'))
    cands_n = set.intersection(cands,range(n))
    maxAntichain = set()
    for x in cands_n:
        if x+n not in cands:
            maxAntichain.add(x)
    maxAntichain = list(maxAntichain)     
    return maxAntichain

##########################################################################################

def test_running_times():
    # Test running time vs graph size
    hard_time_limit = 1000
    soft_time_limit = 100
    verbose = True
    results_file_path = "running_times_vs_graph_sizes.txt"
    log_file_path = "running_times_vs_graph_sizes.log"
    graph_sizes = range(500,12001,500)
    graphs = different_size_graphs(graph_sizes)
    algorithms = [
        (BAC1, "BAC"),
        (BAC2, "BCR"),
        (BAC3, "BCF"),
        (largest_dmin_next, "Greedy"),
        (randomk, "Rand."),
        (optimum_MaxIndSet_binsearch, "Opt."),
    ]
    k = 10

    with open(log_file_path, "w") as f:
        test_running_times_on_graphs(algorithms, graphs, k, graph_parameters=graph_sizes, 
                                     hard_time_limit=hard_time_limit, soft_time_limit=soft_time_limit, 
                                     results_file_path=results_file_path, log_file=f, verbose=verbose)

    # Test running time vs unique distances
    results_file_path = "running_times_vs_unique_distances.txt"
    log_file_path = "running_times_vs_unique_distances.log"
    max_weight_ranges = range(500,12001,500)
    n_vertices = 400
    graphs = different_weight_graphs(n_vertices, max_weight_ranges)
    algorithms = [
        (BAC1, "BAC"),
        (BAC2, "BCR"),
        (BAC3, "BCF"),
    ]

    with open(log_file_path, "w") as f:
        test_running_times_on_graphs(algorithms, graphs, k, graph_parameters=max_weight_ranges, 
                                     hard_time_limit=hard_time_limit, soft_time_limit=soft_time_limit, 
                                     results_file_path=results_file_path, log_file=f, verbose=verbose)


def main():
    datasets = [ft70, kro124, c_elegans, rbg323, wiki_vote_snap, airport, moreno_health, openflight, cora, bitcoin, gnutella_snap]
    algorithms = [
        (BAC1, "BAC"),
        (BAC2, "BCR"),
        (BAC3, "BCF"),
        (largest_dmin_next, "Greedy"),
        (randomk, "Rand."),
        #(optimum_MaxIndSet_binsearch, "Opt.")
    ]
    k = 10

    results_filename = "performance_results.txt"
    verbose = True

    with open(results_filename, "w+") as f:
        test_all_datasets(datasets, algorithms, k, results_file=f, verbose=verbose)

    # Test performance for different values of k
    datasets = [ft70, kro124, rbg323]
    for dataset in datasets:
        Performance_one_Dataset(dataset, krange=None)

    test_running_times()

if __name__ == '__main__':
    main()
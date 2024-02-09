# Read TSP data from TSPLIB

import tsplib95
import networkx as nx


ft70 = 'Z:/Desktop/test_clone/asymmetricmmd/Code_AMMD/datasets/ALL_atsp/ft70.atsp'
kro124 = 'Z:/Desktop/test_clone/asymmetricmmd/Code_AMMD/datasets/ALL_atsp/kro124p.atsp'
rbg323 = 'Z:/Desktop/test_clone/asymmetricmmd/Code_AMMD/datasets/ALL_atsp/rbg323.atsp'

path_name = 'Z:/Desktop/test_clone/asymmetricmmd/Code_AMMD/datasets/ALL_atsp/ftv44.atsp'
with open(kro124) as f:
    text = f.read()

problem = tsplib95.parse(text)

##print(problem.name)
##print(problem.comment)
##print(problem.dimension)
##print(problem.type)

# These distances do not satisfy triangle inequality per se!
# G = problem.get_graph()

# So we take metric closure.
#p = dict(nx.shortest_path_length(G, weight="weight"))

# p is a dict of dicts, we slightly need to redefine in order to read a networkx graph.
#p2 = dict()
#for k, v in p.items():
   # vnew = {x: {"weight": y} for x,y in v.items()}
    #p2[k] = vnew

#D = nx.DiGraph(p2)
#D.remove_edges_from(nx.selfloop_edges(D))

###Check if directed triangle-inequality is satisfied
##for u in D.nodes():
##    for v in D.nodes():
##        for w in D.nodes():
##            if w != v and w != u and v != u:
##                if D[u][v]['weight'] > D[u][w]['weight']+D[w][v]['weight']:
##                    print(u,v,w)
##                    print('Triangle inequality not satisfied!')
##                    break
##        else:
##            continue
##        break
##    else:
##        continue
##    break
    
def cluster_dmax(D, R):
    # The d_max clustering phase from our paper, with parameter R>0.
    # D_max is a NetworkX Graph, with edge attribute 'weights' the directed distances d_max (satisfying triangle ineq.)

    centers = set()
    marked = []
    check = True

    unmarked = set(D.nodes())
    while unmarked:
        c = unmarked.pop()
        centers.add(c)
        bad_v = set([v for v in unmarked if max(D[v][c]['weight'],D[c][v]['weight']) < R/3])
        unmarked.difference_update(bad_v)

    D_new = D.subgraph(centers) 
    return D_new

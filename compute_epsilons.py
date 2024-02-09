from AMMD_functions import *

datasets = [ft70, kro124, rbg323, airport, moreno_health, openflight, cora, bitcoin]
for dataset in datasets:
    D_w1 = read_graph_from_file(dataset, results_file=None, verbose=True)
    np.fill_diagonal(D_w1, 1)
    D_w2 = D_w1.transpose()
    m1 = np.divide(D_w1,D_w2)
    m2 = np.divide(D_w2,D_w1)
    epsis = np.maximum(m1,m2)
    
    eps_max = np.amax(epsis)
    eps_max = eps_max-1

    eps_avg = epsis.mean()
    eps_avg = eps_avg-1
    print(dataset, eps_avg)
    

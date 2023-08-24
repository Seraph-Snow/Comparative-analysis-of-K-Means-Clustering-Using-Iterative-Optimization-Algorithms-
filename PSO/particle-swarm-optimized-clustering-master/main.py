import numpy as np
import pandas as pd

from pso import ParticleSwarmOptimizedClustering
from utils import normalize
if __name__ == "__main__":
    data = pd.read_csv(r'C:\Users\dhira\PycharmProjects\pythonProject\seed.csv')
    #x = data.drop([7], axis=1)
    # print(x.head())
    data = data.values
    data = normalize(data)
    pso = ParticleSwarmOptimizedClustering(
        n_cluster=3, n_particles=10, data=data, hybrid=True)  #, max_iter=2000, print_debug=50)
    pso.run()

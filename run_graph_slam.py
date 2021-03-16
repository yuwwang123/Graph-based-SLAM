from graph_slam import  read_graph_g2o, run_graph_slam
import matplotlib.pyplot as plt




if __name__ == '__main__':
    filename = 'data/dlr.g2o'
    graph = read_graph_g2o(filename)
    run_graph_slam(graph, numIterations=100)

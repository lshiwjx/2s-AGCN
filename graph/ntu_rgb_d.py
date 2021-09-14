import sys

sys.path.extend(['../'])
from graph import tools

num_node = 25
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
                    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
                    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                    (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

new_self_index = [3, 8, 4, 23, 21, 16, 12, 19, 15]
new_self_links = [(i, i) for i in new_self_index]
newup_links_index = [(24, 4), (22, 4), (17, 9), (13, 5), (20, 24), (16, 22), (20, 17), (16, 13), (17, 24), (13, 22), (25, 22), (20, 16)]
newup_links = [(i - 1, j - 1) for (i, j) in newup_links_index]
newdown_links = [(j, i) for (i, j) in newup_links]
all_links = new_self_links + newup_links + newdown_links

class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward, all_links)
        elif labeling_mode == 'new_sep':
            A = tools.get_newsep_graph(num_node, self_link, inward, outward, new_self_links, newup_links, newdown_links)
        else:
            raise ValueError()
        return A


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    # os.environ['DISPLAY'] = 'localhost:11.0'
    A = Graph('spatial').get_adjacency_matrix()
    for i in A:
        plt.imshow(i, cmap='gray')
        plt.show()
    print(A)

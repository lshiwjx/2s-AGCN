import numpy as np


def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A


def normalize_digraph(A):  # 除以每列的和
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD


def get_spatial_graph(num_node, self_link, inward, outward, all_link):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    all = normalize_digraph(edge2mat(all_link, num_node))
    A = np.stack((I, In, Out, all))
    return A


def get_newsep_graph(num_node, self_link, inward, outward, new_self_links, newup_links, newdown_links):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    I_new = edge2mat(new_self_links, num_node)
    up = normalize_digraph(edge2mat(newup_links, num_node))
    down = normalize_digraph(edge2mat(newdown_links, num_node))
    A = np.stack((I, In, Out, I_new, up, down))
    return A
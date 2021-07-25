import networkx as nx
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import splu
import numpy as np
from tqdm import tqdm

def efficient_node_classification(G, alpha=0.99, max_iter=5, label_name="label"):
    """Node classification by Local and Global Consistency

    Parameters
    ----------
    G : NetworkX Graph
    alpha : float
        Clamping factor
    max_iter : int
        Maximum number of iterations allowed
    label_name : string
        Name of target labels to predict

    Returns
    ----------
    predicted : array, shape = [n_samples]
        Array of predicted labels

    Raises
    ------
    NetworkXError
        If no nodes on `G` has `label_name`.

    Examples
    --------
    >>> from networkx.algorithms import node_classification
    >>> G = nx.path_graph(4)
    >>> G.nodes[0]["label"] = "A"
    >>> G.nodes[3]["label"] = "B"
    >>> G.nodes(data=True)
    NodeDataView({0: {'label': 'A'}, 1: {}, 2: {}, 3: {'label': 'B'}})
    >>> G.edges()
    EdgeView([(0, 1), (1, 2), (2, 3)])
    >>> predicted = node_classification.local_and_global_consistency(G)
    >>> predicted
    ['A', 'A', 'B', 'B']


    References
    ----------
    Zhou, D., Bousquet, O., Lal, T. N., Weston, J., & Schölkopf, B. (2004).
    Learning with local and global consistency.
    Advances in neural information processing systems, 16(16), 321-328.
    """
    try:
        import numpy as np
    except ImportError as e:
        raise ImportError(
            "local_and_global_consistency() requires numpy: ", "http://numpy.org/ "
        ) from e
    try:
        from scipy import sparse
    except ImportError as e:
        raise ImportError(
            "local_and_global_consistensy() requires scipy: ", "http://scipy.org/ "
        ) from e

    def _build_propagation_matrix(X, labels, alpha):
        """Build propagation matrix of Local and global consistency

        Parameters
        ----------
        X : scipy sparse matrix, shape = [n_samples, n_samples]
            Adjacency matrix
        labels : array, shape = [n_samples, 2]
            Array of pairs of node id and label id
        alpha : float
            Clamping factor

        Returns
        ----------
        S : scipy sparse matrix, shape = [n_samples, n_samples]
            Propagation matrix

        """
        degrees = X.sum(axis=0).A[0]
        degrees[degrees == 0] = 1  # Avoid division by 0
        D2 = np.sqrt(sparse.diags((1.0 / degrees), offsets=0))
        S = alpha * D2.dot(X).dot(D2)
        return S

    def _build_base_matrix(X, labels, alpha, n_classes):
        """Build base matrix of Local and global consistency

        Parameters
        ----------
        X : scipy sparse matrix, shape = [n_samples, n_samples]
            Adjacency matrix
        labels : array, shape = [n_samples, 2]
            Array of pairs of node id and label id
        alpha : float
            Clamping factor
        n_classes : integer
            The number of classes (distinct labels) on the input graph

        Returns
        ----------
        B : array, shape = [n_samples, n_classes]
            Base matrix
        """

        n_samples = X.shape[0]
        #B = np.zeros((n_samples, n_classes))
        B = csr_matrix((n_samples, n_classes))
        B[labels[:, 0], labels[:, 1]] = 1 - alpha
        return B

    def _propagate(P, F, B):
        """Propagate labels by one step

        Parameters
        ----------
        P : scipy sparse matrix, shape = [n_samples, n_samples]
            Propagation matrix
        F : numpy array, shape = [n_samples, n_classes]
            Label matrix
        B : numpy array, shape = [n_samples, n_classes]
            Base matrix

        Returns
        ----------
        F_new : array, shape = [n_samples, n_classes]
            Label matrix
        """
        F_new = P.dot(F) + B
        return F_new

    def _get_label_info(G, label_name):
        """Get and return information of labels from the input graph

        Parameters
        ----------
        G : Network X graph
        label_name : string
            Name of the target label

        Returns
        ----------
        labels : numpy array, shape = [n_labeled_samples, 2]
            Array of pairs of labeled node ID and label ID
        label_dict : numpy array, shape = [n_classes]
            Array of labels
            i-th element contains the label corresponding label ID `i`
        """
        import numpy as np

        labels = []
        label_to_id = {}
        lid = 0
        for i, n in enumerate(G.nodes(data=True)):
            if label_name in n[1]:
                label = n[1][label_name]
                if label not in label_to_id:
                    label_to_id[label] = lid
                    lid += 1
                labels.append([i, label_to_id[label]])
        labels = np.array(labels)
        label_dict = np.array(
            [label for label, _ in sorted(label_to_id.items(), key=lambda x: x[1])]
        )
        return (labels, label_dict)

    def _init_label_matrix(n_samples, n_classes):
        """Create and return zero matrix

        Parameters
        ----------
        n_samples : integer
            The number of nodes (samples) on the input graph
        n_classes : integer
            The number of classes (distinct labels) on the input graph

        Returns
        ----------
        F : numpy array, shape = [n_samples, n_classes]
            Label matrix
        """
        import numpy as np

        F = np.zeros((n_samples, n_classes))
        return F

    def _predict(F, label_dict):
        """Predict labels by learnt label matrix

        Parameters
        ----------
        F : numpy array, shape = [n_samples, n_classes]
            Learnt (resulting) label matrix
        label_dict : numpy array, shape = [n_classes]
            Array of labels
            i-th element contains the label corresponding label ID `i`

        Returns
        ----------
        predicted : numpy array, shape = [n_samples]
            Array of predicted labels
        """
        import numpy as np

        predicted_label_ids = np.argmax(F, axis=1)
        predicted_label_ids = [item.item() for sublist in predicted_label_ids for item in sublist]
        predicted = label_dict[predicted_label_ids].tolist()
        return predicted

    X = nx.to_scipy_sparse_matrix(G)  # adjacency matrix
    labels, label_dict = _get_label_info(G, label_name)

    if labels.shape[0] == 0:
        raise nx.NetworkXError(
            "No node on the input graph is labeled by '" + label_name + "'."
        )

    n_samples = X.shape[0]
    n_classes = label_dict.shape[0]
    F = _init_label_matrix(n_samples, n_classes)
    #F = csr_matrix((n_samples, n_classes))

    P = _build_propagation_matrix(X, labels, alpha)
    B = _build_base_matrix(X, labels, alpha, n_classes)

    remaining_iter = max_iter
    while remaining_iter > 0:
        print(remaining_iter)
        F = _propagate(P, F, B)
        remaining_iter -= 1

    predicted = _predict(F, label_dict)

    return predicted


def custom_label_propagation(net, label_dict):
    labeled = list(label_dict.keys())
    unlabeled = [str(i) for i in list(range(len(net.nodes)))]
    unlabeled = list(set(unlabeled) - set(labeled))
    # get Laplacian and its submatrices
    L = nx.normalized_laplacian_matrix(net, nodelist=labeled + unlabeled)
    first_unlabeled_index = len(labeled)
    L_uu = L[first_unlabeled_index:, first_unlabeled_index:]
    L_us = L[first_unlabeled_index:, :first_unlabeled_index]
    # get labeled one-hot encoding
    label_arr = np.array(list(label_dict.values()))
    y_l = lil_matrix((label_arr.size, label_arr.max()))
    y_l[np.arange(label_arr.size), label_arr - 1] = 1

    tmp = L_us * y_l

    unlabeled_dict = {}
    L_i = splu(-L_uu.tocsc())
    for k, unlabeled_node in tqdm(enumerate(unlabeled)):
        b = np.zeros((len(unlabeled),))
        b[k] = 1
        inv_row = L_i.solve(b)
        max_val = 0
        for s in range(label_arr.max()):
            score = inv_row * tmp[:, s]
            if score > max_val:
                unlabeled_dict[unlabeled_node] = s + 1
                max_val = score
    return unlabeled_dict

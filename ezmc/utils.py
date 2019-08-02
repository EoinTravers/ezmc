
def expand_1d_chain(chain, by):
    i = len(chain)
    new_chain = np.zeros(i + by)
    new_chain[:i] = chain
    return new_chain


def uniform_within_bounds(bounds):
    bounds = np.array(bounds)
    return np.random.uniform(bounds[:, 0], bounds[:, 1])

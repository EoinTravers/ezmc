
    # def update_results(self, burn_in=100, thin=4):
    #     ix = self.iterations
    #     samples = pd.DataFrame(self.parameter_chain[burn_in:ix:thin], columns=self.par_names)
    #     samples['ll'] = self.ll_chain[burn_in:ix:thin]
    #     self.results = samples

    # def get_results(self, burn_in=100, thin=4):
    #     self.update_results(burn_in=burn_in, thin=thin)
    #     return self.results

    # def plot_chains(self, par_names=None, thin=4):
    #     if par_names is None:
    #         par_names = self.par_names
    #     self.update_results(thin=thin)
    #     n = len(par_names)
    #     for i in range(n):
    #         plt.subplot(1, n, i+1)
    #         plt.title(self.par_names[i])
    #         x = parameter_chain[:ix:thin, i]
    #         plt.plot(x)


# import numba as nb
# @nb.njit
# def np_apply_along_axis(func1d, axis, arr):
#     assert arr.ndim == 2
#     assert axis in [0, 1]
#     if axis == 0:
#         result = np.empty(arr.shape[1])
#         for i in range(len(result)):
#             result[i] = func1d(arr[:, i])
#     else:
#         result = np.empty(arr.shape[0])
#         for i in range(len(result)):
#             result[i] = func1d(arr[i, :])
#     return result

# @nb.njit
# def np_mean(array, axis):
#     return np_apply_along_axis(np.mean, axis, array)

# @nb.njit
# def np_std(array, axis):
#     return np_apply_along_axis(np.std, axis, array)

from sys import stdout
from time import sleep
def foo(i):
    txt = ('%i' % i) * (50-i)
    # txt = '\r' + txt
    txt = txt + ' ' * 80 + '\r'
    stdout.write(txt)
    stdout.flush()
    sleep(.1)

for i in range(50):
    foo(i)

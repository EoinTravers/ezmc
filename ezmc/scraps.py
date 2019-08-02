
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

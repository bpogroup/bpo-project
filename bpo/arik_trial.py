import numpy as np
import random
from enum import Enum
from scipy import stats
import networkx as nx
# test


class DistributionType(Enum):
    normal = 1
    exponential = 2
    uniform = 3
    scheduled = 4
    immediate = 5
    kernel_density_estimate = 6
    deterministic = 7
    empirical = 8
    lognormal = 9
    erlang = 10



class Distribution:
    """
    a class to capture supported distributions that can be fit to data
    """

    def __init__(self, dist_type=DistributionType.exponential, **kwargs):
        fit = kwargs.get('fit', False)
        if not fit:
            if dist_type == DistributionType.normal:
                self._param = [kwargs.get('mean', 5), kwargs.get('sd', 1)]

                self._dist = stats.norm(loc = self._param[0], scale = self._param[1])

            elif dist_type == DistributionType.exponential:
                self._param = [kwargs.get('rate', 1)]
            elif dist_type == DistributionType.uniform:
                self._param = [kwargs.get('low', 0), kwargs.get('high', 1)]
            elif dist_type == DistributionType.lognormal:
                self._param = [kwargs.get('mu', 0), kwargs.get('sigma', 1)]
                self._dist = stats.lognorm(s = self._param[1], scale = np.exp(self._param[0]))
            elif dist_type == DistributionType.scheduled:
                self._param = [kwargs.get('mintime', 0)]
            elif dist_type == DistributionType.deterministic:
                self._param = [kwargs.get('time', 0)]

            elif dist_type == DistributionType.erlang:
                self._param = [kwargs.get('K'), kwargs.get('rate')]
            else:  # immediate transition
                self._param = [0]
                dist_type = DistributionType.immediate
        else:  # fit distributions to data
            # Assuming that the values are stored in a 1-d vector called "values"

            if dist_type == DistributionType.empirical:
                self._param = [kwargs.get('values', [0, 0, 1])]
            elif dist_type==DistributionType.kernel_density_estimate:
                self._param = [kwargs.get('values', [0, 0, 1]), kwargs.get('bw_', 'scott'),
                               kwargs.get('conditional', False), kwargs.get('cond_', [])]
                self._dist = stats.gaussian_kde(self._param[0],bw_method =self._param[1] )
                dist_type = DistributionType.kernel_density_estimate

                #self._dist = stats.erlang(scale = 1/self._param[1], shape = self._param[0])


                self.erlang_fit_mle = stats.erlang.fit(data=self._param[0], fa = self._param[1], floc=0)
                dist_type = DistributionType.erlang



        self._name = dist_type
        self.type = dist_type
        # print "creating {} distribution with {} values".format(dist_type, self._param)



    def __repr__(self):
        return "{} distribution (params:{})".format(self._name, self._param)

    def sample(self, **kwargs):
        """
        Samples a random value from the distribution that is used.
        :return: a sample from the distribution
        """
        strat = kwargs.get('strat', [])

        if self._name == DistributionType.normal:
            return random.gauss(self._param[0], self._param[1])
        elif self._name == DistributionType.exponential:
            return random.expovariate(self._param[0])
        elif self._name == DistributionType.lognormal:
            #return random.lognormvariate(self._param[0], self._param[1])
            return self._dist.rvs(1)[0]
        elif self._name == DistributionType.erlang:
            #return self._dist.rvs(1)[0]
            return sum([random.expovariate(self._param[1]) for k in range(self._param[0])])

        elif self._name == DistributionType.immediate:
            return 0
        elif self._name == DistributionType.scheduled:
            current_time = kwargs.get('time', 0)
            return max(self._param[0], current_time) - current_time
        elif self._name == DistributionType.kernel_density_estimate:
            x =self._dist.resample(size=1)[0][0]
            #if x<0:
            #    print('Negative x')
            return x #max(x,0)#abs(x)

        elif self._name == DistributionType.deterministic:
            return self._param[0]
        elif self._name == DistributionType.empirical:
            # draw one of the past samples uniformly
            if len(self._param[0])>0:
                return self._param[0][random.randint(0, len(self._param[0]) - 1)]
            else:
                return 0
        else:  # assume uniform distribution
            return random.uniform(self._param[0], self._param[1])

    def get_mean(self):
        if self._name == DistributionType.kernel_density_estimate or self._name == DistributionType.empirical:
            return np.mean(self._param)
        elif self._name == DistributionType.exponential:
            return 1 / self._param[0]
        elif self._name == DistributionType.deterministic:
            return self._param[0]
        elif self._name == DistributionType.immediate:
            return 0
        else:
            print("please implement for the Mean {} - debug me!".format(self._name))
            return -1

if __name__ == "__main__":



    #problem setting:

    print('start')
    #number of jobs:
    N = 1000
    J=100

    #activities:

    activity_names = [0, 1, 2]
    probabilities = [0.5, 0.4, 0.1]

    #we will use Erlang distribution per activity. The mean of the 3 activities is the same (K*mean), but the variance is different (K*mean^2)
    mean = [1 / 10, 1 / 5, 1]
    steps = [10, 5, 1]
    stdev = [steps[i]*(mean[i]**2) for i in range(len(mean))]
    #generate jobs with one activity per case (for now):

    activities_training = np.random.choice(activity_names, size=N, p=probabilities)

    #print(activities_training[0:10])



    #generate N erlang durations, same mean, diff variance per activity


    processing_times = [Distribution(dist_type=DistributionType.erlang, K=steps[activities_training[i]], rate=1/mean[activities_training[i]]).sample() for i in range(N)]

    #print(processing_times[0:10])
    #print(np.mean(processing_times), np.std(processing_times))

    est_mean = np.mean(processing_times)
    est_stdev = np.std(processing_times)


    activities_test = np.random.choice(activity_names, size=J, p=probabilities)

    E = 10

    for experiment in range(E):
        settings = [1,2]
        for setting in settings:

            #setting = 1


            if setting == 1:
                # setting 1:
                # schedule according to mu (predicted value)

                sequence = [i for i in range(len(activities_test))]  #any arbitrary sequence works as all of them have the same mean

            elif setting==2:
                #setting 2:
                #schedule smallest variance first
                variance_order = [(i, stdev[activities_test[i]]) for i in range(len(activities_test))]
                sorted_variance = sorted(variance_order, key=lambda x: x[1])
                sequence = [el[0] for el in sorted_variance]

            schedule = []
            for s in sequence:
                schedule.append(mean[activities_test[s]]*steps[activities_test[s]])


            #estimate cost based on R runs:
            R = 10000
            est_W = []
            est_I = []
            for r in range(R):
                processing_times = [Distribution(dist_type=DistributionType.erlang, K=steps[activities_test[sequence[i]]],
                                             rate= 1 / mean[activities_test[sequence[i]]]).sample() for i in range(J)]
                W = [0]
                I = [0]

                for i in range(J-1):
                    W.append(max(0, W[len(W)-1] + processing_times[i]-schedule[i]))
                    I.append(max(0, -(W[len(W)-1] + processing_times[i]-schedule[i])))
                est_W.append(np.mean(W))
                est_I.append(np.mean(I))
            omega = 0.5
            est_c = omega*np.mean(est_W) + (1-omega)*(np.mean(est_I))

            print('Estimated cost for setting', setting, 'experiment number', experiment, 'is', est_c)



import sys
import copy
import numpy as np
from numpy import linalg as la

class draftPositionMarkov:
    def __init__(self,
                 updown={3: 0.6, -2: 0.4},
                 draft_limits=[1,30],
                 picks_options=[1,5]):
        self.updown = self.renormalize_dict(updown)
        self.draft_limits = draft_limits
        self.draft_range = 1+draft_limits[1]-draft_limits[0]

        self.picks_options = np.array(picks_options, dtype='i4')

        self.picks_file = 'picks.csv'
        self.picks_matrix = self.make_picks_matrix()
        self.rank_matrix = self.make_season_finish_matrix()
        self.transition_matrix = self.make_transition_matrix()

        self.current_matrix = copy.copy(self.transition_matrix)

    def __str__(self):
        return str(self.picks_matrix)

    def make_picks_matrix(self):
        mm = self.initialize_matrix()

        lottery = np.genfromtxt(self.picks_file,
                             delimiter=',',
                             skip_header=0)[0:,0:]

        lottery = lottery.transpose()
        nrow_lottery, ncol_lottery = np.shape(lottery)
        assert nrow_lottery == ncol_lottery
        n_lottery = nrow_lottery

        nrow, ncol = np.shape(mm)
        for irow in range(nrow):
            for icol in range(ncol):
                if irow<n_lottery and icol<n_lottery:
                    mm[irow, icol] = lottery[irow, icol]
                else:
                    mm[irow, icol] = 1 if irow==icol else 0

        return mm

    def initialize_matrix(self):
        nx = self.draft_range
        nx += len(self.picks_options)
        return np.zeros((nx, nx))

    def rank_clamp(self, x):
        return min(max(x, self.draft_limits[0]), self.draft_limits[1])

    def index_to_rank(self, idx):
        return self.rank_clamp(idx+1)

    def rank_to_index(self, rank):
        return self.rank_clamp(rank)-1

    def make_season_finish_matrix(self):
        mm = self.initialize_matrix()
        for idx_old in range(self.draft_range):
            rank_old = self.index_to_rank(idx_old)
            for delta_value, delta_prob in self.updown.items():
                rank_new = self.rank_clamp(rank_old + delta_value)
                idx_new = self.rank_to_index(rank_new)
                #print idx_old, rank_old, idx_new, rank_new, delta_value, delta_prob
                mm[idx_new, idx_old] += delta_prob
        return mm

    def make_transition_matrix(self):
        mm = self.initialize_matrix()
        nopt = len(self.picks_options)

        # first, the probabilities to land a draft pick we are
        # interested in
        for i in range(nopt):
            idx = nopt - i
            draft_idx = self.rank_to_index(self.picks_options[i])
            mm[-idx,:] = self.picks_matrix[draft_idx,:]


        # now the others
        delta = mm[-nopt:,:].sum(0)
        mm[0:self.draft_range] = self.rank_matrix[0:self.draft_range]*(1-delta)

        # one for the endpoint states
        for i in range(nopt):
            idx = self.draft_range + i
            mm[idx, idx] = 1
        return mm

    def convergence_stat(self, matrix1, matrix2):
        return ((matrix1-matrix2)**2).sum()

    def converge(self, tol=1e-12, maxiter = 100):
        '''
        :return: converged transition matrix

        uses the method of exponentiation by squaring
        '''

        u = la.matrix_power(self.transition_matrix, 2**20)
        u[u<1e-6] = 0
        return u

    def pretty_print(self, ww):
        ofp = sys.stdout
        ofp.write('start ')
        for i in self.picks_options:
            ofp.write('%2d ' % i),
        ofp.write('\n')
        ofp.write('************************\n')
        for i in range(self.draft_range):
            tmp = ww[-(len(self.picks_options)):,i]*100
            ofp.write('%02d ' % (i+1))
            for t in tmp:
                ofp.write('%5.1f ' % t)
            ofp.write('\n')

    def renormalize_dict(self, dict):
        dict_sum = np.sum([v for k, v in dict.items()])
        tmp = {}
        for k, v in dict.items():
            tmp[k] = v/(1.0*dict_sum)
        if abs(dict_sum-1)>1e-3:
            sys.stdout.write('renorm {} {}\n'.format(tmp, dict_sum))
        return tmp

if __name__=='__main__':
    updown={3: 0.6, -2: 0.4}
    picks_options=[1,5]

    for ia, a in enumerate(sys.argv):

        if a=='-picks':
            tmp = []
            for t in sys.argv[ia+1].split(','):
                tmp.append(t)
            picks_options = tmp[:]
        if a=='-updown':
            tmp = {}
            for i, t in enumerate(sys.argv[ia+1].split(',')):
                idx = i%2
                if idx==0:
                    k = int(t)
                else:
                    v = float(t)
                    tmp[k] = v

            updown = copy.copy(tmp)

    dpm = draftPositionMarkov(updown=updown, picks_options=picks_options)
    ww = dpm.converge()
    dpm.pretty_print(ww)
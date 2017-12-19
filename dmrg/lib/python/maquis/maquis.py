#!/usr/bin/env python

import libmaquis
import numpy as np
import tempfile
import shutil
import os.path

class DummyCSF:

    def __init__(self):
        self.total_ndet = 0
        self.total_nCSF = 0


class DMRGBox:

    def __init__(self, options, Hnp, Inp):

        self.options = { 
                        "nsweeps" : 8
                        , "max_bond_dimension" : 400
                        , "init_bond_dimension" : 400
                          #
                        , "init_state" : 'default'
                        , "donotsave" : 0
                        , "chkp_each" : 100
                        , "measure_each" : 1
                        , "chkpfile" : 'chkp_dummy.h5'
                        , "resultfile" : 'result_dummy.h5'
                          #"storagedir" : '/scratch/sebkelle/boundaries'
                          #
                        , "ietl_jcd_maxiter" : 9
                        , "ietl_jcd_tol" : 1e-7
                        , "truncation_initial" : 1e-7
                        , "truncation_final" : 1e-7
                        , "symmetry" : 'su2u1'
                          #
                        , "integral_cutoff" : 1e-20
                        , "conv_thresh" : 1e-6
                          #
                        , "MEASURE[ChemEntropy]" : 1
                        , "MEASURE[1rdm]" : 1
                        , "MEASURE[2rdm]" : 1
                        , "MEASURE[Energy]" : 1
                       }

        self.options.update(options)

        if Hnp is not None and Inp is not None:
            Ls = Inp.shape
            assert(min(Ls) == max(Ls))
            self.options["L"] = Ls[0]

            Ecore = 0.0;
            if options.has_key("Ecore"):
                Ecore = options["Ecore"]

            self.options["integrals"] = dmrg.pack_integrals(Hnp, Inp, self.options["L"], Ecore)

        self.solvers = {}

        self.tempdir = tempfile.mkdtemp(prefix='dmrg_', dir='.')

    def __del__(self):
        if os.path.exists(self.tempdir):
            if 'keep_files' in self.options.keys():
                if self.options['keep_files'] == False:
                    shutil.rmtree(self.tempdir)
            else:
                shutil.rmtree(self.tempdir)

    def compute_states(self, S_ind, S_nstate):
        for S,N in zip(S_ind, S_nstate):
            self.options['spin'] = S
            ortho_states = ''
            self.solvers[S] = {}
            for n in range(N):
                self.options['chkpfile'] = os.path.join(self.tempdir, "state_S" + str(S) + "_" + str(n))
                self.options['resultfile'] = os.path.join(self.tempdir, "results_S" + str(S) + "_" + str(n))
                self.options['n_ortho_states'] = n
                self.options['ortho_states'] = ortho_states
                self.solvers[S][n] = libmaquis.interface(self.options)
                self.solvers[S][n].optimize()
                ortho_states += self.options['chkpfile'] + ','

    def energy(self, S, state):
        self.solvers[S][state].measure("Energy")
        return self.solvers[S][state].getObservable("Energy")

    def opdm(self, S, state, evecs2, total):
        """calculate the 1-body reduced density matrix"""

        self.solvers[S][state].measure("oneptdm")
        r1 = self.solvers[S][state].getObservable("oneptdm")
        r1l = self.solvers[S][state].getLabels("oneptdm")
        rdm1 = DMRGBox.expand_1rdm(r1, r1l, self.options["L"])
        return rdm1

    def tpdm(self, S, state, evecs2, symmetrize):
        """calculate the 2-body reduced density matrix"""

        self.solvers[S][state].measure("twoptdm")
        r2 = self.solvers[S][state].getObservable("twoptdm")
        r2l = self.solvers[S][state].getLabels("twoptdm")
        rdm2 = DMRGBox.expand_2rdm(r2, r2l, self.options["L"])
        return rdm2


    # unimplemented CASBox functionality
    def CSF_basis(self, S):
        return DummyCSF()

    def amplitude_string(self, a,b,c,d,e):
        return "Amplitudes not implemented"

    def metric_det(m):
        raise ValueError("overlaps not implemented")


    def write_fcidump(self):

        nintegrals = len(self.options["integrals"]) / 24
        integrals = np.fromstring(self.options["integrals"], count=nintegrals, dtype=np.float64)
        indices = np.fromstring(self.options["integrals"][8*nintegrals:], count=4*nintegrals, dtype=np.int32)

        f = open("FCIDUMP", 'w')
        for i in range(nintegrals):
            f.write('  % 20.14e' % integrals[i] + '  ' + '  '.join([str(x) for x in [indices[4*i], indices[4*i+1], indices[4*i+2], indices[4*i+3]]]) + '\n')
        f.close()

        ifile = open("di", 'w')
        for k,v in self.options.items():
            if k != "integrals":
                ifile.write(str(k) + ' = ' + str(v) + '\n')
        ifile.close()

    @staticmethod
    def expand_1rdm(rdm, labels, L):
        odm = np.zeros([L,L])

        for lab, val in zip(labels, rdm):
            i = lab[0]
            j = lab[1]

            odm[i,j] = val
            odm[j,i] = val

        return odm

    @staticmethod
    def expand_2rdm(rdm, labels, L):
        odm = np.zeros([L,L,L,L])

        for lab, val_raw in zip(labels, rdm):
            i = lab[0]
            j = lab[1]
            k = lab[2]
            l = lab[3]

            # lightspeed doesnt use canonical ordering
            j,l = l,j

            val = 0.5 * val_raw
            odm[i,j,k,l] = val

            if l != k or i != j:
                odm[j,i,l,k] = val

            if min(i,j) != min(l,k) or max(i,j) != max(l,k):
                odm[k,l,i,j] = val
                if l != k or i != j:
                    odm[l,k,j,i] = val

        return odm


    @staticmethod
    def pack_integrals(Hnp, Inp, L, Ecore):
        integrals = []
        indices = []

        for t in range(L):
            for u in range(t+1):
                integrals.append(Hnp[t,u])
                indices.extend([t+1,u+1,0,0])
        for t in range(L):
            for u in range(t+1):
                for v in range(L):
                    for w in range(v+1):
                        if (t > v or (t==v and u>=w)):
                            integrals.append(Inp[t,u,v,w])
                            indices.extend([t+1,u+1,v+1,w+1])
        integrals.append(Ecore)
        indices.extend([0,0,0,0])

        return np.array(integrals, dtype=np.float64).tobytes() + np.array(indices, dtype=np.int32).tobytes()

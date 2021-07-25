#cython: language_level=3, wraparound=False, boundscheck=False, initializedcheck=False, nonecheck=False, cdivision=True
import numpy as np
import time
import cython
cimport numpy as np
from libc.math cimport exp, log

# some declarations for cython purposes

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

cdef extern from "math.h" nogil:
    float exp(float x)
    float log(float x)



# takes encounters from mobility simulation
# encounters have to be of the following format:
# Example: List of length n (for n people)
# -------- Each entry has a flexible length m (one for each encounter)
# -------- Each encounter has to have at least three attributes: 0 - encounter id, 1 - intensity, 2 - duration, (3 - place, not needed right now)
# -------> N x M_max x 4 matrix will be created in this class - might face some memory issues with very large simulations

cdef class DiseaseSpread:

    cdef dict infections, isolation, quarantine
    cdef set immune, cured
    cdef np.ndarray F, p
    cdef np.float32_t s, alpha, beta, offset
    cdef int total_infections
    cdef Py_ssize_t m
    cdef np.ndarray encounter_matrix, infected_matrix
    cdef np.float32_t float_one, float_zero


    def __init__(self, dict infections, dict isolation, dict quarantine, set immune, set cured):



        # dict of infected people with their according disease duration
        self.infections = infections

        # set of nodes currently in isolation due to being tested postive
        self.isolation = isolation

        # set of nodes currently in quarantine due to a contact person being tested positive
        self.quarantine = quarantine

        # set of vaccinated people
        self.immune = immune

        # set of cured people (might unify this with immune)
        self.cured = cured


        # parameters necessary for backpropagating
        # p - probability vector with all nodes getting assigned a probability of getting infected on a given day
        # F - result of
        cdef np.ndarray[ndim=1, dtype=DTYPE_t] p
        self.p = None
        cdef np.ndarray[ndim=2, dtype=DTYPE_t] F
        self.F = None
        self.s = 0
        self.m = 0
        self.float_zero = np.float32(0)
        self.float_one = np.float32(1)
        self.alpha = np.float32(0.3)
        self.beta = np.float32(0.5)
        self.offset = np.float32(10)
        self.total_infections = 0
        self.encounter_matrix = None
        self.infected_matrix = None


    cpdef void update_encounters(self, list encounters):
        #print(encounters[0])
        self.get_encounter_matrix(encounters)

    cdef np.uint8_t check_encounter_for_infection(self, int node_id):
        return self.infections[node_id][0]

    cdef np.uint8_t check_self_for_immunity(self, int node_id):
        return node_id in self.immune

    @cython.nonecheck(False)
    cdef Py_ssize_t get_maximum_encounter_dimension(self, list encounters):
        cpdef Py_ssize_t max_dim = 0
        cpdef Py_ssize_t enc_length = len(encounters)
        cpdef Py_ssize_t tmp
        for i in range(enc_length):
            tmp = len(encounters[i])
            if  tmp > max_dim:
                max_dim = tmp
        return max_dim

    cpdef int get_total_infections(self):
        return self.total_infections

    cpdef Py_ssize_t return_maxdim(self):
        return self.m




    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef np.ndarray get_encounter_matrix(self, list encounters):
        cpdef Py_ssize_t max_dim = self.get_maximum_encounter_dimension(encounters)
        cpdef Py_ssize_t enc_length = len(encounters)
        cpdef Py_ssize_t node_length = 0
        cpdef Py_ssize_t node_index, encounter_index
        self.m = max_dim
        self.encounter_matrix = np.zeros(shape=(enc_length, max_dim, 2), dtype=np.float32)
        self.infected_matrix = np.zeros(shape=(enc_length, max_dim, 2), dtype=np.uint8)
        cdef float[:, :, :] values
        cdef np.uint8_t[:, :, :] bool_values
        for node_index in range(enc_length):
            node_length = len(encounters[node_index])
            for encounter_index in range(node_length):
                self.encounter_matrix[node_index][encounter_index][0] = encounters[node_index][encounter_index][1]
                self.encounter_matrix[node_index][encounter_index][1] = encounters[node_index][encounter_index][2]
                self.infected_matrix[node_index][encounter_index][0] = self.check_encounter_for_infection(encounters[node_index][encounter_index][0])
                self.infected_matrix[node_index][encounter_index][1] = True




    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.nonecheck(False)
    @cython.cdivision(True)
    cdef np.float32_t sigmoid(self, np.float32_t x, np.float32_t offset):
        return self.float_one/(self.float_one + exp(-x + offset))



    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef step(self):

        # variable declarations for cython, CRUCIAL for speedup
        cdef Py_ssize_t xmax = len(self.encounter_matrix)
        cdef Py_ssize_t ymax = self.return_maxdim()
        cdef np.float32_t alpha = self.alpha
        cdef np.float32_t beta = self.beta
        cdef np.float32_t offset = self.offset
        cdef np.float32_t maximum_sigmoid_value = np.float32(0.999999)
        cdef np.float32_t[:, :] intensities = self.encounter_matrix[:, :, 0]
        cdef np.float32_t[:, :] durations = self.encounter_matrix[:, :, 1]
        cdef np.uint8_t[:, :] infected = self.infected_matrix[:, :, 0]
        cdef np.uint8_t[:, :] active = self.infected_matrix[:, :, 1]
        self.F = np.zeros([xmax, ymax], dtype=DTYPE)
        self.p = np.zeros([xmax], dtype=DTYPE)
        cdef np.float32_t[:, :] F = self.F
        cdef np.float32_t[:] p = self.p
        cdef int infection_status = 0
        cdef np.float32_t infection_probability = np.float32(0.0)
        cdef np.float32_t tmp, tmp2
        cdef Py_ssize_t row, y
        print('Simulating disease spread...')

        # unifying intensity matrix, duration matrix with their appropriate weights
        for row in range(xmax):
            p[row] = self.float_zero
            for y in range(ymax):
                infection_status = infected[row, y]
                tmp = infection_status * self.sigmoid(alpha * intensities[row, y] + beta * durations[row, y], offset)
                if tmp < maximum_sigmoid_value:
                    F[row, y] = tmp

                else:
                    F[row, y] = maximum_sigmoid_value
                p[row] = p[row] + log(self.float_one - F[row, y]) * active[row, y]
            p[row] = exp(p[row])


        # sum of all probablities, idk why i included this, might be useful at some point
        self.s = xmax - self.fast_sum(self.p)

        # deciding for each node whether they got infected or not
        for x in range(xmax):
            if not self.infections[x][0]:
                infection_probability = p[x]
                if np.random.rand() < (1 - infection_probability):
                    self.infections[x][0] = True
        print('Disease simulation step complete.')



    cdef void write_to_matrix(self, DTYPE_t[:, :] arr, DTYPE_t val, Py_ssize_t x, Py_ssize_t y):
        arr[x, y] = val



    cdef void write_to_vector(self, DTYPE_t[:] vec, DTYPE_t val, Py_ssize_t x):
        #print(x, val, vec)
        vec[x] = val


    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef np.float32_t fast_sum(self, np.float32_t[:] arr):
        cdef np.float32_t sum = 0
        cdef Py_ssize_t index_x
        cdef Py_ssize_t length = len(arr)
        for index_x in range(length):
            sum = sum + arr[index_x]
        return sum




    @cython.boundscheck(False)
    @cython.wraparound(False)
    # Used to update the current active cases count, but will later be extended to update isolations and quarantine measures
    cpdef update_node_status(self):
        cdef int total_infections = 0
        for key, node in self.infections.items():
            if node[0] and node[1] == 0:
                self.infections[key][1] += 1
                total_infections += 1
            # 14 days used as placeholder to cure people and add them to the non-infected set again
            if node[1] > 14:
                self.infections[key][0] = False
                self.infections[key][1] = 0
                total_infections -= 1
        self.total_infections = total_infections
        return self.infections




    # TODO: implement this in C++, else this takes ages (and more short term memory than me before exams)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void backward(self, int truth, int len_a, int len_b):

        print(self.alpha, self.beta, self.offset)
        start = time.time()

        cdef int difference = self.total_infections - truth

        # dL/dalpha = (total_infections - true_infections) * Truce(Infection_probabilites * Sigmoid_matrix * Transposed_Intensity_Matrix)
        # backpropagating stuff
        cdef float[:, :] transpose_intensity = np.transpose(self.encounter_matrix[:, :, 0])
        cdef float[:, :] transpose_duration = np.transpose(self.encounter_matrix[:, :, 1])
        cdef float[:] p = self.p
        cdef float[:, :] F = self.F


        cdef float truce_alpha = 0
        cdef float truce_beta = 0
        cdef float truce_offset = 0

        # i know this is super inefficient but the direct way would require n^2 memory complexity ~ 100 GiB for 118k nodes
        # loop will be more efficient after transferred to C++


        if difference != 0:
            for i in range(0, len_a):
                if p[i] != 0:
                    for j in range(0, len_a):
                        if i == j:
                            for k in range(0, len_b):
                                if transpose_intensity[k, i] == 0:
                                    break
                                if F[j, k] != 0:
                                    truce_alpha += self.mult_values(p[i], F[j, k], transpose_intensity[k, i])
                                    truce_beta += self.mult_values(p[i], F[j, k], transpose_duration[k, i])
                                    truce_offset += self.mult_values(p[i], F[j, k], 1)

            self.alpha = self.alpha - 0.001 * (difference * truce_alpha)
            if not self.alpha >= 1e-9:
                self.alpha = 1e-9

            self.beta = self.beta - 0.001 * (difference * truce_beta)
            if not self.beta >= 1e-9:
                self.beta = 1e-9

            self.offset = self.offset - 0.01 * (difference * truce_offset)


            end = time.time()
            print('backward pass took:', end-start, 'seconds')
            print(difference, truce_alpha, truce_beta, truce_offset)
            print(self.alpha, self.beta, self.offset)
        else:
            end = time.time()
            print('No difference detected.')


    cdef float mult_values(self, float a, float b, float c):
        return a * b * c
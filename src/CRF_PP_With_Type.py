import theano
import theano.tensor as T
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

class CRFPP_With_Type_FB(object):
    """Conditional Random Field."""

    def __init__(self, n_tags, n_types):
        """Initialize CRF params."""
        self.n_tags = n_tags
        self.n_types = n_types

        _transition_matrix = np.random.rand(self.n_tags + 2, self.n_tags + 2)
        self.transition_matrix = theano.shared(_transition_matrix.astype(np.float32), borrow=True, name='A')

        self.n_entity_boundary_tags = 5  # as with BIOES
        self.eps = -1000
        self.b_s = np.array(
            [[self.eps] * self.n_tags + [0, self.eps]]
        ).astype(np.float32)
        self.e_s = np.array(
            [[self.eps] * self.n_tags + [self.eps, 0]]
        ).astype(np.float32)
        self.b_id = theano.shared(
            value=np.array([self.n_tags], dtype=np.int32)
        )
        self.e_id = theano.shared(
            value=np.array([self.n_tags + 1], dtype=np.int32)
        )

        self.params = [self.transition_matrix]

    def log_sum_exp(self, x, axis=None):
        """Sum probabilities in the log-space."""
        xmax = x.max(axis=axis, keepdims=True)
        xmax_ = x.max(axis=axis)
        return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))

    def alpha_recursion(
        self,
        observations,
        transitions,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False
    ):
        assert not return_best_sequence or (viterbi and not return_alpha)
        
        def recurrence(obs, transition, previous):
            previous = previous.dimshuffle(0, 'x')
            obs = obs.dimshuffle('x', 0)
            transition_org = transition.reshape((self.n_entity_boundary_tags+2, self.n_entity_boundary_tags+2))
            transition_col_1 = transition_org[:,:1]
            transition_col_2 = transition_org[:,1:5]
            transition_col_3 = transition_org[:,5:]
            transition_col = T.concatenate([transition_col_1, transition_col_2, transition_col_2, transition_col_3], axis=1)
            transition_row_1 = transition_col[:1,:]
            transition_row_2 = transition_col[1:5, :]
            transition_row_3 = transition_col[5:, :]
            _transition = T.concatenate([transition_row_1, transition_row_2, transition_row_2, transition_row_3], axis=0)
            if viterbi:
                scores = previous + obs + _transition + self.transition_matrix
                out = scores.max(axis=0)
                if return_best_sequence:
                    out2 = scores.argmax(axis=0)
                    return out, out2
                else:
                    return out
            else:
                return self.log_sum_exp(
                    previous + obs + _transition + self.transition_matrix,
                    axis=0
                )

        initial = observations[0]
        self.alpha, _ = theano.scan(
            fn=recurrence,
            outputs_info=(initial, None) if return_best_sequence else initial,
            sequences=[observations[1:], transitions],
        )

        if return_alpha:
            return self.alpha
        elif return_best_sequence:
            sequence, _ = theano.scan(
                fn=lambda beta_i, previous: beta_i[previous],
                outputs_info=T.cast(T.argmax(self.alpha[0][-1]), 'int32'),
                sequences=T.cast(self.alpha[1][::-1], 'int32')
            )
            sequence = T.concatenate([
                sequence[::-1],
                [T.argmax(self.alpha[0][-1])]
            ])
            return sequence
        else:
            if viterbi:
                return self.alpha[-1].max(axis=0)
            else:
                return self.log_sum_exp(self.alpha[-1], axis=0)

    def forward(self, observations, transitions):
        def recurrence(obs, transition, previous):
            previous = previous.dimshuffle(0, 'x')
            obs = obs.dimshuffle('x', 0)
            transition_org = transition.reshape((self.n_entity_boundary_tags+2, self.n_entity_boundary_tags+2))
            transition_col_1 = transition_org[:,:1]
            transition_col_2 = transition_org[:,1:5]
            transition_col_3 = transition_org[:,5:]
            transition_col = T.concatenate([transition_col_1, transition_col_2, transition_col_2, transition_col_3], axis=1)
            transition_row_1 = transition_col[:1,:]
            transition_row_2 = transition_col[1:5, :]
            transition_row_3 = transition_col[5:, :]
            _transition = T.concatenate([transition_row_1, transition_row_2, transition_row_2, transition_row_3], axis=0)
            return self.log_sum_exp(
                previous + obs + _transition + self.transition_matrix,
                axis=0
            )

        initial = observations[0]
        alpha, _ = theano.scan(
            fn=recurrence,
            outputs_info=initial,
            sequences=[observations[1:], transitions]
        )

        return alpha[:-1]

    def backward(self, observations, transitions):
        def recurrence(obs, transition, previous):
            previous = previous.dimshuffle(0, 'x')
            obs = obs.dimshuffle('x', 0)
            transition_org = transition.reshape((self.n_entity_boundary_tags+2, self.n_entity_boundary_tags+2))
            transition_col_1 = transition_org[:,:1]
            transition_col_2 = transition_org[:,1:5]
            transition_col_3 = transition_org[:,5:]
            transition_col = T.concatenate([transition_col_1, transition_col_2, transition_col_2, transition_col_3], axis=1)
            transition_row_1 = transition_col[:1,:]
            transition_row_2 = transition_col[1:5, :]
            transition_row_3 = transition_col[5:, :]
            _transition = T.concatenate([transition_row_1, transition_row_2, transition_row_2, transition_row_3], axis=0)
            return self.log_sum_exp(
                previous + obs + _transition + self.transition_matrix.T,
                axis=0 
            )

        initial = observations[0]
        beta, _ = theano.scan(
            fn=recurrence,
            outputs_info=initial,
            sequences=[observations[1:], transitions]
        )

        return beta[:-1]

    def forward_backward(self, observations, transitions):
        alpha = self.forward(observations, transitions)
        beta = self.backward(observations[::-1], transitions[::-1])
        return alpha, beta
    
    def likelihood(self, observations, transitions, true_labels):
        self.alpha, self.beta = self.forward_backward(observations, transitions)
        self.gamma = self.alpha + self.beta[::-1]
        model = T.nnet.softmax(self.gamma)
        loss = -T.log(model)[T.arange(true_labels.shape[0]), true_labels].sum()
        return loss

    def fprop(
        self,
        input,
        transition_input,
        ground_truth,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False,
        mode='train'
    ):
        """Propogate input through the CRF."""
        seq_length = input.shape[0]
        observations = T.concatenate(
            [input, self.eps * T.ones((seq_length, 2))],
            axis=1
        )
        observations = T.concatenate(
            [self.b_s, observations, self.e_s],
            axis=0
        )
        if mode != 'train':
            return self.alpha_recursion(
                observations,
                transition_input,
                viterbi,
                return_alpha,
                return_best_sequence
            )
        
        return self.likelihood(observations, transition_input, ground_truth)


class CRFPP_With_Type(object):
    """Conditional Random Field."""

    def __init__(self, n_tags, n_types):
        """Initialize CRF params."""
        self.n_tags = n_tags

        _transition_matrix = np.random.rand(self.n_tags + 2, self.n_tags + 2)
        self.transition_matrix = theano.shared(_transition_matrix.astype(np.float32), borrow=True, name='A')

        self.n_entity_boundary_tags = 5  # as with BIOES
        self.n_types = n_types
        self.eps = -1000
        self.b_s = np.array(
            [[self.eps] * self.n_tags + [0, self.eps]]
        ).astype(np.float32)
        self.e_s = np.array(
            [[self.eps] * self.n_tags + [self.eps, 0]]
        ).astype(np.float32)
        self.b_id = theano.shared(
            value=np.array([self.n_tags], dtype=np.int32)
        )
        self.e_id = theano.shared(
            value=np.array([self.n_tags + 1], dtype=np.int32)
        )

        self.params = [self.transition_matrix]

    def log_sum_exp(self, x, axis=None):
        """Sum probabilities in the log-space."""
        xmax = x.max(axis=axis, keepdims=True)
        xmax_ = x.max(axis=axis)
        return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))

    def alpha_recursion(
        self,
        observations,
        transitions,
        type_transitions,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False
    ):
        assert not return_best_sequence or (viterbi and not return_alpha)
        
        def recurrence(obs, transition, type_transition, previous):
            previous = previous.dimshuffle(0, 'x')
            obs = obs.dimshuffle('x', 0)
            transition_org = transition.reshape((self.n_entity_boundary_tags+2, self.n_entity_boundary_tags+2))
            transition_col_1 = transition_org[:,:1]
            transition_col_2 = transition_org[:,1:5]
            transition_col_3 = transition_org[:,5:]
            transition_col = T.concatenate([transition_col_1, transition_col_2, transition_col_2, transition_col_3], axis=1)
            transition_row_1 = transition_col[:1,:]
            transition_row_2 = transition_col[1:5,:]
            transition_row_3 = transition_col[5:,:]
            _transition = T.concatenate([transition_row_1, transition_row_2, transition_row_2, transition_row_3], axis=0)
            
            type_transition_org = type_transition.reshape((self.n_types+2, self.n_types+2))
            type_transition_col_1 = type_transition_org[:,:1]
            type_transition_col_2 = T.tile(type_transition_org[:,1:2],(1,4))
            type_transition_col_3 = T.tile(type_transition_org[:,2:3],(1,4))
            type_transition_col_4 = type_transition_org[:,3:]
            type_transition_col = T.concatenate([type_transition_col_1, type_transition_col_2, type_transition_col_3, type_transition_col_4], axis=1)
            type_transition_row_1 = type_transition_col[:1,:]
            type_transition_row_2 = T.tile(type_transition_col[1:2,:],(4,1))
            type_transition_row_3 = T.tile(type_transition_col[2:3,:],(4,1))
            type_transition_row_4 = type_transition_col[3:,:]
            _type_transition = T.concatenate([type_transition_row_1, type_transition_row_2, type_transition_row_3, type_transition_row_4], axis=0)
            
            if viterbi:
                scores = previous + obs + _transition + _type_transition + self.transition_matrix
                out = scores.max(axis=0)
                if return_best_sequence:
                    out2 = scores.argmax(axis=0)
                    return out, out2
                else:
                    return out
            else:
                return self.log_sum_exp(
                    previous + obs + _transition + _type_transition + self.transition_matrix,
                    axis=0
                )

        initial = observations[0]
        self.alpha, _ = theano.scan(
            fn=recurrence,
            outputs_info=(initial, None) if return_best_sequence else initial,
            sequences=[observations[1:], transitions, type_transitions]
        )

        if return_alpha:
            return self.alpha
        elif return_best_sequence:
            sequence, _ = theano.scan(
                fn=lambda beta_i, previous: beta_i[previous],
                outputs_info=T.cast(T.argmax(self.alpha[0][-1]), 'int32'),
                sequences=T.cast(self.alpha[1][::-1], 'int32')
            )
            sequence = T.concatenate([
                sequence[::-1],
                [T.argmax(self.alpha[0][-1])]
            ])
            return sequence
        else:
            if viterbi:
                return self.alpha[-1].max(axis=0)
            else:
                return self.log_sum_exp(self.alpha[-1], axis=0)

    def fprop(
        self,
        input,
        transition_input,
        type_transition_input,
        ground_truth,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False,
        mode='train'
    ):
        """Propogate input through the CRF."""
        seq_length = input.shape[0]
        observations = T.concatenate(
            [input, self.eps * T.ones((seq_length, 2))],
            axis=1
        )
        observations = T.concatenate(
            [self.b_s, observations, self.e_s],
            axis=0
        )
        if mode != 'train':
            return self.alpha_recursion(
                observations,
                transition_input,
                type_transition_input,
                viterbi,
                return_alpha,
                return_best_sequence
            )
        real_path_score = input[T.arange(seq_length), ground_truth].sum()
        
        padded_tags_ids = T.concatenate(
            [self.b_id, ground_truth, self.e_id],
            axis=0
        )
        
        real_path_score += self.transition_matrix[
            padded_tags_ids[T.arange(seq_length + 1)],
            padded_tags_ids[T.arange(seq_length + 1) + 1]
        ].sum()
        
        def transition_scoring(tran, type_tran, from_id, to_id):
            transition_org = tran.reshape((self.n_entity_boundary_tags+2, self.n_entity_boundary_tags+2))
            transition_col_1 = transition_org[:,:1]
            transition_col_2 = transition_org[:,1:5]
            transition_col_3 = transition_org[:,5:]
            transition_col = T.concatenate([transition_col_1, transition_col_2, transition_col_2, transition_col_3], axis=1)
            transition_row_1 = transition_col[:1,:]
            transition_row_2 = transition_col[1:5, :]
            transition_row_3 = transition_col[5:, :]
            _transition = T.concatenate([transition_row_1, transition_row_2, transition_row_2, transition_row_3], axis=0)
            
            type_transition_org = type_tran.reshape((self.n_types+2, self.n_types+2))
            type_transition_col_1 = type_transition_org[:,:1]
            type_transition_col_2 = T.tile(type_transition_org[:,1:2],(1,4))
            type_transition_col_3 = T.tile(type_transition_org[:,2:3],(1,4))
            type_transition_col_4 = type_transition_org[:,3:]
            type_transition_col = T.concatenate([type_transition_col_1, type_transition_col_2, type_transition_col_3, type_transition_col_4], axis=1)
            type_transition_row_1 = type_transition_col[:1,:]
            type_transition_row_2 = T.tile(type_transition_col[1:2,:],(4,1))
            type_transition_row_3 = T.tile(type_transition_col[2:3,:],(4,1))
            type_transition_row_4 = type_transition_col[3:,:]
            _type_transition = T.concatenate([type_transition_row_1, type_transition_row_2, type_transition_row_3, type_transition_row_4], axis=0)

            _score = _transition[from_id, to_id] + _type_transition[from_id, to_id] 
            return _score
        
        transition_scores, _ = theano.scan(fn=transition_scoring, 
                                           sequences=[transition_input,
                                                      type_transition_input,
                                                      padded_tags_ids[T.arange(seq_length + 1)], 
                                                      padded_tags_ids[T.arange(seq_length + 1) + 1]]
                                           )
        
        real_path_score += transition_scores.sum()
        
        all_paths_scores = self.alpha_recursion(
            observations,
            transition_input,
            type_transition_input,
            viterbi,
            return_alpha,
            return_best_sequence
        )
        
        cost = - (real_path_score - all_paths_scores)
        
        return cost

class CRFPP_SentenceLevelType(object):
    """Conditional Random Field."""

    def __init__(self, n_tags, n_types):
        """Initialize CRF params."""
        self.n_tags = n_tags

        _transition_matrix = np.random.rand(self.n_tags + 2, self.n_tags + 2)
        self.transition_matrix = theano.shared(_transition_matrix.astype(np.float32), borrow=True, name='A')

        self.n_entity_boundary_tags = 5  # as with BIOES
        self.n_types = n_types
        self.eps = -1000
        self.b_s = np.array(
            [[self.eps] * self.n_tags + [0, self.eps]]
        ).astype(np.float32)
        self.e_s = np.array(
            [[self.eps] * self.n_tags + [self.eps, 0]]
        ).astype(np.float32)
        self.b_id = theano.shared(
            value=np.array([self.n_tags], dtype=np.int32)
        )
        self.e_id = theano.shared(
            value=np.array([self.n_tags + 1], dtype=np.int32)
        )

        self.params = [self.transition_matrix]

        print 'Initialize CRFPP_SentenceLevelType'
        
    def log_sum_exp(self, x, axis=None):
        """Sum probabilities in the log-space."""
        xmax = x.max(axis=axis, keepdims=True)
        xmax_ = x.max(axis=axis)
        return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))

    def alpha_recursion(
        self,
        observations,
        transitions,
        type_transitions,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False
    ):
        assert not return_best_sequence or (viterbi and not return_alpha)

        type_transition_org = type_transitions.reshape((self.n_types+2, self.n_types+2))
        type_transition_col_1 = type_transition_org[:,:1]
        type_transition_col_2 = T.tile(type_transition_org[:,1:2],(1,4))
        type_transition_col_3 = T.tile(type_transition_org[:,2:3],(1,4))
        type_transition_col_4 = type_transition_org[:,3:]
        type_transition_col = T.concatenate([type_transition_col_1, type_transition_col_2, type_transition_col_3, type_transition_col_4], axis=1)
        type_transition_row_1 = type_transition_col[:1,:]
        type_transition_row_2 = T.tile(type_transition_col[1:2,:],(4,1))
        type_transition_row_3 = T.tile(type_transition_col[2:3,:],(4,1))
        type_transition_row_4 = type_transition_col[3:,:]
        _type_transition = T.concatenate([type_transition_row_1, type_transition_row_2, type_transition_row_3, type_transition_row_4], axis=0)
        
        def recurrence(obs, transition, previous, type_transition):
            previous = previous.dimshuffle(0, 'x')
            obs = obs.dimshuffle('x', 0)
            transition_org = transition.reshape((self.n_entity_boundary_tags+2, self.n_entity_boundary_tags+2))
            transition_col_1 = transition_org[:,:1]
            transition_col_2 = transition_org[:,1:5]
            transition_col_3 = transition_org[:,5:]
            transition_col = T.concatenate([transition_col_1, transition_col_2, transition_col_2, transition_col_3], axis=1)
            transition_row_1 = transition_col[:1,:]
            transition_row_2 = transition_col[1:5,:]
            transition_row_3 = transition_col[5:,:]
            _transition = T.concatenate([transition_row_1, transition_row_2, transition_row_2, transition_row_3], axis=0)
            
            if viterbi:
                scores = previous + obs + _transition + type_transition + self.transition_matrix
                out = scores.max(axis=0)
                if return_best_sequence:
                    out2 = scores.argmax(axis=0)
                    return out, out2
                else:
                    return out
            else:
                return self.log_sum_exp(
                    previous + obs + _transition + type_transition + self.transition_matrix,
                    axis=0
                )

        initial = observations[0]
        self.alpha, _ = theano.scan(
            fn=recurrence,
            outputs_info=(initial, None) if return_best_sequence else initial,
            sequences=[observations[1:], transitions],
            non_sequences=_type_transition
        )

        if return_alpha:
            return self.alpha
        elif return_best_sequence:
            sequence, _ = theano.scan(
                fn=lambda beta_i, previous: beta_i[previous],
                outputs_info=T.cast(T.argmax(self.alpha[0][-1]), 'int32'),
                sequences=T.cast(self.alpha[1][::-1], 'int32')
            )
            sequence = T.concatenate([
                sequence[::-1],
                [T.argmax(self.alpha[0][-1])]
            ])
            return sequence
        else:
            if viterbi:
                return self.alpha[-1].max(axis=0)
            else:
                return self.log_sum_exp(self.alpha[-1], axis=0)

    def fprop(
        self,
        input,
        transition_input,
        type_transition_input,
        ground_truth,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False,
        mode='train'
    ):
        """Propogate input through the CRF."""
        seq_length = input.shape[0]
        observations = T.concatenate(
            [input, self.eps * T.ones((seq_length, 2))],
            axis=1
        )
        observations = T.concatenate(
            [self.b_s, observations, self.e_s],
            axis=0
        )
        if mode != 'train':
            return self.alpha_recursion(
                observations,
                transition_input,
                type_transition_input,
                viterbi,
                return_alpha,
                return_best_sequence
            )
        real_path_score = input[T.arange(seq_length), ground_truth].sum()
        
        padded_tags_ids = T.concatenate(
            [self.b_id, ground_truth, self.e_id],
            axis=0
        )
        
        real_path_score += self.transition_matrix[
            padded_tags_ids[T.arange(seq_length + 1)],
            padded_tags_ids[T.arange(seq_length + 1) + 1]
        ].sum()

        type_transition_org = type_transition_input.reshape((self.n_types+2, self.n_types+2))
        type_transition_col_1 = type_transition_org[:,:1]
        type_transition_col_2 = T.tile(type_transition_org[:,1:2],(1,4))
        type_transition_col_3 = T.tile(type_transition_org[:,2:3],(1,4))
        type_transition_col_4 = type_transition_org[:,3:]
        type_transition_col = T.concatenate([type_transition_col_1, type_transition_col_2, type_transition_col_3, type_transition_col_4], axis=1)
        type_transition_row_1 = type_transition_col[:1,:]
        type_transition_row_2 = T.tile(type_transition_col[1:2,:],(4,1))
        type_transition_row_3 = T.tile(type_transition_col[2:3,:],(4,1))
        type_transition_row_4 = type_transition_col[3:,:]
        type_transitions = T.concatenate([type_transition_row_1, type_transition_row_2, type_transition_row_3, type_transition_row_4], axis=0)
        
        def transition_scoring(tran, from_id, to_id, type_transition):
            transition_org = tran.reshape((self.n_entity_boundary_tags+2, self.n_entity_boundary_tags+2))
            transition_col_1 = transition_org[:,:1]
            transition_col_2 = transition_org[:,1:5]
            transition_col_3 = transition_org[:,5:]
            transition_col = T.concatenate([transition_col_1, transition_col_2, transition_col_2, transition_col_3], axis=1)
            transition_row_1 = transition_col[:1,:]
            transition_row_2 = transition_col[1:5, :]
            transition_row_3 = transition_col[5:, :]
            _transition = T.concatenate([transition_row_1, transition_row_2, transition_row_2, transition_row_3], axis=0)
            
            _score = _transition[from_id, to_id] + type_transition[from_id, to_id] 
            return _score
        
        transition_scores, _ = theano.scan(fn=transition_scoring, 
                                           sequences=[transition_input,
                                                      padded_tags_ids[T.arange(seq_length + 1)], 
                                                      padded_tags_ids[T.arange(seq_length + 1) + 1]],
                                           non_sequences=type_transitions
                                           )
        
        real_path_score += transition_scores.sum()
        
        all_paths_scores = self.alpha_recursion(
            observations,
            transition_input,
            type_transition_input,
            viterbi,
            return_alpha,
            return_best_sequence
        )
        
        cost = - (real_path_score - all_paths_scores)
        
        return cost

class CRFPP_SentenceLevelType_Condensed(object):
    """Conditional Random Field."""

    def __init__(self, n_tags, n_types):
        """Initialize CRFPP_SentenceLevelType_Condensed params."""
        self.n_tags = n_tags

        _transition_matrix = np.random.rand(self.n_tags, self.n_tags)
        self.transition_matrix = theano.shared(_transition_matrix.astype(theano.config.floatX), borrow=True, name='A')

        self.n_entity_boundary_tags = 5  # as with BIOES
        self.n_types = n_types
        
        self.params = [self.transition_matrix]

        print 'Initialize CRFPP_SentenceLevelType_Condensed'
        
    def log_sum_exp(self, x, axis=None):
        """Sum probabilities in the log-space."""
        xmax = x.max(axis=axis, keepdims=True)
        xmax_ = x.max(axis=axis)
        return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))

    def alpha_recursion(
        self,
        observations,
        transitions,
        type_transitions,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False
    ):
        assert not return_best_sequence or (viterbi and not return_alpha)

        type_transition_org = type_transitions.reshape((self.n_types, self.n_types))
        type_transition_col_1 = type_transition_org[:,:1]
        type_transition_col_2 = T.tile(type_transition_org[:,1:2],(1,4))
        type_transition_col_3 = T.tile(type_transition_org[:,2:],(1,4))
        type_transition_col = T.concatenate([type_transition_col_1, type_transition_col_2, type_transition_col_3], axis=1)
        type_transition_row_1 = type_transition_col[:1,:]
        type_transition_row_2 = T.tile(type_transition_col[1:2,:],(4,1))
        type_transition_row_3 = T.tile(type_transition_col[2:,:],(4,1))
        _type_transition = T.concatenate([type_transition_row_1, type_transition_row_2, type_transition_row_3], axis=0)
        
        def recurrence(obs, transition, previous, type_transition):
            previous = previous.dimshuffle(0, 'x')
            obs = obs.dimshuffle('x', 0)
            transition_org = transition.reshape((self.n_entity_boundary_tags, self.n_entity_boundary_tags))
            transition_col_1 = transition_org[:,:1]
            transition_col_2 = T.tile(transition_org[:,1:], (1,2))
            transition_col = T.concatenate([transition_col_1, transition_col_2], axis=1)
            transition_row_1 = transition_col[:1,:]
            transition_row_2 = T.tile(transition_col[1:,:], (2,1))
            _transition = T.concatenate([transition_row_1, transition_row_2], axis=0)
            
            if viterbi:
                scores = previous + obs + _transition + type_transition + self.transition_matrix
                out = scores.max(axis=0)
                if return_best_sequence:
                    out2 = scores.argmax(axis=0)    # previous index
                    return out, out2           # scores, previous sequence from which corresponding index
                else:
                    return out
            else:
                return self.log_sum_exp(
                    previous + obs + _transition + type_transition + self.transition_matrix,
                    axis=0
                )

        initial = observations[0]
        self.alpha, _ = theano.scan(
            fn=recurrence,
            sequences=[observations[1:], transitions],
            outputs_info=(initial, None) if return_best_sequence else initial,
            non_sequences=_type_transition
        )

        if return_alpha:
            return self.alpha
        elif return_best_sequence:
            sequence, _ = theano.scan(                
                fn=lambda beta_i, previous: beta_i[previous],     # back tracking
                sequences=T.cast(self.alpha[1][::-1], 'int32'),    # previous sequence
                outputs_info=T.cast(T.argmax(self.alpha[0][-1]), 'int32')   # last index of sequence having maximum score
            )
            sequence = T.concatenate([
                sequence[::-1],
                [T.argmax(self.alpha[0][-1])]
            ])
            return sequence
        else:
            if viterbi:
                return self.alpha[-1].max(axis=0)
            else:
                return self.log_sum_exp(self.alpha[-1], axis=0)

    def fprop(
        self,
        input_sequence,
        transition_input,
        type_transition_input,
        ground_truth,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False,
        mode='train'
    ):
        """Propogate input through the CRF."""
        seq_length = input_sequence.shape[0]
        if mode != 'train':
            return self.alpha_recursion(
                input_sequence,
                transition_input,
                type_transition_input,
                viterbi,
                return_alpha,
                return_best_sequence
            )
        real_path_score = input_sequence[T.arange(seq_length), ground_truth].sum()
                
        real_path_score += self.transition_matrix[
            ground_truth[T.arange(seq_length - 1)],
            ground_truth[T.arange(seq_length - 1) + 1]
        ].sum()

        type_transition_org = type_transition_input.reshape((self.n_types, self.n_types))
        type_transition_col_1 = type_transition_org[:,:1]
        type_transition_col_2 = T.tile(type_transition_org[:,1:2],(1,4))
        type_transition_col_3 = T.tile(type_transition_org[:,2:],(1,4))
        type_transition_col = T.concatenate([type_transition_col_1, type_transition_col_2, type_transition_col_3], axis=1)
        type_transition_row_1 = type_transition_col[:1,:]
        type_transition_row_2 = T.tile(type_transition_col[1:2,:],(4,1))
        type_transition_row_3 = T.tile(type_transition_col[2:,:],(4,1))
        type_transitions = T.concatenate([type_transition_row_1, type_transition_row_2, type_transition_row_3], axis=0)
        
        def transition_scoring(tran, from_id, to_id, type_transition):
            transition_org = tran.reshape((self.n_entity_boundary_tags, self.n_entity_boundary_tags))
            transition_col_1 = transition_org[:,:1]
            transition_col_2 = T.tile(transition_org[:,1:], (1,2))
            transition_col = T.concatenate([transition_col_1, transition_col_2], axis=1)
            transition_row_1 = transition_col[:1,:]
            transition_row_2 = T.tile(transition_col[1:,:], (2,1))
            _transition = T.concatenate([transition_row_1, transition_row_2], axis=0)
            _score = _transition[from_id, to_id] + type_transition[from_id, to_id] 
            return _score
        
        transition_scores, _ = theano.scan(fn=transition_scoring, 
                                           sequences=[transition_input,
                                                      ground_truth[T.arange(seq_length - 1)], 
                                                      ground_truth[T.arange(seq_length - 1) + 1]],
                                           non_sequences=type_transitions
                                           )
        
        real_path_score += transition_scores.sum()
        
        all_paths_scores = self.alpha_recursion(
            input_sequence,
            transition_input,
            type_transition_input,
            viterbi,
            return_alpha,
            return_best_sequence
        )
        
        cost = - (real_path_score - all_paths_scores)
        
        return cost

class CRFPP_SentenceLevelForBothPhraseAndType(object):
    """Conditional Random Field."""

    def __init__(self, n_tags, n_types, start_end_added=False):
        """Initialize CRFPP_SentenceLevelForBothPhraseAndType_ params."""
        self.start_end_added = start_end_added
        self.n_tags = n_tags
        
        if start_end_added == True:
            print 'Begin End Tag Added'
            _transition_matrix = np.random.rand(self.n_tags+2, self.n_tags+2)
        else:
            print 'Begin End Tag Removed'
            _transition_matrix = np.random.rand(self.n_tags, self.n_tags)
        self.transition_matrix = theano.shared(_transition_matrix.astype(theano.config.floatX), borrow=True, name='A')

        self.n_entity_boundary_tags = 5  # as with BIOES
        self.n_types = n_types
        
        if start_end_added == True:
            self.eps = -1000
            self.b_s = np.array(
                [[self.eps] * self.n_tags + [0, self.eps]]
            ).astype(np.float32)
            self.e_s = np.array(
                [[self.eps] * self.n_tags + [self.eps, 0]]
            ).astype(np.float32)
            self.b_id = theano.shared(
                value=np.array([self.n_tags], dtype=np.int32)
            )
            self.e_id = theano.shared(
                value=np.array([self.n_tags + 1], dtype=np.int32)
            )

        self.params = [self.transition_matrix]

        print 'Initialize CRFPP_SentenceLevelForBothPhraseAndType'
    
    def type_transition_shaping(self, type_transitions):
        if self.start_end_added == True:
            type_transition_org = type_transitions.reshape((self.n_types, self.n_types))
            type_transition_col_1 = type_transition_org[:,:1]
            type_transition_col_2 = T.tile(type_transition_org[:,1:2],(1,4))
            type_transition_col_3 = T.tile(type_transition_org[:,2:3],(1,4))
            type_transition_col_4 = type_transition_org[:,3:]
            type_transition_col = T.concatenate([type_transition_col_1, type_transition_col_2, type_transition_col_3, type_transition_col_4], axis=1)
            type_transition_row_1 = type_transition_col[:1,:]
            type_transition_row_2 = T.tile(type_transition_col[1:2,:],(4,1))
            type_transition_row_3 = T.tile(type_transition_col[2:3,:],(4,1))
            type_transition_row_4 = type_transition_col[3:,:]
            _type_transition = T.concatenate([type_transition_row_1, type_transition_row_2, type_transition_row_3, type_transition_row_4], axis=0)
        else:
            type_transition_org = type_transitions.reshape((self.n_types, self.n_types))
            type_transition_col_1 = type_transition_org[:,:1]
            type_transition_col_2 = T.tile(type_transition_org[:,1:2],(1,4))
            type_transition_col_3 = T.tile(type_transition_org[:,2:],(1,4))
            type_transition_col = T.concatenate([type_transition_col_1, type_transition_col_2, type_transition_col_3], axis=1)
            type_transition_row_1 = type_transition_col[:1,:]
            type_transition_row_2 = T.tile(type_transition_col[1:2,:],(4,1))
            type_transition_row_3 = T.tile(type_transition_col[2:,:],(4,1))
            _type_transition = T.concatenate([type_transition_row_1, type_transition_row_2, type_transition_row_3], axis=0)
        return _type_transition

    def phrase_transition_shaping(self, phrase_transitions):
        if self.start_end_added == True:
            phrase_transition_org = transitions.reshape((self.n_entity_boundary_tags+2, self.n_entity_boundary_tags+2))
            phrase_transition_col_1 = phrase_transition_org[:,:1]
            phrase_transition_col_2 = T.tile(phrase_transition_org[:,1:5], (1,2))
            phrase_transition_col_3 = phrase_transition_org[:,5:]
            phrase_transition_col = T.concatenate([phrase_transition_col_1, phrase_transition_col_2, phrase_transition_col_3], axis=1)
            phrase_transition_row_1 = phrase_transition_col[:1,:]
            phrase_transition_row_2 = T.tile(phrase_transition_col[1:5,:], (2,1))
            phrase_transition_row_3 = phrase_transition_col[5:,:]
            _phrase_transition = T.concatenate([phrase_transition_row_1, phrase_transition_row_2, phrase_transition_row_3], axis=0)
        else:
            phrase_transition_org = transitions.reshape((self.n_entity_boundary_tags, self.n_entity_boundary_tags))
            phrase_transition_col_1 = phrase_transition_org[:,:1]
            phrase_transition_col_2 = T.tile(phrase_transition_org[:,1:], (1,2))
            phrase_transition_col = T.concatenate([phrase_transition_col_1, phrase_transition_col_2], axis=1)
            phrase_transition_row_1 = phrase_transition_col[:1,:]
            phrase_transition_row_2 = T.tile(phrase_transition_col[1:,:], (2,1))
            _phrase_transition = T.concatenate([phrase_transition_row_1, phrase_transition_row_2], axis=0)
        return _phrase_transition
    
    def log_sum_exp(self, x, axis=None):
        """Sum probabilities in the log-space."""
        xmax = x.max(axis=axis, keepdims=True)
        xmax_ = x.max(axis=axis)
        return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))

    def alpha_recursion(
        self,
        observations,
        transitions,
        type_transitions,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False
    ):
        assert not return_best_sequence or (viterbi and not return_alpha)
        
        _type_transition = self.type_transition_shaping(type_transitions)
        _phrase_transition = self.phrase_transition_shaping(transitions)
        
        def recurrence(obs, previous, type_transition, phrase_transition):
            previous = previous.dimshuffle(0, 'x')
            obs = obs.dimshuffle('x', 0)
            
            if viterbi:
                scores = previous + obs + phrase_transition + type_transition + self.transition_matrix
                out = scores.max(axis=0)
                if return_best_sequence:
                    out2 = scores.argmax(axis=0)    # previous index
                    return out, out2           # scores, previous sequence from which corresponding index
                else:
                    return out
            else:
                return self.log_sum_exp(
                    previous + obs + phrase_transition + type_transition + self.transition_matrix,
                    axis=0
                )

        initial = observations[0]
        self.alpha, _ = theano.scan(
            fn=recurrence,
            sequences=observations[1:],
            outputs_info=(initial, None) if return_best_sequence else initial,
            non_sequences=[_type_transition, _phrase_transition]
        )

        if return_alpha:
            return self.alpha
        elif return_best_sequence:
            sequence, _ = theano.scan(                
                fn=lambda beta_i, previous: beta_i[previous],     # back tracking
                sequences=T.cast(self.alpha[1][::-1], 'int32'),    # previous sequence
                outputs_info=T.cast(T.argmax(self.alpha[0][-1]), 'int32')   # last index of sequence having maximum score
            )
            sequence = T.concatenate([
                sequence[::-1],
                [T.argmax(self.alpha[0][-1])]
            ])
            return sequence
        else:
            if viterbi:
                return self.alpha[-1].max(axis=0)
            else:
                return self.log_sum_exp(self.alpha[-1], axis=0)

    def fprop(
        self,
        input_sequence,
        transition_input,
        type_transition_input,
        ground_truth,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False,
        mode='train'
    ):
        """Propogate input through the CRF."""
        if self.start_end_added == True:
            seq_length = input_sequence.shape[0]
            observations = T.concatenate(
                [input_sequence, self.eps * T.ones((seq_length, 2))],
                axis=1
            )
            observations = T.concatenate(
                [self.b_s, observations, self.e_s],
                axis=0
            )
            if mode != 'train':
                return self.alpha_recursion(
                    observations,
                    transition_input,
                    type_transition_input,
                    viterbi,
                    return_alpha,
                    return_best_sequence
                )
            real_path_score = input_sequence[T.arange(seq_length), ground_truth].sum()
            padded_tags_ids = T.concatenate(
                [self.b_id, ground_truth, self.e_id],
                axis=0
            )
            real_path_score += self.transition_matrix[
                padded_tags_ids[T.arange(seq_length + 1)],
                padded_tags_ids[T.arange(seq_length + 1) + 1]
            ].sum()
        else:
            seq_length = input_sequence.shape[0]
            if mode != 'train':
                return self.alpha_recursion(
                    input_sequence,
                    transition_input,
                    type_transition_input,
                    viterbi,
                    return_alpha,
                    return_best_sequence
                )
            real_path_score = input_sequence[T.arange(seq_length), ground_truth].sum()
            real_path_score += self.transition_matrix[
                ground_truth[T.arange(seq_length - 1)],
                ground_truth[T.arange(seq_length - 1) + 1]
            ].sum()
        
        type_transitions = self.type_transition_shaping(type_transition_input)
        phrase_transitions = self.phrase_transition_shaping(transition_input)

        def transition_scoring(from_id, to_id, phrase_transition, type_transition):
            _score = phrase_transition[from_id, to_id] + type_transition[from_id, to_id] 
            return _score

        if self.start_end_added == True:        
            transition_scores, _ = theano.scan(fn=transition_scoring, 
                                               sequences=[padded_tags_ids[T.arange(seq_length + 1)], 
                                                          padded_tags_ids[T.arange(seq_length + 1) + 1]],
                                               non_sequences=[phrase_transitions, type_transitions]
                                               )
            real_path_score += transition_scores.sum()
            all_paths_scores = self.alpha_recursion(
                observations,
                transition_input,
                type_transition_input,
                viterbi,
                return_alpha,
                return_best_sequence
            )
        else:
            transition_scores, _ = theano.scan(fn=transition_scoring, 
                                               sequences=[ground_truth[T.arange(seq_length - 1)], 
                                                          ground_truth[T.arange(seq_length - 1) + 1]],
                                               non_sequences=[phrase_transitions, type_transitions]
                                               )
            real_path_score += transition_scores.sum()
            all_paths_scores = self.alpha_recursion(
                input_sequence,
                transition_input,
                type_transition_input,
                viterbi,
                return_alpha,
                return_best_sequence
            )
        
        cost = - (real_path_score - all_paths_scores)
        
        return cost

class CRFPP_SentenceAndPairwiseLevelForBothPhraseAndType(object):
    """Conditional Random Field."""

    def __init__(self, n_tags, n_types, start_end_added=False):
        """Initialize CRFPP_SentenceLevelForBothPhraseAndType_ params."""
        self.start_end_added = start_end_added
        self.n_tags = n_tags
        
        if start_end_added == True:
            print 'Begin End Tag Added'
            _transition_matrix = np.random.rand(self.n_tags+2, self.n_tags+2)
        else:
            print 'Begin End Tag Removed'
            _transition_matrix = np.random.rand(self.n_tags, self.n_tags)
        self.transition_matrix = theano.shared(_transition_matrix.astype(theano.config.floatX), borrow=True, name='A')

        self.n_entity_boundary_tags = 5  # as with BIOES
        self.n_types = n_types
        
        if start_end_added == True:
            self.eps = -1000
            self.b_s = np.array(
                [[self.eps] * self.n_tags + [0, self.eps]]
            ).astype(np.float32)
            self.e_s = np.array(
                [[self.eps] * self.n_tags + [self.eps, 0]]
            ).astype(np.float32)
            self.b_id = theano.shared(
                value=np.array([self.n_tags], dtype=np.int32)
            )
            self.e_id = theano.shared(
                value=np.array([self.n_tags + 1], dtype=np.int32)
            )

        self.params = [self.transition_matrix]

        print 'Initialize CRFPP_SentenceLevelForBothPhraseAndType'
    
    def type_transition_shaping(self, type_transitions):
        if self.start_end_added == True:
            type_transition_org = type_transitions.reshape((self.n_types, self.n_types))
            type_transition_col_1 = type_transition_org[:,:1]
            type_transition_col_2 = T.tile(type_transition_org[:,1:2],(1,4))
            type_transition_col_3 = T.tile(type_transition_org[:,2:3],(1,4))
            type_transition_col_4 = type_transition_org[:,3:]
            type_transition_col = T.concatenate([type_transition_col_1, type_transition_col_2, type_transition_col_3, type_transition_col_4], axis=1)
            type_transition_row_1 = type_transition_col[:1,:]
            type_transition_row_2 = T.tile(type_transition_col[1:2,:],(4,1))
            type_transition_row_3 = T.tile(type_transition_col[2:3,:],(4,1))
            type_transition_row_4 = type_transition_col[3:,:]
            _type_transition = T.concatenate([type_transition_row_1, type_transition_row_2, type_transition_row_3, type_transition_row_4], axis=0)
        else:
            type_transition_org = type_transitions.reshape((self.n_types, self.n_types))
            type_transition_col_1 = type_transition_org[:,:1]
            type_transition_col_2 = T.tile(type_transition_org[:,1:2],(1,4))
            type_transition_col_3 = T.tile(type_transition_org[:,2:],(1,4))
            type_transition_col = T.concatenate([type_transition_col_1, type_transition_col_2, type_transition_col_3], axis=1)
            type_transition_row_1 = type_transition_col[:1,:]
            type_transition_row_2 = T.tile(type_transition_col[1:2,:],(4,1))
            type_transition_row_3 = T.tile(type_transition_col[2:,:],(4,1))
            _type_transition = T.concatenate([type_transition_row_1, type_transition_row_2, type_transition_row_3], axis=0)
        return _type_transition

    def phrase_transition_shaping(self, phrase_transitions):
        if self.start_end_added == True:
            phrase_transition_org = transitions.reshape((self.n_entity_boundary_tags+2, self.n_entity_boundary_tags+2))
            phrase_transition_col_1 = phrase_transition_org[:,:1]
            phrase_transition_col_2 = T.tile(phrase_transition_org[:,1:5], (1,2))
            phrase_transition_col_3 = phrase_transition_org[:,5:]
            phrase_transition_col = T.concatenate([phrase_transition_col_1, phrase_transition_col_2, phrase_transition_col_3], axis=1)
            phrase_transition_row_1 = phrase_transition_col[:1,:]
            phrase_transition_row_2 = T.tile(phrase_transition_col[1:5,:], (2,1))
            phrase_transition_row_3 = phrase_transition_col[5:,:]
            _phrase_transition = T.concatenate([phrase_transition_row_1, phrase_transition_row_2, phrase_transition_row_3], axis=0)
        else:
            phrase_transition_org = transitions.reshape((self.n_entity_boundary_tags, self.n_entity_boundary_tags))
            phrase_transition_col_1 = phrase_transition_org[:,:1]
            phrase_transition_col_2 = T.tile(phrase_transition_org[:,1:], (1,2))
            phrase_transition_col = T.concatenate([phrase_transition_col_1, phrase_transition_col_2], axis=1)
            phrase_transition_row_1 = phrase_transition_col[:1,:]
            phrase_transition_row_2 = T.tile(phrase_transition_col[1:,:], (2,1))
            _phrase_transition = T.concatenate([phrase_transition_row_1, phrase_transition_row_2], axis=0)
        return _phrase_transition
    
    def log_sum_exp(self, x, axis=None):
        """Sum probabilities in the log-space."""
        xmax = x.max(axis=axis, keepdims=True)
        xmax_ = x.max(axis=axis)
        return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))

    def alpha_recursion(
        self,
        observations,
        transitions,
        type_transitions,
        phrase_transition_inputs,
        type_transition_inputs,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False
    ):
        assert not return_best_sequence or (viterbi and not return_alpha)
        
        _type_transition = self.type_transition_shaping(type_transitions)
        _phrase_transition = self.phrase_transition_shaping(transitions)
        
        def recurrence(obs, previous, phrase_transition_pairwise_input, type_transition_pairwise_input, type_transition, phrase_transition):
            previous = previous.dimshuffle(0, 'x')
            obs = obs.dimshuffle('x', 0)
            phrase_transition_shaped = self.phrase_transition_shaping(phrase_transition_pairwise_input)
            type_transition_shaped = self.type_transition_shaping(type_transition_pairwise_input)
            
            if viterbi:
                scores = previous + obs + phrase_transition_shaped + type_transition_shaped + phrase_transition + type_transition + self.transition_matrix
                out = scores.max(axis=0)
                if return_best_sequence:
                    out2 = scores.argmax(axis=0)    # previous index
                    return out, out2           # scores, previous sequence from which corresponding index
                else:
                    return out
            else:
                return self.log_sum_exp(
                    previous + obs + phrase_transition_shaped + type_transition_shaped + phrase_transition + type_transition + self.transition_matrix,
                    axis=0
                )

        initial = observations[0]
        self.alpha, _ = theano.scan(
            fn=recurrence,
            sequences=[observations[1:], phrase_transition_inputs, type_transition_inputs],
            outputs_info=(initial, None) if return_best_sequence else initial,
            non_sequences=[_type_transition, _phrase_transition]
        )

        if return_alpha:
            return self.alpha
        elif return_best_sequence:
            sequence, _ = theano.scan(                
                fn=lambda beta_i, previous: beta_i[previous],     # back tracking
                sequences=T.cast(self.alpha[1][::-1], 'int32'),    # previous sequence
                outputs_info=T.cast(T.argmax(self.alpha[0][-1]), 'int32')   # last index of sequence having maximum score
            )
            sequence = T.concatenate([
                sequence[::-1],
                [T.argmax(self.alpha[0][-1])]
            ])
            return sequence
        else:
            if viterbi:
                return self.alpha[-1].max(axis=0)
            else:
                return self.log_sum_exp(self.alpha[-1], axis=0)

    def fprop(
        self,
        input_sequence,
        transition_input,
        type_transition_input,
        transition_input_pairwise,
        type_transition_input_pairwise,
        ground_truth,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False,
        mode='train'
    ):
        """Propogate input through the CRF."""
        if self.start_end_added == True:
            seq_length = input_sequence.shape[0]
            observations = T.concatenate(
                [input_sequence, self.eps * T.ones((seq_length, 2))],
                axis=1
            )
            observations = T.concatenate(
                [self.b_s, observations, self.e_s],
                axis=0
            )
            if mode != 'train':
                return self.alpha_recursion(
                    observations,
                    transition_input,
                    type_transition_input,
                    transition_input_pairwise,
                    type_transition_input_pairwise,
                    viterbi,
                    return_alpha,
                    return_best_sequence
                )
            real_path_score = input_sequence[T.arange(seq_length), ground_truth].sum()
            padded_tags_ids = T.concatenate(
                [self.b_id, ground_truth, self.e_id],
                axis=0
            )
            real_path_score += self.transition_matrix[
                padded_tags_ids[T.arange(seq_length + 1)],
                padded_tags_ids[T.arange(seq_length + 1) + 1]
            ].sum()
        else:
            seq_length = input_sequence.shape[0]
            if mode != 'train':
                return self.alpha_recursion(
                    input_sequence,
                    transition_input,
                    type_transition_input,
                    transition_input_pairwise,
                    type_transition_input_pairwise,
                    viterbi,
                    return_alpha,
                    return_best_sequence
                )
            real_path_score = input_sequence[T.arange(seq_length), ground_truth].sum()
            real_path_score += self.transition_matrix[
                ground_truth[T.arange(seq_length - 1)],
                ground_truth[T.arange(seq_length - 1) + 1]
            ].sum()
        
        type_transitions = self.type_transition_shaping(type_transition_input)
        phrase_transitions = self.phrase_transition_shaping(transition_input)

        def transition_scoring(phrase_transition_input, type_transition_input, from_id, to_id, phrase_transition, type_transition):
            _score = self.phrase_transition_shaping(phrase_transition_input)
            _score += self.type_transition_shaping(type_transition_input) 
            _score += phrase_transition[from_id, to_id] + type_transition[from_id, to_id] 
            return _score

        if self.start_end_added == True:        
            transition_scores, _ = theano.scan(fn=transition_scoring, 
                                               sequences=[transition_input_pairwise,
                                                          type_transition_input_pairwise,
                                                          padded_tags_ids[T.arange(seq_length + 1)], 
                                                          padded_tags_ids[T.arange(seq_length + 1) + 1]],
                                               non_sequences=[phrase_transitions, type_transitions]
                                               )
            real_path_score += transition_scores.sum()
            all_paths_scores = self.alpha_recursion(
                observations,
                transition_input,
                type_transition_input,
                viterbi,
                return_alpha,
                return_best_sequence
            )
        else:
            transition_scores, _ = theano.scan(fn=transition_scoring, 
                                               sequences=[transition_input_pairwise,
                                                          type_transition_input_pairwise,
                                                          ground_truth[T.arange(seq_length - 1)], 
                                                          ground_truth[T.arange(seq_length - 1) + 1]],
                                               non_sequences=[phrase_transitions, type_transitions]
                                               )
            real_path_score += transition_scores.sum()
            all_paths_scores = self.alpha_recursion(
                input_sequence,
                transition_input,
                type_transition_input,
                viterbi,
                return_alpha,
                return_best_sequence
            )
        
        cost = - (real_path_score - all_paths_scores)
        
        return cost

class CRFPP_FB_Fx(object):
    """Conditional Random Field Pairwise Potential."""

    def __init__(self, n_tags):
        self.n_tags = n_tags
        
#         _transition_matrix = np.random.rand(self.n_tags, self.n_tags)
#         self.transition_matrix = theano.shared(_transition_matrix.astype(theano.config.floatX), borrow=True, name='A')
#  
#         self.params = [self.transition_matrix]

        print 'Initialize CRFPP_FB_Fx'
    
    def log_sum_exp(self, x, axis=None):
        """Sum probabilities in the log-space."""
        xmax = x.max(axis=axis, keepdims=True)
        xmax_ = x.max(axis=axis)
        return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))

    def alpha_recursion(
        self,
        observations,
        transitions,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False
    ):
        assert not return_best_sequence or (viterbi and not return_alpha)
                
        def recurrence(obs, transition_input, previous):
            previous = previous.dimshuffle(0, 'x')
            obs = obs.dimshuffle('x', 0)
            pairwise_transition = transition_input.reshape((self.n_tags, self.n_tags))
            
            if viterbi:
#                 scores = previous + obs + pairwise_transition + self.transition_matrix
                scores = previous + obs + pairwise_transition
#                 scores = previous + obs + pairwise_transition
                out = scores.max(axis=0)
                if return_best_sequence:
                    out2 = scores.argmax(axis=0)    # previous index
                    return out, out2           # scores, previous sequence from which corresponding index
                else:
                    return out
            else:
                return self.log_sum_exp(
                    previous + obs + pairwise_transition,
#                     previous + obs + pairwise_transition,
                    axis=0
                )

        initial = observations[0]
        self.alpha, _ = theano.scan(
            fn=recurrence,
            sequences=[observations[1:], transitions],
            outputs_info=(initial, None) if return_best_sequence else initial
        )

        if return_alpha:
            return self.alpha
        elif return_best_sequence:
            sequence, _ = theano.scan(                
                fn=lambda beta_i, previous: beta_i[previous],     # back tracking
                sequences=T.cast(self.alpha[1][::-1], 'int32'),    # previous sequence
                outputs_info=T.cast(T.argmax(self.alpha[0][-1]), 'int32')   # last index of sequence having maximum score
            )
            sequence = T.concatenate([
                sequence[::-1],
                [T.argmax(self.alpha[0][-1])]
            ])
            return sequence
        else:
            if viterbi:
                return self.alpha[-1].max(axis=0)
            else:
                return self.log_sum_exp(self.alpha[-1], axis=0)

    def forward(self, observations, transitions):
        def recurrence(obs, transition, previous):
            previous = previous.dimshuffle(0, 'x')
            obs = obs.dimshuffle('x', 0)
            pairwise_transition = transition.reshape((self.n_tags, self.n_tags))
            return self.log_sum_exp(
                previous + obs + pairwise_transition,
                axis=0
            )

        initial = observations[0]
        alpha, _ = theano.scan(
            fn=recurrence,
            outputs_info=initial,
            sequences=[observations[1:], transitions]
        )

        return alpha[:-1]

    def backward(self, observations, transitions):
        def recurrence(obs, transition, previous):
            previous = previous.dimshuffle(0, 'x')
            obs = obs.dimshuffle('x', 0)
            pairwise_transition = transition.reshape((self.n_tags, self.n_tags))
            return self.log_sum_exp(
                previous + obs + pairwise_transition,
                axis=0 
            )

        initial = observations[0]
        beta, _ = theano.scan(
            fn=recurrence,
            outputs_info=initial,
            sequences=[observations[1:], transitions]
        )

        return beta[:-1]

    def forward_backward(self, observations, transitions):
        alpha = self.forward(observations, transitions)
        beta = self.backward(observations[::-1], transitions[::-1])
        return alpha, beta
    
    def likelihood(self, observations, transitions, true_labels):
        self.alpha, self.beta = self.forward_backward(observations, transitions)
        self.gamma = self.alpha + self.beta[::-1]
        model = T.nnet.softmax(self.gamma)
        loss = -T.log(model)[T.arange(true_labels.shape[0]), true_labels].mean()
        return loss

    def fprop(
        self,
        input_sequence,
        transition_input,
        ground_truth,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False,
        mode='train'
    ):
        """Propogate input through the CRF."""
#         seq_length = input_sequence.shape[0]
        if mode != 'train':
            return self.alpha_recursion(
                input_sequence,
                transition_input,
                viterbi,
                return_alpha,
                return_best_sequence
            )
            
        cost = self.likelihood(input_sequence, transition_input, ground_truth)
            
        return cost

class CRFPP_Fx(object):
    """Conditional Random Field Pairwise Potential."""

    def __init__(self, n_tags):
        self.n_tags = n_tags
        
        _transition_matrix = np.random.rand(self.n_tags, self.n_tags)
        self.transition_matrix = theano.shared(_transition_matrix.astype(theano.config.floatX), borrow=True, name='A')
  
        self.params = [self.transition_matrix]

        print 'Initialize CRFPP_Fx'
    
    def log_sum_exp(self, x, axis=None):
        """Sum probabilities in the log-space."""
        xmax = x.max(axis=axis, keepdims=True)
        xmax_ = x.max(axis=axis)
        return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))

    def alpha_recursion(
        self,
        observations,
        transitions,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False
    ):
        assert not return_best_sequence or (viterbi and not return_alpha)
                
        def recurrence(obs, transition_input, previous):
            previous = previous.dimshuffle(0, 'x')
            obs = obs.dimshuffle('x', 0)
            pairwise_transition = transition_input.reshape((self.n_tags, self.n_tags))
            
            if viterbi:
                scores = previous + obs + pairwise_transition + self.transition_matrix
#                 scores = previous + obs + pairwise_transition
#                 scores = previous + obs + pairwise_transition
                out = scores.max(axis=0)
                if return_best_sequence:
                    out2 = scores.argmax(axis=0)    # previous index
                    return out, out2           # scores, previous sequence from which corresponding index
                else:
                    return out
            else:
                return self.log_sum_exp(
                    previous + obs + pairwise_transition + self.transition_matrix,
#                     previous + obs + pairwise_transition,
                    axis=0
                )

        initial = observations[0]
        self.alpha, _ = theano.scan(
            fn=recurrence,
            sequences=[observations[1:], transitions],
            outputs_info=(initial, None) if return_best_sequence else initial
        )

        if return_alpha:
            return self.alpha
        elif return_best_sequence:
            sequence, _ = theano.scan(                
                fn=lambda beta_i, previous: beta_i[previous],     # back tracking
                sequences=T.cast(self.alpha[1][::-1], 'int32'),    # previous sequence
                outputs_info=T.cast(T.argmax(self.alpha[0][-1]), 'int32')   # last index of sequence having maximum score
            )
            sequence = T.concatenate([
                sequence[::-1],
                [T.argmax(self.alpha[0][-1])]
            ])
            return sequence
        else:
            if viterbi:
                return self.alpha[-1].max(axis=0)
            else:
                return self.log_sum_exp(self.alpha[-1], axis=0)

    def fprop(
        self,
        input_sequence,
        transition_input,
        ground_truth,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False,
        mode='train'
    ):
        """Propogate input through the CRF."""
        seq_length = input_sequence.shape[0]
        if mode != 'train':
            return self.alpha_recursion(
                input_sequence,
                transition_input,
                viterbi,
                return_alpha,
                return_best_sequence
            )
        real_path_score = input_sequence[T.arange(seq_length), ground_truth].sum()
        real_path_score += self.transition_matrix[
            ground_truth[T.arange(seq_length - 1)],
            ground_truth[T.arange(seq_length - 1) + 1]
        ].sum()
        
        def transition_scoring(transition_input, from_id, to_id):
            pairwise_transition = transition_input.reshape((self.n_tags, self.n_tags)) 
            score = pairwise_transition[from_id, to_id] 
            return score

        transition_scores, _ = theano.scan(fn=transition_scoring, 
                                           sequences=[transition_input,
                                                      ground_truth[T.arange(seq_length - 1)], 
                                                      ground_truth[T.arange(seq_length - 1) + 1]]
                                           )
        real_path_score += transition_scores.sum()
        all_paths_scores = self.alpha_recursion(
            input_sequence,
            transition_input,
            viterbi,
            return_alpha,
            return_best_sequence
        )
        
        cost = - (real_path_score - all_paths_scores)
        
        return cost

class CRF_Focus_PP_Fx(object):
    """Conditional Random Field Pairwise Potential."""

    def __init__(self, n_tags):
        self.n_tags = n_tags
        _transition_matrix = np.random.rand(self.n_tags, self.n_tags)
        self.transition_matrix = theano.shared(_transition_matrix.astype(theano.config.floatX), borrow=True, name='A')
        self.params = [self.transition_matrix]

        print 'Initialize CRFPP_Focused for BC5CDR'
    
    def log_sum_exp(self, x, axis=None):
        """Sum probabilities in the log-space."""
        xmax = x.max(axis=axis, keepdims=True)
        xmax_ = x.max(axis=axis)
        return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))

    def alpha_recursion(
        self,
        observations,
        transitions,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False
    ):
        assert not return_best_sequence or (viterbi and not return_alpha)
                
        def recurrence(obs, transition_input, previous, pairwise_transition):
            previous = previous.dimshuffle(0, 'x')
            obs = obs.dimshuffle('x', 0)
#             pairwise_transition1 = T.inc_subtensor(pairwise_transition[:,:], transition_input[8])
            pairwise_transition1 = T.set_subtensor(pairwise_transition[[0,0,0,0,0,1,1,2,2,3,4,5,5,6,6,7,8], 
                                                                       [0,1,4,5,8,2,3,2,3,0,0,6,7,6,7,0,0]], 
                                                                       transition_input)
            
            if viterbi:
                scores = previous + obs + pairwise_transition1
                out = scores.max(axis=0)
                if return_best_sequence:
                    out2 = scores.argmax(axis=0)    # previous index
                    return out, out2           # scores, previous sequence from which corresponding index
                else:
                    return out
            else:
                return self.log_sum_exp(
                    previous + obs + pairwise_transition1,
                    axis=0
                )

        initial = observations[0]
        self.alpha, _ = theano.scan(
            fn=recurrence,
            sequences=[observations[1:], transitions],
            outputs_info=(initial, None) if return_best_sequence else initial,
            non_sequences=self.transition_matrix
        )

        if return_alpha:
            return self.alpha
        elif return_best_sequence:
            sequence, _ = theano.scan(                
                fn=lambda beta_i, previous: beta_i[previous],     # back tracking
                sequences=T.cast(self.alpha[1][::-1], 'int32'),    # previous sequence
                outputs_info=T.cast(T.argmax(self.alpha[0][-1]), 'int32')   # last index of sequence having maximum score
            )
            sequence = T.concatenate([
                sequence[::-1],
                [T.argmax(self.alpha[0][-1])]
            ])
            return sequence
        else:
            if viterbi:
                return self.alpha[-1].max(axis=0)
            else:
                return self.log_sum_exp(self.alpha[-1], axis=0)

    def fprop(
        self,
        input_sequence,
        transition_input,
        ground_truth,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False,
        mode='train'
    ):
        """Propogate input through the CRF."""
        seq_length = input_sequence.shape[0]
        if mode != 'train':
            return self.alpha_recursion(
                input_sequence,
                transition_input,
                viterbi,
                return_alpha,
                return_best_sequence
            )
        real_path_score = input_sequence[T.arange(seq_length), ground_truth].sum()
        
        def transition_scoring(transition_input, from_id, to_id, pairwise_transition):
#             pairwise_transition1 = T.inc_subtensor(pairwise_transition[:,:], transition_input[8])
#             pairwise_transition2 = T.inc_subtensor(pairwise_transition1[[0,0,0,0,3,4,7,8],[1,4,5,8,0,0,0,0]], transition_input[:8])
            return T.set_subtensor(pairwise_transition[[0,0,0,0,0,1,1,2,2,3,4,5,5,6,6,7,8], 
                                                       [0,1,4,5,8,2,3,2,3,0,0,6,7,6,7,0,0]], 
                                                       transition_input)[from_id, to_id]
 
        transition_scores, _ = theano.scan(fn=transition_scoring, 
                                           sequences=[transition_input,
                                                      ground_truth[T.arange(seq_length - 1)], 
                                                      ground_truth[T.arange(seq_length - 1) + 1]],
                                           non_sequences=self.transition_matrix
                                           )
        real_path_score += transition_scores.sum()
        all_paths_scores = self.alpha_recursion(
            input_sequence,
            transition_input,
            viterbi,
            return_alpha,
            return_best_sequence
        )
        
        cost = - (real_path_score - all_paths_scores)
        
        return cost

class CRF_Focus_Only_PP(object):
    """Conditional Random Field Pairwise Potential."""

    def __init__(self, n_tags):
        self.n_tags = n_tags
        _transition_matrix = np.random.rand(self.n_tags, self.n_tags)
        self.transition_matrix = theano.shared(_transition_matrix.astype(theano.config.floatX), borrow=True, name='A')
        self.params = [self.transition_matrix]

        print 'Initialize CRFPP_Focused with Only PP'
    
    def log_sum_exp(self, x, axis=None):
        """Sum probabilities in the log-space."""
        xmax = x.max(axis=axis, keepdims=True)
        xmax_ = x.max(axis=axis)
        return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))

    def alpha_recursion(
        self,
        transitions,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False
    ):
        assert not return_best_sequence or (viterbi and not return_alpha)
                
        def recurrence(transition_input, previous, pairwise_transition):
            previous = previous.dimshuffle(0, 'x')
            pairwise_transition1 = T.set_subtensor(pairwise_transition[[0,0,0,0,0,1,1,2,2,3,4,5,5,6,6,7,8], 
                                                                       [0,1,4,5,8,2,3,2,3,0,0,6,7,6,7,0,0]], 
                                                                       transition_input)
            
            if viterbi:
                scores = previous + pairwise_transition1
                out = scores.max(axis=0)
                if return_best_sequence:
                    out2 = scores.argmax(axis=0)    # previous index
                    return out, out2           # scores, previous sequence from which corresponding index
                else:
                    return out
            else:
                return self.log_sum_exp(
                    previous + pairwise_transition1,
                    axis=0
                )

        initial = np.zeros((self.n_tags,), dtype=theano.config.floatX)
        self.alpha, _ = theano.scan(
            fn=recurrence,
            sequences=transitions,
            outputs_info=(initial, None) if return_best_sequence else initial,
            non_sequences=self.transition_matrix
        )

        if return_alpha:
            return self.alpha
        elif return_best_sequence:
            sequence, _ = theano.scan(                
                fn=lambda beta_i, previous: beta_i[previous],     # back tracking
                sequences=T.cast(self.alpha[1][::-1], 'int32'),    # previous sequence
                outputs_info=T.cast(T.argmax(self.alpha[0][-1]), 'int32')   # last index of sequence having maximum score
            )
            sequence = T.concatenate([
                sequence[::-1],
                [T.argmax(self.alpha[0][-1])]
            ])
            return sequence
        else:
            if viterbi:
                return self.alpha[-1].max(axis=0)
            else:
                return self.log_sum_exp(self.alpha[-1], axis=0)

    def sequence_predict(
        self,
        observations,
        transitions,
    ):
        def recurrence(obs, transition_input, previous, pairwise_transition):
            previous = previous.dimshuffle(0, 'x')
            obs = obs.dimshuffle('x', 0)
            pairwise_transition1 = T.set_subtensor(pairwise_transition[[0,0,0,0,0,1,1,2,2,3,4,5,5,6,6,7,8], 
                                                                       [0,1,4,5,8,2,3,2,3,0,0,6,7,6,7,0,0]], 
                                                                       transition_input)
            
            scores = previous + obs + pairwise_transition1
            out = scores.max(axis=0)
            out2 = scores.argmax(axis=0)    # previous index

            return out, out2           # scores, previous sequence from which corresponding index

        initial = observations[0]
        self.alpha, _ = theano.scan(
            fn=recurrence,
            sequences=[observations[1:], transitions],
            outputs_info=(initial, None),
            non_sequences=self.transition_matrix
        )

        sequence, _ = theano.scan(                
            fn=lambda beta_i, previous: beta_i[previous],     # back tracking
            sequences=T.cast(self.alpha[1][::-1], 'int32'),    # previous sequence
            outputs_info=T.cast(T.argmax(self.alpha[0][-1]), 'int32')   # last index of sequence having maximum score
        )
        
        sequence = T.concatenate([
            sequence[::-1],
            [T.argmax(self.alpha[0][-1])]
        ])

        return sequence

    def fprop(
        self,
        transition_input,
        ground_truth,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False,
        mode='train'
    ):
        """Propogate input through the CRF."""
        seq_length = transition_input.shape[0]
        if mode != 'train':
            return self.alpha_recursion(
                transition_input,
                viterbi,
                return_alpha,
                return_best_sequence
            )
        def transition_scoring(transition_input, from_id, to_id, pairwise_transition):
            return T.set_subtensor(pairwise_transition[[0,0,0,0,0,1,1,2,2,3,4,5,5,6,6,7,8], 
                                                       [0,1,4,5,8,2,3,2,3,0,0,6,7,6,7,0,0]], 
                                                       transition_input)[from_id, to_id]
 
        transition_scores, _ = theano.scan(fn=transition_scoring, 
                                           sequences=[transition_input,
                                                      ground_truth[T.arange(seq_length)], 
                                                      ground_truth[T.arange(seq_length) + 1]],
                                           non_sequences=self.transition_matrix
                                           )
        real_path_score = transition_scores.sum()
        all_paths_scores = self.alpha_recursion(
            transition_input,
            viterbi,
            return_alpha,
            return_best_sequence
        )
        
        cost = - (real_path_score - all_paths_scores)
        
        return cost

class CRF_Focus_Only_PP_Full(object):
    """Conditional Random Field Pairwise Potential."""

    def __init__(self, n_tags):
        self.n_tags = n_tags

        print 'Initialize CRFPP_Focused with Only PP Full'
    
    def log_sum_exp(self, x, axis=None):
        """Sum probabilities in the log-space."""
        xmax = x.max(axis=axis, keepdims=True)
        xmax_ = x.max(axis=axis)
        return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))

    def alpha_recursion(
        self,
        transitions,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False
    ):
        assert not return_best_sequence or (viterbi and not return_alpha)
                
        def recurrence(transition_input, previous):
            previous = previous.dimshuffle(0, 'x')
            pairwise_transition = transition_input.reshape((self.n_tags, self.n_tags))
            
            if viterbi:
                scores = previous + pairwise_transition
                out = scores.max(axis=0)
                if return_best_sequence:
                    out2 = scores.argmax(axis=0)    # previous index
                    return out, out2           # scores, previous sequence from which corresponding index
                else:
                    return out
            else:
                return self.log_sum_exp(
                    previous + pairwise_transition,
                    axis=0
                )

        initial = np.zeros((self.n_tags,), dtype=theano.config.floatX)
        self.alpha, _ = theano.scan(
            fn=recurrence,
            sequences=transitions,
            outputs_info=(initial, None) if return_best_sequence else initial
        )

        if return_alpha:
            return self.alpha
        elif return_best_sequence:
            sequence, _ = theano.scan(                
                fn=lambda beta_i, previous: beta_i[previous],     # back tracking
                sequences=T.cast(self.alpha[1][::-1], 'int32'),    # previous sequence
                outputs_info=T.cast(T.argmax(self.alpha[0][-1]), 'int32')   # last index of sequence having maximum score
            )
            sequence = T.concatenate([
                sequence[::-1],
                [T.argmax(self.alpha[0][-1])]
            ])
            return sequence
        else:
            if viterbi:
                return self.alpha[-1].max(axis=0)
            else:
                return self.log_sum_exp(self.alpha[-1], axis=0)

    def sequence_predict(
        self,
        observations,
        transitions,
        transition_matrix
    ):
        def recurrence(obs, transition_input, previous, transition_matrix):
            previous = previous.dimshuffle(0, 'x')
            obs = obs.dimshuffle('x', 0)
            pairwise_transition = transition_input.reshape((self.n_tags, self.n_tags))
                        
            scores = previous + obs + pairwise_transition + transition_matrix
            out = scores.max(axis=0)
            out2 = scores.argmax(axis=0)    # previous index

            return out, out2           # scores, previous sequence from which corresponding index

        initial = observations[0]
        self.alpha, _ = theano.scan(
            fn=recurrence,
            sequences=[observations[1:], transitions],
            outputs_info=(initial, None),
            non_sequences=transition_matrix
        )

        sequence, _ = theano.scan(                
            fn=lambda beta_i, previous: beta_i[previous],     # back tracking
            sequences=T.cast(self.alpha[1][::-1], 'int32'),    # previous sequence
            outputs_info=T.cast(T.argmax(self.alpha[0][-1]), 'int32')   # last index of sequence having maximum score
        )
        
        sequence = T.concatenate([
            sequence[::-1],
            [T.argmax(self.alpha[0][-1])]
        ])

        return sequence

    def fprop(
        self,
        transition_input,
        ground_truth,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False,
        mode='train'
    ):
        """Propogate input through the CRF."""
        seq_length = transition_input.shape[0]
        if mode != 'train':
            return self.alpha_recursion(
                transition_input,
                viterbi,
                return_alpha,
                return_best_sequence
            )
        def transition_scoring(transition_input, from_id, to_id):
            return transition_input.reshape((self.n_tags, self.n_tags))[from_id, to_id]
 
        transition_scores, _ = theano.scan(fn=transition_scoring, 
                                           sequences=[transition_input,
                                                      ground_truth[T.arange(seq_length)], 
                                                      ground_truth[T.arange(seq_length) + 1]]
                                           )
        real_path_score = transition_scores.sum()
        all_paths_scores = self.alpha_recursion(
            transition_input,
            viterbi,
            return_alpha,
            return_best_sequence
        )
        
        cost = - (real_path_score - all_paths_scores)
        
        return cost

class CRF_Focus_PP_OneType(object):
    """Conditional Random Field Pairwise Potential."""

    def __init__(self, n_tags):
        self.n_tags = n_tags
        _transition_matrix = np.random.rand(self.n_tags, self.n_tags)
        self.transition_matrix = theano.shared(_transition_matrix.astype(theano.config.floatX), borrow=True, name='A')
        self.params = [self.transition_matrix]

        print 'Initialize CRFPP_Focused for One Type'
    
    def log_sum_exp(self, x, axis=None):
        """Sum probabilities in the log-space."""
        xmax = x.max(axis=axis, keepdims=True)
        xmax_ = x.max(axis=axis)
        return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))

    def alpha_recursion(
        self,
        transitions,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False
    ):
        assert not return_best_sequence or (viterbi and not return_alpha)
                
        def recurrence(transition_input, previous, pairwise_transition):
            previous = previous.dimshuffle(0, 'x')
            pairwise_transition1 = T.set_subtensor(pairwise_transition[[0,0,0,1,1,2,2,3,4], 
                                                                       [0,1,4,2,3,2,3,0,0]], 
                                                                       transition_input)
            
            if viterbi:
                scores = previous + pairwise_transition1
                out = scores.max(axis=0)
                if return_best_sequence:
                    out2 = scores.argmax(axis=0)    # previous index
                    return out, out2           # scores, previous sequence from which corresponding index
                else:
                    return out
            else:
                return self.log_sum_exp(
                    previous + pairwise_transition1,
                    axis=0
                )

        initial = np.zeros((self.n_tags,), dtype=theano.config.floatX)
        self.alpha, _ = theano.scan(
            fn=recurrence,
            sequences=transitions,
            outputs_info=(initial, None) if return_best_sequence else initial,
            non_sequences=self.transition_matrix
        )

        if return_alpha:
            return self.alpha
        elif return_best_sequence:
            sequence, _ = theano.scan(                
                fn=lambda beta_i, previous: beta_i[previous],     # back tracking
                sequences=T.cast(self.alpha[1][::-1], 'int32'),    # previous sequence
                outputs_info=T.cast(T.argmax(self.alpha[0][-1]), 'int32')   # last index of sequence having maximum score
            )
            sequence = T.concatenate([
                sequence[::-1],
                [T.argmax(self.alpha[0][-1])]
            ])
            return sequence
        else:
            if viterbi:
                return self.alpha[-1].max(axis=0)
            else:
                return self.log_sum_exp(self.alpha[-1], axis=0)

    def sequence_predict(
        self,
        observations,
        transitions
    ):
        def recurrence(obs, transition_input, previous, pairwise_transition):
            previous = previous.dimshuffle(0, 'x')
            obs = obs.dimshuffle('x', 0)
            pairwise_transition1 = T.set_subtensor(pairwise_transition[[0,0,0,1,1,2,2,3,4], 
                                                                       [0,1,4,2,3,2,3,0,0]], 
                                                                       transition_input)
            
            scores = previous + obs + pairwise_transition1
            out = scores.max(axis=0)
            out2 = scores.argmax(axis=0)    # previous index

            return out, out2           # scores, previous sequence from which corresponding index

        initial = observations[0]
        self.alpha, _ = theano.scan(
            fn=recurrence,
            sequences=[observations[1:], transitions],
            outputs_info=(initial, None),
            non_sequences=self.transition_matrix
        )

        sequence, _ = theano.scan(                
            fn=lambda beta_i, previous: beta_i[previous],     # back tracking
            sequences=T.cast(self.alpha[1][::-1], 'int32'),    # previous sequence
            outputs_info=T.cast(T.argmax(self.alpha[0][-1]), 'int32')   # last index of sequence having maximum score
        )
        
        sequence = T.concatenate([
            sequence[::-1],
            [T.argmax(self.alpha[0][-1])]
        ])

        return sequence

    def fprop(
        self,
        transition_input,
        ground_truth,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False,
        mode='train'
    ):
        """Propogate input through the CRF."""
        seq_length = transition_input.shape[0]
        
        def transition_scoring(transition_input, from_id, to_id, pairwise_transition):
            return T.set_subtensor(pairwise_transition[[0,0,0,1,1,2,2,3,4], 
                                                       [0,1,4,2,3,2,3,0,0]], 
                                                       transition_input)[from_id, to_id]
 
        transition_scores, _ = theano.scan(fn=transition_scoring, 
                                           sequences=[transition_input,
                                                      ground_truth[T.arange(seq_length)], 
                                                      ground_truth[T.arange(seq_length) + 1]],
                                           non_sequences=self.transition_matrix
                                           )
        
        real_path_score = transition_scores.sum()
        all_paths_scores = self.alpha_recursion(
            transition_input,
            viterbi,
            return_alpha,
            return_best_sequence
        )
        
        cost = - (real_path_score - all_paths_scores)
        
        return cost

class CRF_LOP_OneType(object):
    """Conditional Random Field Pairwise Potential."""

    def __init__(self, n_tags):
        self.n_tags = n_tags
        _transition_matrix = np.random.rand(self.n_tags, self.n_tags)
        self.transition_matrix = theano.shared(_transition_matrix.astype(theano.config.floatX), borrow=True, name='A')
        self.params = [self.transition_matrix]

        print 'Initialize CRFPP_Focused for One Type'
    
    def log_sum_exp(self, x, axis=None):
        """Sum probabilities in the log-space."""
        xmax = x.max(axis=axis, keepdims=True)
        xmax_ = x.max(axis=axis)
        return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))

    def alpha_recursion_u(
        self,
        observations,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False
    ):
        assert not return_best_sequence or (viterbi and not return_alpha)
                
        def recurrence(obs, previous):
            previous = previous.dimshuffle(0, 'x')
            obs = obs.dimshuffle('x', 0)
            
            return self.log_sum_exp(
                previous + obs,
                axis=0
            )

        initial = observations[0]
        self.alpha, _ = theano.scan(
            fn=recurrence,
            sequences=observations[1:],
            outputs_info=(initial, None) if return_best_sequence else initial
        )

        return self.log_sum_exp(self.alpha[-1], axis=0)

    def alpha_recursion_pp(
        self,
        transitions,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False
    ):
        assert not return_best_sequence or (viterbi and not return_alpha)
                
        def recurrence(transition_input, previous, pairwise_transition):
            previous = previous.dimshuffle(0, 'x')
            pairwise_transition1 = T.set_subtensor(pairwise_transition[[0,0,0,1,1,2,2,3,4], 
                                                                       [0,1,4,2,3,2,3,0,0]], 
                                                                       transition_input)
            
            return self.log_sum_exp(
                previous + pairwise_transition1,
                axis=0
            )

        initial = np.zeros((self.n_tags,), dtype=theano.config.floatX)
        self.alpha, _ = theano.scan(
            fn=recurrence,
            sequences=transitions,
            outputs_info=(initial, None) if return_best_sequence else initial,
            non_sequences=self.transition_matrix
        )

        return self.log_sum_exp(self.alpha[-1], axis=0)

    def alpha_recursion_lop(
        self,
        observations,
        transitions,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False
    ):                
        def recurrence(observation, transition_input, previous, pairwise_transition):
            previous = previous.dimshuffle(0, 'x')
            observation = observation.dimshuffle('x', 0)
            pairwise_transition1 = T.set_subtensor(pairwise_transition[[0,0,0,1,1,2,2,3,4], 
                                                                       [0,1,4,2,3,2,3,0,0]], 
                                                                       transition_input)
            return self.log_sum_exp(
                    previous + observation + pairwise_transition1,
                    axis=0
            )

        initial = observations[0]
        self.alpha, _ = theano.scan(
            fn=recurrence,
            sequences=[observations[1:], transitions],
            outputs_info=initial,
            non_sequences=self.transition_matrix
        )

        return self.log_sum_exp(self.alpha[-1], axis=0)

    def sequence_predict(
        self,
        observations,
        transitions
    ):
        def recurrence(obs, transition_input, previous, pairwise_transition):
            previous = previous.dimshuffle(0, 'x')
            obs = obs.dimshuffle('x', 0)
            pairwise_transition1 = T.set_subtensor(pairwise_transition[[0,0,0,1,1,2,2,3,4], 
                                                                       [0,1,4,2,3,2,3,0,0]], 
                                                                       transition_input)
            
            scores = previous + obs + pairwise_transition1
            out = scores.max(axis=0)
            out2 = scores.argmax(axis=0)    # previous index

            return out, out2           # scores, previous sequence from which corresponding index

        initial = observations[0]
        self.alpha, _ = theano.scan(
            fn=recurrence,
            sequences=[observations[1:], transitions],
            outputs_info=(initial, None),
            non_sequences=self.transition_matrix
        )

        sequence, _ = theano.scan(                
            fn=lambda beta_i, previous: beta_i[previous],     # back tracking
            sequences=T.cast(self.alpha[1][::-1], 'int32'),    # previous sequence
            outputs_info=T.cast(T.argmax(self.alpha[0][-1]), 'int32')   # last index of sequence having maximum score
        )
        
        sequence = T.concatenate([
            sequence[::-1],
            [T.argmax(self.alpha[0][-1])]
        ])

        return sequence

    def fprop(
        self,
        input_sequence,
        transition_input,
        ground_truth,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False,
        mode='train'
    ):
        """Propogate input through the CRF."""
        seq_length_u = input_sequence.shape[0]
        seq_length_pp = transition_input.shape[0]
        
        def transition_scoring(transition_input, from_id, to_id, pairwise_transition):
            return T.set_subtensor(pairwise_transition[[0,0,0,1,1,2,2,3,4], 
                                                       [0,1,4,2,3,2,3,0,0]], 
                                                       transition_input)[from_id, to_id]
 
        transition_scores, _ = theano.scan(fn=transition_scoring, 
                                           sequences=[transition_input,
                                                      ground_truth[T.arange(seq_length_pp)], 
                                                      ground_truth[T.arange(seq_length_pp) + 1]],
                                           non_sequences=self.transition_matrix
                                           )
        real_path_score_u = input_sequence[T.arange(seq_length_u), ground_truth].sum()
        all_paths_scores_u = self.alpha_recursion_u(
            input_sequence,
            viterbi,
            return_alpha,
            return_best_sequence
        )
        cost_u = - (real_path_score_u - all_paths_scores_u)

        real_path_score_pp = transition_scores.sum()
        all_paths_scores_pp = self.alpha_recursion_pp(
            transition_input,
            viterbi,
            return_alpha,
            return_best_sequence
        )
        
#         all_paths_scores_lop = self.alpha_recursion_lop(
#             input_sequence,
#             transition_input,
#             viterbi,
#             return_alpha,
#             return_best_sequence
#         )

        cost_pp = - (real_path_score_pp - all_paths_scores_pp)
        
        return cost_u + cost_pp, cost_pp

class CRF_LOP_TwoType(object):
    """Conditional Random Field Pairwise Potential."""

    def __init__(self, n_tags):
        self.n_tags = n_tags
        _transition_matrix = np.random.rand(self.n_tags, self.n_tags)
        self.transition_matrix = theano.shared(_transition_matrix.astype(theano.config.floatX), borrow=True, name='A')
        self.params = [self.transition_matrix]

        print 'Initialize PoE for Two Type'
    
    def log_sum_exp(self, x, axis=None):
        """Sum probabilities in the log-space."""
        xmax = x.max(axis=axis, keepdims=True)
        xmax_ = x.max(axis=axis)
        return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))

    def alpha_recursion_u(
        self,
        observations,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False
    ):
        assert not return_best_sequence or (viterbi and not return_alpha)
                
        def recurrence(obs, previous):
            previous = previous.dimshuffle(0, 'x')
            obs = obs.dimshuffle('x', 0)
            
            return self.log_sum_exp(
                previous + obs,
                axis=0
            )

        initial = observations[0]
        self.alpha, _ = theano.scan(
            fn=recurrence,
            sequences=observations[1:],
            outputs_info=(initial, None) if return_best_sequence else initial
        )

        return self.log_sum_exp(self.alpha[-1], axis=0)

    def alpha_recursion_pp(
        self,
        transitions,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False
    ):
        assert not return_best_sequence or (viterbi and not return_alpha)
                
        def recurrence(transition_input, previous, pairwise_transition):
            previous = previous.dimshuffle(0, 'x')
            pairwise_transition1 = T.set_subtensor(pairwise_transition[[0,0,0,0,0,1,1,2,2,3,4,5,5,6,6,7,8], 
                                                                       [0,1,4,5,8,2,3,2,3,0,0,6,7,6,7,0,0]], 
                                                                       transition_input)
            
            return self.log_sum_exp(
                previous + pairwise_transition1,
                axis=0
            )

        initial = np.zeros((self.n_tags,), dtype=theano.config.floatX)
        self.alpha, _ = theano.scan(
            fn=recurrence,
            sequences=transitions,
            outputs_info=(initial, None) if return_best_sequence else initial,
            non_sequences=self.transition_matrix
        )

        return self.log_sum_exp(self.alpha[-1], axis=0)

    def alpha_recursion_lop(
        self,
        observations,
        transitions,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False
    ):                
        def recurrence(observation, transition_input, previous, pairwise_transition):
            previous = previous.dimshuffle(0, 'x')
            observation = observation.dimshuffle('x', 0)
            pairwise_transition1 = T.set_subtensor(pairwise_transition[[0,0,0,0,0,1,1,2,2,3,4,5,5,6,6,7,8], 
                                                                       [0,1,4,5,8,2,3,2,3,0,0,6,7,6,7,0,0]], 
                                                                       transition_input)
            return self.log_sum_exp(
                    previous + observation + pairwise_transition1,
                    axis=0
            )

        initial = observations[0]
        self.alpha, _ = theano.scan(
            fn=recurrence,
            sequences=[observations[1:], transitions],
            outputs_info=initial,
            non_sequences=self.transition_matrix
        )

        return self.log_sum_exp(self.alpha[-1], axis=0)

    def sequence_predict(
        self,
        observations,
        transitions
    ):
        def recurrence(obs, transition_input, previous, pairwise_transition):
            previous = previous.dimshuffle(0, 'x')
            obs = obs.dimshuffle('x', 0)
            pairwise_transition1 = T.set_subtensor(pairwise_transition[[0,0,0,0,0,1,1,2,2,3,4,5,5,6,6,7,8], 
                                                                       [0,1,4,5,8,2,3,2,3,0,0,6,7,6,7,0,0]], 
                                                                       transition_input)
            
            scores = previous + obs + pairwise_transition1
            out = scores.max(axis=0)
            out2 = scores.argmax(axis=0)    # previous index

            return out, out2           # scores, previous sequence from which corresponding index

        initial = observations[0]
        self.alpha, _ = theano.scan(
            fn=recurrence,
            sequences=[observations[1:], transitions],
            outputs_info=(initial, None),
            non_sequences=self.transition_matrix
        )

        sequence, _ = theano.scan(                
            fn=lambda beta_i, previous: beta_i[previous],     # back tracking
            sequences=T.cast(self.alpha[1][::-1], 'int32'),    # previous sequence
            outputs_info=T.cast(T.argmax(self.alpha[0][-1]), 'int32')   # last index of sequence having maximum score
        )
        
        sequence = T.concatenate([
            sequence[::-1],
            [T.argmax(self.alpha[0][-1])]
        ])

        return sequence

    def fprop(
        self,
        input_sequence,
        transition_input,
        ground_truth,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False,
        mode='train'
    ):
        """Propogate input through the CRF."""
        seq_length_u = input_sequence.shape[0]
        seq_length_pp = transition_input.shape[0]
        
        def transition_scoring(transition_input, from_id, to_id, pairwise_transition):
            return T.set_subtensor(pairwise_transition[[0,0,0,0,0,1,1,2,2,3,4,5,5,6,6,7,8], 
                                                       [0,1,4,5,8,2,3,2,3,0,0,6,7,6,7,0,0]], 
                                                       transition_input)[from_id, to_id]
 
        transition_scores, _ = theano.scan(fn=transition_scoring, 
                                           sequences=[transition_input,
                                                      ground_truth[T.arange(seq_length_pp)], 
                                                      ground_truth[T.arange(seq_length_pp) + 1]],
                                           non_sequences=self.transition_matrix
                                           )
        real_path_score_u = input_sequence[T.arange(seq_length_u), ground_truth].sum()
        all_paths_scores_u = self.alpha_recursion_u(
            input_sequence,
            viterbi,
            return_alpha,
            return_best_sequence
        )
#         cost_u = - (real_path_score_u)

        real_path_score_pp = transition_scores.sum()
        all_paths_scores_pp = self.alpha_recursion_pp(
            transition_input,
            viterbi,
            return_alpha,
            return_best_sequence
        )
        
        all_paths_scores_lop = self.alpha_recursion_lop(
            input_sequence,
            transition_input,
            viterbi,
            return_alpha,
            return_best_sequence
        )

        cost_lop = - (0.5*real_path_score_u + 0.5*real_path_score_pp - all_paths_scores_lop - 0.5*all_paths_scores_u - 0.5*all_paths_scores_pp)
        
        return cost_lop, all_paths_scores_u, all_paths_scores_pp, all_paths_scores_lop

class CRF_Only_U(object):
    """Conditional Random Field Pairwise Potential."""

    def __init__(self, n_tags):
        self.n_tags = n_tags
        _transition_matrix = np.random.rand(self.n_tags, self.n_tags)
        self.transition_matrix = theano.shared(_transition_matrix.astype(theano.config.floatX), borrow=True, name='A')
        self.params = [self.transition_matrix]

        print 'Initialize CRF_Only_U'
    
    def log_sum_exp(self, x, axis=None):
        """Sum probabilities in the log-space."""
        xmax = x.max(axis=axis, keepdims=True)
        xmax_ = x.max(axis=axis)
        return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))

    def alpha_recursion(
        self,
        observations,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False
    ):
        assert not return_best_sequence or (viterbi and not return_alpha)
                
        def recurrence(obs, previous):
            previous = previous.dimshuffle(0, 'x')
            obs = obs.dimshuffle('x', 0)
            
            if viterbi:
                scores = previous + obs + self.transition_matrix
                out = scores.max(axis=0)
                if return_best_sequence:
                    out2 = scores.argmax(axis=0)    # previous index
                    return out, out2           # scores, previous sequence from which corresponding index
                else:
                    return out
            else:
                return self.log_sum_exp(
                    previous + obs + self.transition_matrix,
                    axis=0
                )

        initial = observations[0]
        self.alpha, _ = theano.scan(
            fn=recurrence,
            sequences=observations[1:],
            outputs_info=(initial, None) if return_best_sequence else initial
        )

        if return_alpha:
            return self.alpha
        elif return_best_sequence:
            sequence, _ = theano.scan(                
                fn=lambda beta_i, previous: beta_i[previous],     # back tracking
                sequences=T.cast(self.alpha[1][::-1], 'int32'),    # previous sequence
                outputs_info=T.cast(T.argmax(self.alpha[0][-1]), 'int32')   # last index of sequence having maximum score
            )
            sequence = T.concatenate([
                sequence[::-1],
                [T.argmax(self.alpha[0][-1])]
            ])
            return sequence
        else:
            if viterbi:
                return self.alpha[-1].max(axis=0)
            else:
                return self.log_sum_exp(self.alpha[-1], axis=0)

    def fprop(
        self,
        input_sequence,
        ground_truth,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False,
        mode='train'
    ):
        """Propogate input through the CRF."""
        seq_length = input_sequence.shape[0]
        if mode != 'train':
            return self.alpha_recursion(
                input_sequence,
                viterbi,
                return_alpha,
                return_best_sequence
            )
        real_path_score = input_sequence[T.arange(seq_length), ground_truth].sum()
        real_path_score += self.transition_matrix[
            ground_truth[T.arange(seq_length - 1)],
            ground_truth[T.arange(seq_length - 1) + 1]
        ].sum()

        all_paths_scores = self.alpha_recursion(
            input_sequence,
            viterbi,
            return_alpha,
            return_best_sequence
        )
        
        cost = - (real_path_score - all_paths_scores)
        
        return cost

class CRF_U(object):
    """Conditional Random Field Pairwise Potential."""

    def __init__(self, n_tags):
        self.n_tags = n_tags
        _transition_matrix = np.random.rand(self.n_tags, self.n_tags)
        self.transition_matrix = theano.shared(_transition_matrix.astype(theano.config.floatX), borrow=True)
        self.params = [self.transition_matrix]

        print 'Initialize CRF_U'
    
    def log_sum_exp(self, x, axis=None):
        """Sum probabilities in the log-space."""
        xmax = x.max(axis=axis, keepdims=True)
        xmax_ = x.max(axis=axis)
        return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))

    def alpha_recursion(
        self,
        observations,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False
    ):
        assert not return_best_sequence or (viterbi and not return_alpha)
                
        def recurrence(obs, previous):
            previous = previous.dimshuffle(0, 'x')
            obs = obs.dimshuffle('x', 0)
            
            if viterbi:
                scores = previous + obs + self.transition_matrix
                out = scores.max(axis=0)
                if return_best_sequence:
                    out2 = scores.argmax(axis=0)    # previous index
                    return out, out2           # scores, previous sequence from which corresponding index
                else:
                    return out
            else:
                return self.log_sum_exp(
                    previous + obs + self.transition_matrix,
                    axis=0
                )

        initial = observations[0]
        self.alpha, _ = theano.scan(
            fn=recurrence,
            sequences=observations[1:],
            outputs_info=(initial, None) if return_best_sequence else initial
        )

        if return_alpha:
            return self.alpha
        elif return_best_sequence:
            sequence, _ = theano.scan(                
                fn=lambda beta_i, previous: beta_i[previous],     # back tracking
                sequences=T.cast(self.alpha[1][::-1], 'int32'),    # previous sequence
                outputs_info=T.cast(T.argmax(self.alpha[0][-1]), 'int32')   # last index of sequence having maximum score
            )
            sequence = T.concatenate([
                sequence[::-1],
                [T.argmax(self.alpha[0][-1])]
            ])
            return sequence
        else:
            if viterbi:
                return self.alpha[-1].max(axis=0)
            else:
                return self.log_sum_exp(self.alpha[-1], axis=0)

    def sequence_predict(
        self,
        observations1,
        observations2,
        transition_matrix
    ):
        def recurrence(obs1, obs2, previous, transition_matrix):
            previous = previous.dimshuffle(0, 'x')
            obs1 = obs1.dimshuffle('x', 0)
            obs2 = obs2.dimshuffle('x', 0)
                        
            scores = previous + obs1 + obs2 + transition_matrix + self.transition_matrix
            out = scores.max(axis=0)
            out2 = scores.argmax(axis=0)    # previous index

            return out, out2           # scores, previous sequence from which corresponding index

        initial = observations1[0] + observations2[0]
        self.alpha, _ = theano.scan(
            fn=recurrence,
            sequences=[observations1[1:],observations2[1:]],
            outputs_info=(initial, None),
            non_sequences=transition_matrix
        )

        sequence, _ = theano.scan(                
            fn=lambda beta_i, previous: beta_i[previous],     # back tracking
            sequences=T.cast(self.alpha[1][::-1], 'int32'),    # previous sequence
            outputs_info=T.cast(T.argmax(self.alpha[0][-1]), 'int32')   # last index of sequence having maximum score
        )
        
        sequence = T.concatenate([
            sequence[::-1],
            [T.argmax(self.alpha[0][-1])]
        ])

        return sequence

    def sequence_predict_three_unary(
        self,
        observations1,
        observations2,
        observations3,
        transition_matrix1,
        transition_matrix2
    ):
        def recurrence(obs1, obs2, obs3, previous, transition_matrix1, transition_matrix2):
            previous = previous.dimshuffle(0, 'x')
            obs1 = obs1.dimshuffle('x', 0)
            obs2 = obs2.dimshuffle('x', 0)
            obs3 = obs3.dimshuffle('x', 0)
                        
            scores = previous + obs1 + obs2 + obs3 + transition_matrix1 + transition_matrix2 + self.transition_matrix
            out = scores.max(axis=0)
            out2 = scores.argmax(axis=0)    # previous index

            return out, out2           # scores, previous sequence from which corresponding index

        initial = observations1[0] + observations2[0] + observations3[0]
        self.alpha, _ = theano.scan(
            fn=recurrence,
            sequences=[observations1[1:],observations2[1:],observations3[1:]],
            outputs_info=(initial, None),
            non_sequences=[transition_matrix1, transition_matrix2]
        )

        sequence, _ = theano.scan(                
            fn=lambda beta_i, previous: beta_i[previous],     # back tracking
            sequences=T.cast(self.alpha[1][::-1], 'int32'),    # previous sequence
            outputs_info=T.cast(T.argmax(self.alpha[0][-1]), 'int32')   # last index of sequence having maximum score
        )
        
        sequence = T.concatenate([
            sequence[::-1],
            [T.argmax(self.alpha[0][-1])]
        ])

        return sequence

    def fprop(
        self,
        input_sequence,
        ground_truth,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False,
        mode='train'
    ):
        """Propogate input through the CRF."""
        seq_length = input_sequence.shape[0]
        if mode != 'train':
            return self.alpha_recursion(
                input_sequence,
                viterbi,
                return_alpha,
                return_best_sequence
            )
        real_path_score = input_sequence[T.arange(seq_length), ground_truth].sum()
        real_path_score += self.transition_matrix[
            ground_truth[T.arange(seq_length - 1)],
            ground_truth[T.arange(seq_length - 1) + 1]
        ].sum()

        all_paths_scores = self.alpha_recursion(
            input_sequence,
            viterbi,
            return_alpha,
            return_best_sequence
        )
        
        cost = - (real_path_score - all_paths_scores)
        
        return cost

class CRF_Only_U_Logistic_Regression(object):
    """Conditional Random Field Pairwise Potential."""

    def __init__(self, n_tags):
        self.n_tags = n_tags
        print 'Initialize CRF_Only_U'
    
    def log_sum_exp(self, x, axis=None):
        """Sum probabilities in the log-space."""
        xmax = x.max(axis=axis, keepdims=True)
        xmax_ = x.max(axis=axis)
        return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))

    def alpha_recursion(
        self,
        observations,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False
    ):
        assert not return_best_sequence or (viterbi and not return_alpha)
                
        def recurrence(obs):
            return self.log_sum_exp(obs, axis=0)

        self.alpha, _ = theano.scan(
            fn=recurrence,
            sequences=observations
        )

        return self.alpha.sum()

    def fprop(
        self,
        input_sequence,
        ground_truth,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False,
        mode='train'
    ):
        """Propogate input through the CRF."""
        seq_length = input_sequence.shape[0]
        if mode != 'train':
            return self.alpha_recursion(
                input_sequence,
                viterbi,
                return_alpha,
                return_best_sequence
            )
        real_path_score = input_sequence[T.arange(seq_length), ground_truth].sum()

        all_paths_scores = self.alpha_recursion(
            input_sequence,
            viterbi,
            return_alpha,
            return_best_sequence
        )
        
        cost = - (real_path_score - all_paths_scores)
        
        return cost

class CRF_Fx(object):
    """Conditional Random Field Pairwise Potential."""

    def __init__(self, n_tags):
        self.n_tags = n_tags
        
        _transition_matrix = np.random.rand(self.n_tags, self.n_tags)
        self.transition_matrix = theano.shared(_transition_matrix.astype(theano.config.floatX), borrow=True, name='A')

        self.params = [self.transition_matrix]

        print 'Initialize CRF_Fx'
    
    def log_sum_exp(self, x, axis=None):
        """Sum probabilities in the log-space."""
        xmax = x.max(axis=axis, keepdims=True)
        xmax_ = x.max(axis=axis)
        return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))

    def alpha_recursion(
        self,
        observations,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False
    ):
        assert not return_best_sequence or (viterbi and not return_alpha)
                
        def recurrence(obs, previous):
            previous = previous.dimshuffle(0, 'x')
            obs = obs.dimshuffle('x', 0)
            
            if viterbi:
                scores = previous + obs + self.transition_matrix
                out = scores.max(axis=0)
                if return_best_sequence:
                    out2 = scores.argmax(axis=0)    # previous index
                    return out, out2           # scores, previous sequence from which corresponding index
                else:
                    return out
            else:
                return self.log_sum_exp(
                    previous + obs + self.transition_matrix,
                    axis=0
                )

        initial = observations[0]
        self.alpha, _ = theano.scan(
            fn=recurrence,
            sequences=observations[1:],
            outputs_info=(initial, None) if return_best_sequence else initial
        )

        if return_alpha:
            return self.alpha
        elif return_best_sequence:
            sequence, _ = theano.scan(                
                fn=lambda beta_i, previous: beta_i[previous],     # back tracking
                sequences=T.cast(self.alpha[1][::-1], 'int32'),    # previous sequence
                outputs_info=T.cast(T.argmax(self.alpha[0][-1]), 'int32')   # last index of sequence having maximum score
            )
            sequence = T.concatenate([
                sequence[::-1],
                [T.argmax(self.alpha[0][-1])]
            ])
            return sequence
        else:
            if viterbi:
                return self.alpha[-1].max(axis=0)
            else:
                return self.log_sum_exp(self.alpha[-1], axis=0)

    def fprop(
        self,
        input_sequence,
        ground_truth,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False,
        mode='train'
    ):
        """Propogate input through the CRF."""
        seq_length = input_sequence.shape[0]
        if mode != 'train':
            return self.alpha_recursion(
                input_sequence,
                viterbi,
                return_alpha,
                return_best_sequence
            )
        real_path_score = input_sequence[T.arange(seq_length), ground_truth].sum()
        real_path_score += self.transition_matrix[
            ground_truth[T.arange(seq_length - 1)],
            ground_truth[T.arange(seq_length - 1) + 1]
        ].sum()
        
        all_paths_scores = self.alpha_recursion(
            input_sequence,
            viterbi,
            return_alpha,
            return_best_sequence
        )
        
        cost = - (real_path_score - all_paths_scores)
        
        return cost

class CRF_For_PP_Fx(object):
    """Conditional Random Field Only For Pairwise Potential."""

    def __init__(self, n_tags):
        self.n_tags = n_tags
        print 'Initialize CRF_For_Only_PP_Fx'
    
    def log_sum_exp(self, x, axis=None):
        """Sum probabilities in the log-space."""
        xmax = x.max(axis=axis, keepdims=True)
        xmax_ = x.max(axis=axis)
        return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))

    def alpha_recursion(
        self,
        transitions,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False
    ):
        assert not return_best_sequence or (viterbi and not return_alpha)
                
        def recurrence(transition, previous):
            previous = previous.dimshuffle(0, 'x')
            pairwise_transition = transition.reshape((self.n_tags, self.n_tags))
            
            if viterbi:
                scores = previous + pairwise_transition
                out = scores.max(axis=0)
                if return_best_sequence:
                    out2 = scores.argmax(axis=0)    # previous index
                    return out, out2           # scores, previous sequence from which corresponding index
                else:
                    return out
            else:
                return self.log_sum_exp(
                    previous + pairwise_transition,
                    axis=0
                )
        
        initial = np.zeros((self.n_tags,), dtype=theano.config.floatX)
        self.alpha, _ = theano.scan(
            fn=recurrence,
            sequences=transitions,
            outputs_info=(initial, None) if return_best_sequence else initial
        )

        if return_alpha:
            return self.alpha
        elif return_best_sequence:
            sequence, _ = theano.scan(                
                fn=lambda beta_i, previous: beta_i[previous],     # back tracking
                sequences=T.cast(self.alpha[1][::-1], 'int32'),    # previous sequence
                outputs_info=T.cast(T.argmax(self.alpha[0][-1]), 'int32')   # last index of sequence having maximum score
            )
            sequence = T.concatenate([
                sequence[::-1],
                [T.argmax(self.alpha[0][-1])]
            ])
            return sequence
        else:
            if viterbi:
                return self.alpha[-1].max(axis=0)
            else:
                return self.log_sum_exp(self.alpha[-1], axis=0)

    def fprop(
        self,
        transitions,
        ground_truth,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False,
        mode='train'
    ):
        """Propogate input through the CRF."""
        seq_length = transitions.shape[0]
        if mode != 'train':
            return self.alpha_recursion(
                transitions,
                viterbi,
                return_alpha,
                return_best_sequence
            )
        
        def transition_scoring(transition, from_id, to_id):
            pairwise_transition = transition.reshape((self.n_tags, self.n_tags)) 
            score = pairwise_transition[from_id, to_id] 
            return score

        transition_scores, _ = theano.scan(fn=transition_scoring, 
                                           sequences=[transitions,
                                                      ground_truth[T.arange(seq_length)], 
                                                      ground_truth[T.arange(seq_length) + 1]]
                                           )

        real_path_score = transition_scores.sum()
        all_paths_scores = self.alpha_recursion(
            transitions,
            viterbi,
            return_alpha,
            return_best_sequence
        )
        
        cost = - (real_path_score - all_paths_scores)
        
        return cost

class CRF_U_PP_Fx(object):
    """Conditional Random Field Unary Potential and Pairwise Potential."""

    def __init__(self, n_tags):
        self.n_tags = n_tags
        self.params = []
        print 'Initialize CRFPP_Fx'
    
    def log_sum_exp(self, x, axis=None):
        """Sum probabilities in the log-space."""
        xmax = x.max(axis=axis, keepdims=True)
        xmax_ = x.max(axis=axis)
        return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))

    def alpha_recursion(
        self,
        observations,
        transitions,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False
    ):
        assert not return_best_sequence or (viterbi and not return_alpha)
                
        def recurrence(obs, transition, previous):
            previous = previous.dimshuffle(0, 'x')
            obs = obs.dimshuffle('x', 0)
            pp_transition = transition.reshape((self.n_tags, self.n_tags))
            
            if viterbi:
                scores = previous + obs + pp_transition
                out = scores.max(axis=0)
                if return_best_sequence:
                    out2 = scores.argmax(axis=0)    # previous index
                    return out, out2           # scores, previous sequence from which corresponding index
                else:
                    return out
            else:
                return self.log_sum_exp(
                    previous + obs + pp_transition,
                    axis=0
                )

        initial = observations[0]
        self.alpha, _ = theano.scan(
            fn=recurrence,
            sequences=[observations[1:], transitions],
            outputs_info=(initial, None) if return_best_sequence else initial
        )

        if return_alpha:
            return self.alpha
        elif return_best_sequence:
            sequence, _ = theano.scan(                
                fn=lambda beta_i, previous: beta_i[previous],     # back tracking
                sequences=T.cast(self.alpha[1][::-1], 'int32'),    # previous sequence
                outputs_info=T.cast(T.argmax(self.alpha[0][-1]), 'int32')   # last index of sequence having maximum score
            )
            sequence = T.concatenate([
                sequence[::-1],
                [T.argmax(self.alpha[0][-1])]
            ])
            return sequence
        else:
            if viterbi:
                return self.alpha[-1].max(axis=0)
            else:
                return self.log_sum_exp(self.alpha[-1], axis=0)

    def alpha_recursion_u(
        self,
        observations,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False
    ):
        assert not return_best_sequence or (viterbi and not return_alpha)
                
        def recurrence(obs, previous):
            previous = previous.dimshuffle(0, 'x')
            obs = obs.dimshuffle('x', 0)
            
            if viterbi:
                scores = previous + obs
                out = scores.max(axis=0)
                if return_best_sequence:
                    out2 = scores.argmax(axis=0)    # previous index
                    return out, out2           # scores, previous sequence from which corresponding index
                else:
                    return out
            else:
                return self.log_sum_exp(
                    previous + obs,
                    axis=0
                )

        initial = observations[0]
        self.alpha, _ = theano.scan(
            fn=recurrence,
            sequences=observations[1:],
            outputs_info=(initial, None) if return_best_sequence else initial
        )

        if return_alpha:
            return self.alpha
        elif return_best_sequence:
            sequence, _ = theano.scan(                
                fn=lambda beta_i, previous: beta_i[previous],     # back tracking
                sequences=T.cast(self.alpha[1][::-1], 'int32'),    # previous sequence
                outputs_info=T.cast(T.argmax(self.alpha[0][-1]), 'int32')   # last index of sequence having maximum score
            )
            sequence = T.concatenate([
                sequence[::-1],
                [T.argmax(self.alpha[0][-1])]
            ])
            return sequence
        else:
            if viterbi:
                return self.alpha[-1].max(axis=0)
            else:
                return self.log_sum_exp(self.alpha[-1], axis=0)

    def alpha_recursion_pp(
        self,
        transitions,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False
    ):
        assert not return_best_sequence or (viterbi and not return_alpha)
                
        def recurrence(transition, previous):
            previous = previous.dimshuffle(0, 'x')
            pairwise_transition = transition.reshape((self.n_tags, self.n_tags))
            
            if viterbi:
                scores = previous + pairwise_transition
                out = scores.max(axis=0)
                if return_best_sequence:
                    out2 = scores.argmax(axis=0)    # previous index
                    return out, out2           # scores, previous sequence from which corresponding index
                else:
                    return out
            else:
                return self.log_sum_exp(
                    previous + pairwise_transition,
                    axis=0
                )
        
        initial = np.zeros((self.n_tags,), dtype=theano.config.floatX)
        self.alpha, _ = theano.scan(
            fn=recurrence,
            sequences=transitions,
            outputs_info=(initial, None) if return_best_sequence else initial
        )

        if return_alpha:
            return self.alpha
        elif return_best_sequence:
            sequence, _ = theano.scan(                
                fn=lambda beta_i, previous: beta_i[previous],     # back tracking
                sequences=T.cast(self.alpha[1][::-1], 'int32'),    # previous sequence
                outputs_info=T.cast(T.argmax(self.alpha[0][-1]), 'int32')   # last index of sequence having maximum score
            )
            sequence = T.concatenate([
                sequence[::-1],
                [T.argmax(self.alpha[0][-1])]
            ])
            return sequence
        else:
            if viterbi:
                return self.alpha[-1].max(axis=0)
            else:
                return self.log_sum_exp(self.alpha[-1], axis=0)

    def fprop(
        self,
        input_sequence,
        transitions,
        ground_truth,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False,
        mode='train'
    ):
        """Propogate input through the CRF."""
        seq_length = input_sequence.shape[0]
        if mode != 'train':
            return self.alpha_recursion(
                input_sequence,
                transitions,
                viterbi,
                return_alpha,
                return_best_sequence
            )
        real_path_score_u = input_sequence[T.arange(seq_length), ground_truth].sum()
#         real_path_score_u += self.transition_matrix[
#             ground_truth[T.arange(seq_length - 1)],
#             ground_truth[T.arange(seq_length - 1) + 1]
#             ].sum()
        all_paths_scores_u = self.alpha_recursion_u(
            input_sequence,
            viterbi,
            return_alpha,
            return_best_sequence
        )
        
        def transition_scoring(transition, from_id, to_id):
            pairwise_transition = transition.reshape((self.n_tags, self.n_tags)) 
            score = pairwise_transition[from_id, to_id] 
            return score

        transition_scores, _ = theano.scan(fn=transition_scoring, 
                                           sequences=[transitions,
                                                      ground_truth[T.arange(seq_length - 1)], 
                                                      ground_truth[T.arange(seq_length - 1) + 1]]
                                           )
        real_path_score_pp = transition_scores.sum()
        all_paths_scores_pp = self.alpha_recursion_pp(
            transitions,
            viterbi,
            return_alpha,
            return_best_sequence
        )
        
        all_paths_scores = self.alpha_recursion(
            input_sequence,
            transitions,
            viterbi,
            return_alpha,
            return_best_sequence
        )
        real_path_score = real_path_score_u + real_path_score_pp
        
        cost_u = - (real_path_score_u - all_paths_scores_u)
        cost_pp = - (real_path_score_pp - all_paths_scores_pp)
        cost = - (real_path_score - all_paths_scores)
        
        return cost, cost_u, cost_pp

class CRF_U_PP_Dropout(object):
    """Conditional Random Field Unary Potential and Pairwise Potential."""

    def __init__(self, n_tags):
        self.n_tags = n_tags
        _transition_matrix = np.random.rand(self.n_tags, self.n_tags)
        self.transition_matrix = theano.shared(_transition_matrix.astype(theano.config.floatX), borrow=True, name='A')
        self.params = [self.transition_matrix]
        print 'Initialize CRF_U_PP_Dropout'
        
        self.srng = RandomStreams(seed=13572)
        
    def log_sum_exp(self, x, axis=None):
        """Sum probabilities in the log-space."""
        xmax = x.max(axis=axis, keepdims=True)
        xmax_ = x.max(axis=axis)
        return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))

    def alpha_recursion(
        self,
        observations,
        transitions,
        dropout_sequence,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False
    ):
        assert not return_best_sequence or (viterbi and not return_alpha)
                
        def recurrence(obs, transition, previous):
            previous = previous.dimshuffle(0, 'x')
            obs = obs.dimshuffle('x', 0)
            pp_transition = transition.reshape((self.n_tags, self.n_tags))
            
            scores = previous + obs + pp_transition + self.transition_matrix
            out = scores.max(axis=0)
            if return_best_sequence:
                out2 = scores.argmax(axis=0)    # previous index
                return out, out2                # scores, previous sequence from which corresponding index
            else:
                return out

        def recurrence_dropout(obs, transition, dropout, previous):
            previous = previous.dimshuffle(0, 'x')
            obs = obs.dimshuffle('x', 0)
            pp_transition = transition.reshape((self.n_tags, self.n_tags))
            scoreA = self.log_sum_exp(previous + pp_transition, axis=0)
            scoreB = self.log_sum_exp(previous + obs + self.transition_matrix, axis=0)
            score = T.switch(dropout, scoreA, scoreB)
            return score

        initial = observations[0]
        if viterbi == True:
            self.alpha, _ = theano.scan(
                fn=recurrence,
                sequences=[observations[1:], transitions],
                outputs_info=(initial, None) if return_best_sequence else (initial, None)
            )
        else:
            self.alpha, _ = theano.scan(
                fn=recurrence_dropout,
                sequences=[observations[1:], transitions, dropout_sequence],
                outputs_info=initial
            )

        if return_alpha:
            return self.alpha
        elif return_best_sequence:
            sequence, _ = theano.scan(                
                fn=lambda beta_i, previous: beta_i[previous],     # back tracking
                sequences=T.cast(self.alpha[1][::-1], 'int32'),    # previous sequence
                outputs_info=T.cast(T.argmax(self.alpha[0][-1]), 'int32')   # last index of sequence having maximum score
            )
            sequence = T.concatenate([
                sequence[::-1],
                [T.argmax(self.alpha[0][-1])]
            ])
            return sequence
        else:
            if viterbi:
                return self.alpha[-1].max(axis=0)
            else:
                return self.log_sum_exp(self.alpha[-1], axis=0)

    def alpha_recursion_u(
        self,
        observations,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False
    ):
        assert not return_best_sequence or (viterbi and not return_alpha)
                
        def recurrence(obs, previous):
            previous = previous.dimshuffle(0, 'x')
            obs = obs.dimshuffle('x', 0)
            return self.log_sum_exp(previous + obs + self.transition_matrix, axis=0)

        initial = observations[0]
        self.alpha, _ = theano.scan(
            fn=recurrence,
            sequences=observations[1:],
            outputs_info=initial
        )

        return self.log_sum_exp(self.alpha[-1], axis=0)

    def alpha_recursion_pp(
        self,
        transitions,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False
    ):
        assert not return_best_sequence or (viterbi and not return_alpha)

        def recurrence(transition, previous):
            previous = previous.dimshuffle(0, 'x')
            pairwise_transition = transition.reshape((self.n_tags, self.n_tags))
            
            return self.log_sum_exp(previous + pairwise_transition, axis=0)
        
        initial = np.zeros((self.n_tags,), dtype=theano.config.floatX)
        self.alpha, _ = theano.scan(
            fn=recurrence,
            sequences=transitions,
            outputs_info=initial
        )

        return self.log_sum_exp(self.alpha[-1], axis=0)

    def fprop(
        self,
        input_sequence,
        transitions,
        ground_truth,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False,
        mode='train'
    ):
        """Propogate input through the CRF."""
        seq_length = input_sequence.shape[0]
        if mode != 'train':
            return self.alpha_recursion(
                input_sequence,
                transitions,
                None,
                viterbi,
                return_alpha,
                return_best_sequence
            )

        def dropout_sequence_generation(transition):
            dropout = self.srng.binomial(n=1, size=(1,), p=0.65)
            return dropout
 
        dropout_sequence, _ = theano.scan(fn=dropout_sequence_generation, 
                                          sequences=[transitions]
                                          )
        all_paths_scores = self.alpha_recursion(
            input_sequence,
            transitions,
            dropout_sequence,
            viterbi,
            return_alpha,
            return_best_sequence
        )

        all_paths_scores_u = self.alpha_recursion_u(
            input_sequence,
            viterbi,
            return_alpha,
            return_best_sequence
        )

        all_paths_scores_pp = self.alpha_recursion_pp(
            transitions,
            viterbi,
            return_alpha,
            return_best_sequence
        )
        
        unary_sequence = input_sequence[T.arange(seq_length), ground_truth]
        
        matrix_sequence = self.transition_matrix[ground_truth[T.arange(seq_length - 1)], 
                                                 ground_truth[T.arange(seq_length - 1) + 1]
                                                ]
        
        def transition_scoring(transition, from_id, to_id):
            pairwise_transition = transition.reshape((self.n_tags, self.n_tags)) 
            score = pairwise_transition[from_id, to_id] 
            return score
 
        pairwise_sequence, _ = theano.scan(fn=transition_scoring, 
                                  sequences=[transitions,
                                  ground_truth[T.arange(seq_length - 1)], 
                                  ground_truth[T.arange(seq_length - 1) + 1]]
                                  )
                
        def scoring_real_path(unary, matrix_entry, pairwise, dropout):
            return T.switch(dropout, pairwise, unary + matrix_entry)
        
        step_scores, _ = theano.scan(fn=scoring_real_path, sequences=[unary_sequence[1:], 
                                                                      matrix_sequence, 
                                                                      pairwise_sequence, 
                                                                      dropout_sequence])

        real_path_score = unary_sequence[0] + step_scores.sum()
        cost = - (real_path_score - all_paths_scores)
        cost_u = - (unary_sequence.sum() + matrix_sequence.sum() - all_paths_scores_u)
        cost_pp = - (pairwise_sequence.sum() - all_paths_scores_pp) 
        
        return cost, cost_u, cost_pp

class CRF_Only_PP_Boundary(object):
    """Conditional Random Field Pairwise Potential."""

    def __init__(self, n_tags):
        self.n_tags = n_tags

        print 'Initialize CRFPP_with Only PP_Boundary'
    
    def log_sum_exp(self, x, axis=None):
        """Sum probabilities in the log-space."""
        xmax = x.max(axis=axis, keepdims=True)
        xmax_ = x.max(axis=axis)
        return xmax_ + T.log(T.exp(x - xmax).sum(axis=axis))

    def alpha_recursion(
        self,
        transitions,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False
    ):
        assert not return_best_sequence or (viterbi and not return_alpha)
                
        def recurrence(transition_input, previous):
            previous = previous.dimshuffle(0, 'x')
            M = T.alloc(transition_input[3], self.n_tags, self.n_tags)
            if self.n_tags == 5:
                M = T.set_subtensor(M[[0,0,0,0],[1,2,3,4]],transition_input[1])
                M = T.set_subtensor(M[[1,2,3,4],[0,0,0,0]],transition_input[2])
                pairwise_transition = T.set_subtensor(M[0,0],transition_input[0])
            elif self.n_tags == 9:
                M = T.set_subtensor(M[[0,0,0,0,0,0,0,0],[1,2,3,4,5,6,7,8]],transition_input[1])
                M = T.set_subtensor(M[[1,2,3,4,5,6,7,8],[0,0,0,0,0,0,0,0]],transition_input[2])
                pairwise_transition = T.set_subtensor(M[0,0],transition_input[0])

#             elif self.n_tags == 9:
#                 M = T.set_subtensor(M[[0,0,0,0],[1,4,5,8]],transition_input[1])
#                 pairwise_transition = T.set_subtensor(M[[3,4,7,8],[0,0,0,0]],transition_input[2])

            if viterbi:
                scores = previous + pairwise_transition
                out = scores.max(axis=0)
                if return_best_sequence:
                    out2 = scores.argmax(axis=0)    # previous index
                    return out, out2           # scores, previous sequence from which corresponding index
                else:
                    return out
            else:
                return self.log_sum_exp(
                    previous + pairwise_transition,
                    axis=0
                )

        initial = np.zeros((self.n_tags,), dtype=theano.config.floatX)
        self.alpha, _ = theano.scan(
            fn=recurrence,
            sequences=transitions,
            outputs_info=(initial, None) if return_best_sequence else initial
        )

        if return_alpha:
            return self.alpha
        elif return_best_sequence:
            sequence, _ = theano.scan(                
                fn=lambda beta_i, previous: beta_i[previous],     # back tracking
                sequences=T.cast(self.alpha[1][::-1], 'int32'),    # previous sequence
                outputs_info=T.cast(T.argmax(self.alpha[0][-1]), 'int32')   # last index of sequence having maximum score
            )
            sequence = T.concatenate([
                sequence[::-1],
                [T.argmax(self.alpha[0][-1])]
            ])
            return sequence
        else:
            if viterbi:
                return self.alpha[-1].max(axis=0)
            else:
                return self.log_sum_exp(self.alpha[-1], axis=0)

    def sequence_predict(
        self,
        observations,
        transitions,
        transition_matrix
    ):
        def recurrence(obs, transition_input, previous, transition_matrix):
            previous = previous.dimshuffle(0, 'x')
            obs = obs.dimshuffle('x', 0)
#             M = T.alloc(transition_input[0], self.n_tags, self.n_tags)
#             M = T.set_subtensor(M[[0,0],[1,4]],transition_input[1])
#             pairwise_transition = T.set_subtensor(M[[3,4],[0,0]],transition_input[2])
#             scores = previous + obs + pairwise_transition + transition_matrix
            if self.n_tags == 5:
                M = T.alloc(transition_input[3], self.n_tags, self.n_tags)
                M = T.set_subtensor(M[[0,0,0,0],[1,2,3,4]],transition_input[1])
                M = T.set_subtensor(M[[1,2,3,4],[0,0,0,0]],transition_input[2])
                pairwise_transition = T.set_subtensor(M[0,0],transition_input[0])
            elif self.n_tags == 9:
                M = T.alloc(transition_input[3], self.n_tags, self.n_tags)
                M = T.set_subtensor(M[[0,0,0,0,0,0,0,0],[1,2,3,4,5,6,7,8]],transition_input[1])
                M = T.set_subtensor(M[[1,2,3,4,5,6,7,8],[0,0,0,0,0,0,0,0]],transition_input[2])
                pairwise_transition = T.set_subtensor(M[0,0],transition_input[0])

#             elif self.n_tags == 9:
#                 transition_matrix = T.set_subtensor(transition_matrix[[0,0,0,0],[1,4,5,8]],transition_input[1])
#                 pairwise_transition = T.set_subtensor(transition_matrix[[3,4,7,8],[0,0,0,0]],transition_input[2])

            scores = previous + obs + pairwise_transition + transition_matrix
            out = scores.max(axis=0)
            out2 = scores.argmax(axis=0)    # previous index

            return out, out2           # scores, previous sequence from which corresponding index

        initial = observations[0]
        self.alpha, _ = theano.scan(
            fn=recurrence,
            sequences=[observations[1:], transitions],
            outputs_info=(initial, None),
            non_sequences=transition_matrix
        )

        sequence, _ = theano.scan(                
            fn=lambda beta_i, previous: beta_i[previous],     # back tracking
            sequences=T.cast(self.alpha[1][::-1], 'int32'),    # previous sequence
            outputs_info=T.cast(T.argmax(self.alpha[0][-1]), 'int32')   # last index of sequence having maximum score
        )
        
        sequence = T.concatenate([
            sequence[::-1],
            [T.argmax(self.alpha[0][-1])]
        ])

        return sequence

    def fprop(
        self,
        transition_input,
        ground_truth,
        viterbi=False,
        return_alpha=False,
        return_best_sequence=False,
        mode='train'
    ):
        """Propogate input through the CRF."""
        seq_length = transition_input.shape[0]
        if mode != 'train':
            return self.alpha_recursion(
                transition_input,
                viterbi,
                return_alpha,
                return_best_sequence
            )
        def transition_scoring(transition_input, from_id, to_id):
            if self.n_tags == 5:
                M = T.alloc(transition_input[3], self.n_tags, self.n_tags)
                M = T.set_subtensor(M[[0,0,0,0],[1,2,3,4]],transition_input[1])
                M = T.set_subtensor(M[[1,2,3,4],[0,0,0,0]],transition_input[2])
                M = T.set_subtensor(M[0,0],transition_input[0])
            elif self.n_tags == 9:
                M = T.alloc(transition_input[3], self.n_tags, self.n_tags)
                M = T.set_subtensor(M[[0,0,0,0,0,0,0,0],[1,2,3,4,5,6,7,8]],transition_input[1])
                M = T.set_subtensor(M[[1,2,3,4,5,6,7,8],[0,0,0,0,0,0,0,0]],transition_input[2])
                M = T.set_subtensor(M[0,0],transition_input[0])

#                 M = T.alloc(transition_input[0], self.n_tags, self.n_tags)
#                 M = T.set_subtensor(M[[0,0],[1,4]],transition_input[1])
#                 M = T.set_subtensor(M[[3,4],[0,0]],transition_input[2])
#             elif self.n_tags == 9:
#                 M = T.set_subtensor(M[[0,0,0,0],[1,4,5,8]],transition_input[1])
#                 M = T.set_subtensor(M[[3,4,7,8],[0,0,0,0]],transition_input[2])
            return M[from_id, to_id]
 
        transition_scores, _ = theano.scan(fn=transition_scoring, 
                                           sequences=[transition_input,
                                                      ground_truth[T.arange(seq_length)], 
                                                      ground_truth[T.arange(seq_length) + 1]]
                                           )
        real_path_score = transition_scores.sum()
        all_paths_scores = self.alpha_recursion(
            transition_input,
            viterbi,
            return_alpha,
            return_best_sequence
        )
        
        cost = - (real_path_score - all_paths_scores)
        
        return cost

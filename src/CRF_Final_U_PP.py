import theano
import theano.tensor as T
import numpy as np
import CRF_PP_With_Type as CRFPP
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

class CRF_Final_U_PP:
    def __init__(self, orthogonal_weight_init, w2v_matrix, word_embedding_dim, n_characters, char_embedding_dim, n_pos, pos_embedding_dim, max_token_length, 
                 n_filter, filter_width, 
                 cnn_output_dim, lstm_hidden_dim2, lstm_hidden_dim, output_dim, mlp_dim, 
                 train_words, train_lengths, train_targets, train_char_samples, train_pos,
                 test_words, test_lengths, test_targets, test_char_samples, testing_char_lengths, test_pos,
                 gradient_threshold,  
                 highway_network_enable):
        self.orthogonal_weight_init = orthogonal_weight_init
        self.word_embedding_dim = word_embedding_dim
        self.n_characters = n_characters
        self.char_embedding_dim = char_embedding_dim
        self.n_pos = n_pos
        self.pos_embedding_dim = pos_embedding_dim
        self.max_token_length = max_token_length
        self.n_filter = n_filter
        self.filter_width = filter_width
        self.cnn_output_dim = cnn_output_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_hidden_dim2 = lstm_hidden_dim2
        self.output_dim = output_dim
        self.mlp_dim = mlp_dim
        self.n_labels = output_dim
        self.gradient_threshold = gradient_threshold
        self.highway_network_enable = highway_network_enable
        self.PP_output_dim = output_dim
        self.filter_shapes = []
        self.pool_shapes = []
        self.n_rnn_matrix = 4
        self._eps = 1e-5
        self.unary_view = False
        self.multiview = True
        self.double_unary = False
        self.three_unary = False
        self.crf_dropout = True
        self.cnn_pos_embedding = False
        self.cnn_on = True
        self.pos_on = True
        self.initial = True
        self.pp_addition = True
        self.separate_deep = True
        self.LN = True
        self.without_transition_matrix = False
        self.entity_segmentation = False
        self.output_mlp = False
        
        print 'Welcome to CRF_Separate_PP_with_CNN'
        
        print 'CNN on = ', self.cnn_on
        print 'POS on = ', self.pos_on
        print 'Unary View = ', self.unary_view
        print 'Multi View = ', self.multiview
        print 'Double Unary = ', self.double_unary
        print 'Three Unary = ', self.three_unary
        print 'CRF Dropout = ', self.crf_dropout
        print 'Initial State = ', self.initial
        print 'PP addition = ', self.pp_addition
        print 'Deep Separated = ', self.separate_deep
        print 'Without Transition Matrix = ', self.without_transition_matrix
        print 'Highway network enable = ', self.highway_network_enable
        print 'Output MLP enable = ', self.output_mlp
        print 'Layer Normalization = ', self.LN
        
        print 'CNN filters = ', n_filter
        
        print 'CNN output dimension = ', cnn_output_dim
        
        print 'n_characters = ', n_characters
        print 'n_pos = ', n_pos
        if self.cnn_pos_embedding == False:
            char_embedding_dim = n_characters
            pos_embedding_dim = n_pos
        self.char_embedding_dim = char_embedding_dim 
        self.pos_embedding_dim = pos_embedding_dim 

        F = []
        F_pp = []
        F_u = []
        for i, width in enumerate(self.filter_width):
            filter_shape = (n_filter[i], 1, char_embedding_dim, width)
            self.filter_shapes.append(filter_shape)
            pool_shape = (1,max_token_length-width+1)
            self.pool_shapes.append(pool_shape)
            F.append(np.random.uniform(-0.05, 0.05, (n_filter[i], 1, char_embedding_dim, width)))
            F_pp.append(np.random.uniform(-0.05, 0.05, (n_filter[i], 1, char_embedding_dim, width)))
            F_u.append(np.random.uniform(-0.05, 0.05, (n_filter[i], 1, char_embedding_dim, width)))
            print filter_shape, pool_shape
        
        if self.cnn_pos_embedding == True:
            P = np.random.uniform(-0.99, 0.99, (n_characters, char_embedding_dim)).astype(theano.config.floatX)
            POS = np.random.uniform(-0.99, 0.99, (n_pos, pos_embedding_dim)).astype(theano.config.floatX)
        else:
            P = np.eye(n_characters, dtype=theano.config.floatX)
            POS = np.eye(n_pos, dtype=theano.config.floatX)        

        if self.highway_network_enable == True:
            mlp_highway_gate_h = np.random.uniform(-np.sqrt(6./np.sum([cnn_output_dim, cnn_output_dim])), np.sqrt(6./np.sum([cnn_output_dim, cnn_output_dim])), (cnn_output_dim, cnn_output_dim)) 
            mlp_highway_gate_b = np.zeros((cnn_output_dim,))
            mlp_highway_h = np.random.uniform(-np.sqrt(6./np.sum([cnn_output_dim, cnn_output_dim])), np.sqrt(6./np.sum([cnn_output_dim, cnn_output_dim])), (cnn_output_dim, cnn_output_dim)) 
            mlp_highway_b = np.zeros((cnn_output_dim,))
            mlp_highway_gate_h_pp = np.random.uniform(-np.sqrt(6./np.sum([cnn_output_dim, cnn_output_dim])), np.sqrt(6./np.sum([cnn_output_dim, cnn_output_dim])), (cnn_output_dim, cnn_output_dim)) 
            mlp_highway_gate_b_pp = np.zeros((cnn_output_dim,))
            mlp_highway_h_pp = np.random.uniform(-np.sqrt(6./np.sum([cnn_output_dim, cnn_output_dim])), np.sqrt(6./np.sum([cnn_output_dim, cnn_output_dim])), (cnn_output_dim, cnn_output_dim)) 
            mlp_highway_b_pp = np.zeros((cnn_output_dim,))
            if self.three_unary == True:
                mlp_highway_gate_h_u = np.random.uniform(-np.sqrt(6./np.sum([cnn_output_dim, cnn_output_dim])), np.sqrt(6./np.sum([cnn_output_dim, cnn_output_dim])), (cnn_output_dim, cnn_output_dim)) 
                mlp_highway_gate_b_u = np.zeros((cnn_output_dim,))
                mlp_highway_h_u = np.random.uniform(-np.sqrt(6./np.sum([cnn_output_dim, cnn_output_dim])), np.sqrt(6./np.sum([cnn_output_dim, cnn_output_dim])), (cnn_output_dim, cnn_output_dim)) 
                mlp_highway_b_u = np.zeros((cnn_output_dim,))
        
        unary_input_dim = word_embedding_dim
        if self.cnn_on == True:
            unary_input_dim += cnn_output_dim
        if self.pos_on == True:
            unary_input_dim += pos_embedding_dim 
        
        init_hidden_mlp_input_dim = word_embedding_dim + pos_embedding_dim
        init_hidden_mlp_h = np.random.uniform(-np.sqrt(6./np.sum([init_hidden_mlp_input_dim, lstm_hidden_dim])), 
                                              np.sqrt(6./np.sum([init_hidden_mlp_input_dim, lstm_hidden_dim])), 
                                              (init_hidden_mlp_input_dim, lstm_hidden_dim))
        init_hidden_mlp_b = np.zeros((lstm_hidden_dim,), dtype=theano.config.floatX)
        init_cell_mlp_h = np.random.uniform(-np.sqrt(6./np.sum([init_hidden_mlp_input_dim, lstm_hidden_dim])), 
                                            np.sqrt(6./np.sum([init_hidden_mlp_input_dim, lstm_hidden_dim])), 
                                            (init_hidden_mlp_input_dim, lstm_hidden_dim))
        init_cell_mlp_b = np.zeros((lstm_hidden_dim,), dtype=theano.config.floatX) 

        init_hidden_mlp_h_pp = np.random.uniform(-np.sqrt(6./np.sum([init_hidden_mlp_input_dim, lstm_hidden_dim])), 
                                                 np.sqrt(6./np.sum([init_hidden_mlp_input_dim, lstm_hidden_dim])), 
                                                 (init_hidden_mlp_input_dim, lstm_hidden_dim))
        init_hidden_mlp_b_pp = np.zeros((lstm_hidden_dim,), dtype=theano.config.floatX)
        init_cell_mlp_h_pp = np.random.uniform(-np.sqrt(6./np.sum([init_hidden_mlp_input_dim, lstm_hidden_dim])), 
                                               np.sqrt(6./np.sum([init_hidden_mlp_input_dim, lstm_hidden_dim])), 
                                               (init_hidden_mlp_input_dim, lstm_hidden_dim))
        init_cell_mlp_b_pp = np.zeros((lstm_hidden_dim,), dtype=theano.config.floatX) 

        if self.three_unary == True:
            init_hidden_mlp_h_u = np.random.uniform(-np.sqrt(6./np.sum([init_hidden_mlp_input_dim, lstm_hidden_dim])), 
                                                     np.sqrt(6./np.sum([init_hidden_mlp_input_dim, lstm_hidden_dim])), 
                                                     (init_hidden_mlp_input_dim, lstm_hidden_dim))
            init_hidden_mlp_b_u = np.zeros((lstm_hidden_dim,), dtype=theano.config.floatX)
            init_cell_mlp_h_u = np.random.uniform(-np.sqrt(6./np.sum([init_hidden_mlp_input_dim, lstm_hidden_dim])), 
                                                   np.sqrt(6./np.sum([init_hidden_mlp_input_dim, lstm_hidden_dim])), 
                                                   (init_hidden_mlp_input_dim, lstm_hidden_dim))
            init_cell_mlp_b_u = np.zeros((lstm_hidden_dim,), dtype=theano.config.floatX) 

        U = np.random.uniform(-np.sqrt(6./np.sum([unary_input_dim, lstm_hidden_dim])), np.sqrt(6./np.sum([unary_input_dim, lstm_hidden_dim])), (2*self.n_rnn_matrix, unary_input_dim, lstm_hidden_dim))
        W = np.random.uniform(-np.sqrt(6./np.sum([lstm_hidden_dim, lstm_hidden_dim])), np.sqrt(6./np.sum([lstm_hidden_dim, lstm_hidden_dim])), (2*self.n_rnn_matrix, lstm_hidden_dim, lstm_hidden_dim))
        U_pp = np.random.uniform(-np.sqrt(6./np.sum([unary_input_dim, lstm_hidden_dim])), np.sqrt(6./np.sum([unary_input_dim, lstm_hidden_dim])), (2*self.n_rnn_matrix, unary_input_dim, lstm_hidden_dim))
        W_pp = np.random.uniform(-np.sqrt(6./np.sum([lstm_hidden_dim, lstm_hidden_dim])), np.sqrt(6./np.sum([lstm_hidden_dim, lstm_hidden_dim])), (2*self.n_rnn_matrix, lstm_hidden_dim, lstm_hidden_dim))
        if self.three_unary == True:
            U_u = np.random.uniform(-np.sqrt(6./np.sum([unary_input_dim, lstm_hidden_dim])), np.sqrt(6./np.sum([unary_input_dim, lstm_hidden_dim])), (2*self.n_rnn_matrix, unary_input_dim, lstm_hidden_dim))
            W_u = np.random.uniform(-np.sqrt(6./np.sum([lstm_hidden_dim, lstm_hidden_dim])), np.sqrt(6./np.sum([lstm_hidden_dim, lstm_hidden_dim])), (2*self.n_rnn_matrix, lstm_hidden_dim, lstm_hidden_dim))

        b = np.zeros((2*self.n_rnn_matrix, lstm_hidden_dim), dtype=theano.config.floatX)
        b[0] = np.float32(1.)
        b[self.n_rnn_matrix] = np.float32(1.)
        
        b_pp = np.zeros((2*self.n_rnn_matrix, lstm_hidden_dim), dtype=theano.config.floatX)
        b_pp[0] = np.float32(1.)
        b_pp[self.n_rnn_matrix] = np.float32(1.)
        
        if self.three_unary == True:
            b_u = np.zeros((2*self.n_rnn_matrix, lstm_hidden_dim), dtype=theano.config.floatX)
            b_u[0] = np.float32(1.)
            b_u[self.n_rnn_matrix] = np.float32(1.)

        if self.LN == True:
            LN_g_LSTM_init = np.ones((2, lstm_hidden_dim), dtype=theano.config.floatX)
            LN_b_LSTM_init = np.zeros((2, lstm_hidden_dim), dtype=theano.config.floatX)
            LN_g_LSTM = np.ones((4, 4*lstm_hidden_dim), dtype=theano.config.floatX)
            LN_b_LSTM = np.zeros((4, 4*lstm_hidden_dim), dtype=theano.config.floatX)
            LN_g_LSTM_c = np.ones((2, lstm_hidden_dim), dtype=theano.config.floatX)
            LN_b_LSTM_c = np.zeros((2, lstm_hidden_dim), dtype=theano.config.floatX)
            LN_g_LSTM_init_pp = np.ones((2, lstm_hidden_dim), dtype=theano.config.floatX)
            LN_b_LSTM_init_pp = np.zeros((2, lstm_hidden_dim), dtype=theano.config.floatX)
            LN_g_LSTM_pp = np.ones((4, 4*lstm_hidden_dim), dtype=theano.config.floatX)
            LN_b_LSTM_pp = np.zeros((4, 4*lstm_hidden_dim), dtype=theano.config.floatX)
            LN_g_LSTM_c_pp = np.ones((2, lstm_hidden_dim), dtype=theano.config.floatX)
            LN_b_LSTM_c_pp = np.zeros((2, lstm_hidden_dim), dtype=theano.config.floatX)
            if self.three_unary == True:
                LN_g_LSTM_init_u = np.ones((2, lstm_hidden_dim), dtype=theano.config.floatX)
                LN_b_LSTM_init_u = np.zeros((2, lstm_hidden_dim), dtype=theano.config.floatX)
                LN_g_LSTM_u = np.ones((4, 4*lstm_hidden_dim), dtype=theano.config.floatX)
                LN_b_LSTM_u = np.zeros((4, 4*lstm_hidden_dim), dtype=theano.config.floatX)
                LN_g_LSTM_c_u = np.ones((2, lstm_hidden_dim), dtype=theano.config.floatX)
                LN_b_LSTM_c_u = np.zeros((2, lstm_hidden_dim), dtype=theano.config.floatX)

        if self.output_mlp == True:        
            mlp_V = np.random.uniform(-np.sqrt(6./np.sum([2*lstm_hidden_dim, 1024])), np.sqrt(6./np.sum([2*lstm_hidden_dim, 1024])), (2*lstm_hidden_dim, 1024)) 
            mlp_V_b = np.zeros((1024,))
            V_input_dim = 1024
        else:
            V_input_dim = 2*lstm_hidden_dim
        V = np.random.uniform(-np.sqrt(6./np.sum([V_input_dim, output_dim])), np.sqrt(6./np.sum([V_input_dim, output_dim])), (V_input_dim, output_dim))
        c = np.zeros((output_dim,), dtype=theano.config.floatX)

        if self.three_unary == True:
            V_u = np.random.uniform(-np.sqrt(6./np.sum([V_input_dim, output_dim])), np.sqrt(6./np.sum([V_input_dim, output_dim])), (V_input_dim, output_dim))
            c_u = np.zeros((output_dim,), dtype=theano.config.floatX)
        
        if self.n_labels == 9:
#             PP_output_dim = 17
            if self.entity_segmentation == True:
                PP_output_dim = 4
            elif self.double_unary == True:
                PP_output_dim = self.n_labels
            else:
                PP_output_dim = self.n_labels*self.n_labels
        elif self.n_labels == 5:
#             PP_output_dim = 9
#             PP_output_dim = self.n_labels*self.n_labels
            if self.entity_segmentation == True:
                PP_output_dim = 4
            elif self.double_unary == True:
                PP_output_dim = self.n_labels
            else:
                PP_output_dim = self.n_labels*self.n_labels
        elif self.n_labels == 21:
#             PP_output_dim = 9
            PP_output_dim = self.n_labels*self.n_labels
        
        if self.double_unary == True:
            pp_mlp_input_dim = 2*lstm_hidden_dim
        elif self.pp_addition == True:
            pp_mlp_input_dim = 6*lstm_hidden_dim
        else:
            pp_mlp_input_dim = 4*lstm_hidden_dim 
#         pp_mlp_input_dim = 2*unary_input_dim
        
        if self.unary_view == False:
            if self.output_mlp == True:
                mlp_V_pp = np.random.uniform(-np.sqrt(6./np.sum([pp_mlp_input_dim, 2048])), np.sqrt(6./np.sum([pp_mlp_input_dim, 2048])), (pp_mlp_input_dim, 2048)) 
                mlp_V_pp_b = np.zeros((2048,))
                V_P_intput_dim = 2048
            else:
                V_P_intput_dim = pp_mlp_input_dim
            V_P = np.random.uniform(-np.sqrt(6./np.sum([V_P_intput_dim, PP_output_dim])), np.sqrt(6./np.sum([V_P_intput_dim, PP_output_dim])), (V_P_intput_dim, PP_output_dim))
            c_P = np.zeros((PP_output_dim,), dtype=theano.config.floatX)
                    
        print 'In CRF_Separate_PP_basedOn_CNN'
             
        print 'Params initialized...'

        self.E = theano.shared(name='E', value=w2v_matrix.astype(theano.config.floatX), borrow=True)

        if self.cnn_on == True:        
            self.F = []
            self.F_pp = []
            self.F_u = []
            for i, _ in enumerate(self.filter_width):
                self.F.append(theano.shared(value=F[i].astype(theano.config.floatX), borrow=True))
                if self.unary_view == False and self.separate_deep == True:
                    self.F_pp.append(theano.shared(value=F_pp[i].astype(theano.config.floatX), borrow=True))
                if self.three_unary == True:
                    self.F_u.append(theano.shared(value=F_u[i].astype(theano.config.floatX), borrow=True))
            self.P = theano.shared(name='P', value=P.astype(theano.config.floatX), borrow=True)

        self.POS = theano.shared(name='POS', value=POS.astype(theano.config.floatX), borrow=True) 
        
        if self.cnn_on == True:
            if self.highway_network_enable == True:
                self.mlp_highway_gate_h = theano.shared(name='mlp_highway_gate_h', value=mlp_highway_gate_h.astype(theano.config.floatX), borrow=True)
                self.mlp_highway_gate_b = theano.shared(name='mlp_highway_gate_b', value=mlp_highway_gate_b.astype(theano.config.floatX), borrow=True)
                self.mlp_highway_h = theano.shared(name='mlp_highway_h', value=mlp_highway_h.astype(theano.config.floatX), borrow=True)
                self.mlp_highway_b = theano.shared(name='mlp_highway_b', value=mlp_highway_b.astype(theano.config.floatX), borrow=True)
                if self.unary_view == False and self.separate_deep == True:
                    self.mlp_highway_gate_h_pp = theano.shared(name='mlp_highway_gate_h_pp', value=mlp_highway_gate_h_pp.astype(theano.config.floatX), borrow=True)
                    self.mlp_highway_gate_b_pp = theano.shared(name='mlp_highway_gate_b_pp', value=mlp_highway_gate_b_pp.astype(theano.config.floatX), borrow=True)
                    self.mlp_highway_h_pp = theano.shared(name='mlp_highway_h_pp', value=mlp_highway_h_pp.astype(theano.config.floatX), borrow=True)
                    self.mlp_highway_b_pp = theano.shared(name='mlp_highway_b_pp', value=mlp_highway_b_pp.astype(theano.config.floatX), borrow=True)
                if self.three_unary == True:
                    self.mlp_highway_gate_h_u = theano.shared(name='mlp_highway_gate_h_u', value=mlp_highway_gate_h_u.astype(theano.config.floatX), borrow=True)
                    self.mlp_highway_gate_b_u = theano.shared(name='mlp_highway_gate_b_u', value=mlp_highway_gate_b_u.astype(theano.config.floatX), borrow=True)
                    self.mlp_highway_h_u = theano.shared(name='mlp_highway_h_u', value=mlp_highway_h_u.astype(theano.config.floatX), borrow=True)
                    self.mlp_highway_b_u = theano.shared(name='mlp_highway_b_u', value=mlp_highway_b_u.astype(theano.config.floatX), borrow=True)

        if self.LN == True:        
            self.LN_g_LSTM = theano.shared(name='LN_g_LSTM', value=LN_g_LSTM.astype(theano.config.floatX), borrow=True)
            self.LN_b_LSTM = theano.shared(name='LN_b_LSTM', value=LN_b_LSTM.astype(theano.config.floatX), borrow=True)
            self.LN_g_LSTM_c = theano.shared(name='LN_g_LSTM_c', value=LN_g_LSTM_c.astype(theano.config.floatX), borrow=True)
            self.LN_b_LSTM_c = theano.shared(name='LN_b_LSTM_c', value=LN_b_LSTM_c.astype(theano.config.floatX), borrow=True)
            if self.unary_view == False and self.separate_deep == True:
                self.LN_g_LSTM_pp = theano.shared(name='LN_g_LSTM_pp', value=LN_g_LSTM_pp.astype(theano.config.floatX), borrow=True)
                self.LN_b_LSTM_pp = theano.shared(name='LN_b_LSTM_pp', value=LN_b_LSTM_pp.astype(theano.config.floatX), borrow=True)
                self.LN_g_LSTM_c_pp = theano.shared(name='LN_g_LSTM_c_pp', value=LN_g_LSTM_c_pp.astype(theano.config.floatX), borrow=True)
                self.LN_b_LSTM_c_pp = theano.shared(name='LN_b_LSTM_c_pp', value=LN_b_LSTM_c_pp.astype(theano.config.floatX), borrow=True)
                if self.three_unary == True:
                    self.LN_g_LSTM_u = theano.shared(name='LN_g_LSTM_u', value=LN_g_LSTM_u.astype(theano.config.floatX), borrow=True)
                    self.LN_b_LSTM_u = theano.shared(name='LN_b_LSTM_u', value=LN_b_LSTM_u.astype(theano.config.floatX), borrow=True)
                    self.LN_g_LSTM_c_u = theano.shared(name='LN_g_LSTM_c_u', value=LN_g_LSTM_c_u.astype(theano.config.floatX), borrow=True)
                    self.LN_b_LSTM_c_u = theano.shared(name='LN_b_LSTM_c_u', value=LN_b_LSTM_c_u.astype(theano.config.floatX), borrow=True)
            
        if self.initial == True:
            self.init_hidden_mlp_h = theano.shared(name='init_hidden_mlp_h', value=init_hidden_mlp_h.astype(theano.config.floatX), borrow=True)
            self.init_hidden_mlp_b = theano.shared(name='init_hidden_mlp_b', value=init_hidden_mlp_b.astype(theano.config.floatX), borrow=True)
            self.init_cell_mlp_h = theano.shared(name='init_cell_mlp_h', value=init_cell_mlp_h.astype(theano.config.floatX), borrow=True)
            self.init_cell_mlp_b = theano.shared(name='init_cell_mlp_b', value=init_cell_mlp_b.astype(theano.config.floatX), borrow=True)
            if self.unary_view == False and self.separate_deep == True:
                self.init_hidden_mlp_h_pp = theano.shared(name='init_hidden_mlp_h_pp', value=init_hidden_mlp_h_pp.astype(theano.config.floatX), borrow=True)
                self.init_hidden_mlp_b_pp = theano.shared(name='init_hidden_mlp_b_pp', value=init_hidden_mlp_b_pp.astype(theano.config.floatX), borrow=True)
                self.init_cell_mlp_h_pp = theano.shared(name='init_cell_mlp_h_pp', value=init_cell_mlp_h_pp.astype(theano.config.floatX), borrow=True)
                self.init_cell_mlp_b_pp = theano.shared(name='init_cell_mlp_b_pp', value=init_cell_mlp_b_pp.astype(theano.config.floatX), borrow=True)
            if self.three_unary == True:
                self.init_hidden_mlp_h_u = theano.shared(name='init_hidden_mlp_h_u', value=init_hidden_mlp_h_u.astype(theano.config.floatX), borrow=True)
                self.init_hidden_mlp_b_u = theano.shared(name='init_hidden_mlp_b_u', value=init_hidden_mlp_b_u.astype(theano.config.floatX), borrow=True)
                self.init_cell_mlp_h_u = theano.shared(name='init_cell_mlp_h_u', value=init_cell_mlp_h_u.astype(theano.config.floatX), borrow=True)
                self.init_cell_mlp_b_u = theano.shared(name='init_cell_mlp_b_u', value=init_cell_mlp_b_u.astype(theano.config.floatX), borrow=True)
    
            if self.LN == True:
                self.LN_g_LSTM_init = theano.shared(name='LN_g_LSTM_init', value=LN_g_LSTM_init.astype(theano.config.floatX), borrow=True)
                self.LN_b_LSTM_init = theano.shared(name='LN_b_LSTM_init', value=LN_b_LSTM_init.astype(theano.config.floatX), borrow=True)
                if self.unary_view == False and self.separate_deep == True:
                    self.LN_g_LSTM_init_pp = theano.shared(name='LN_g_LSTM_init_pp', value=LN_g_LSTM_init_pp.astype(theano.config.floatX), borrow=True)
                    self.LN_b_LSTM_init_pp = theano.shared(name='LN_b_LSTM_init_pp', value=LN_b_LSTM_init_pp.astype(theano.config.floatX), borrow=True)
                    if self.three_unary == True:
                        self.LN_g_LSTM_init_u = theano.shared(name='LN_g_LSTM_init_u', value=LN_g_LSTM_init_u.astype(theano.config.floatX), borrow=True)
                        self.LN_b_LSTM_init_u = theano.shared(name='LN_b_LSTM_init_u', value=LN_b_LSTM_init_u.astype(theano.config.floatX), borrow=True)

        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX), borrow=True)
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX), borrow=True)
        self.b = theano.shared(name='b', value=b.astype(theano.config.floatX), borrow=True)

        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX), borrow=True)
        self.c = theano.shared(name='c', value=c.astype(theano.config.floatX), borrow=True)

        if self.unary_view == False:
            if self.separate_deep == True:
                self.U_pp = theano.shared(name='U_pp', value=U_pp.astype(theano.config.floatX), borrow=True)
                self.W_pp = theano.shared(name='W_pp', value=W_pp.astype(theano.config.floatX), borrow=True)
                self.b_pp = theano.shared(name='b_pp', value=b_pp.astype(theano.config.floatX), borrow=True)
            self.V_P = theano.shared(name='V_P', value=V_P.astype(theano.config.floatX), borrow=True)
            self.c_P = theano.shared(name='c_P', value=c_P.astype(theano.config.floatX), borrow=True)
        
        if self.three_unary == True:
            self.U_u = theano.shared(name='U_u', value=U_u.astype(theano.config.floatX), borrow=True)
            self.W_u = theano.shared(name='W_u', value=W_u.astype(theano.config.floatX), borrow=True)
            self.b_u = theano.shared(name='b_u', value=b_u.astype(theano.config.floatX), borrow=True)
            self.V_u = theano.shared(name='V_u', value=V_u.astype(theano.config.floatX), borrow=True)
            self.c_u = theano.shared(name='c_u', value=c_u.astype(theano.config.floatX), borrow=True)
            
        if self.output_mlp == True:
            self.mlp_V = theano.shared(name='mlp_V', value=mlp_V.astype(theano.config.floatX), borrow=True)
            self.mlp_V_b = theano.shared(name='mlp_V_b', value=mlp_V_b.astype(theano.config.floatX), borrow=True)
            if self.unary_view == False:
                self.mlp_V_pp = theano.shared(name='mlp_V_pp', value=mlp_V_pp.astype(theano.config.floatX), borrow=True)
                self.mlp_V_pp_b = theano.shared(name='mlp_V_pp_b', value=mlp_V_pp_b.astype(theano.config.floatX), borrow=True)
  
        self.test_words = theano.shared(value=test_words.astype('int32'), borrow=True)
        self.test_lengths = theano.shared(value=test_lengths.astype('int32'), borrow=True)
        self.test_targets = theano.shared(value=test_targets.astype('int32'), borrow=True)
        self.test_char_samples = theano.shared(value=test_char_samples.astype('int32'), borrow=True)
        self.test_pos = theano.shared(value=test_pos.astype('int32'), borrow=True)

        self.train_words = theano.shared(value=train_words.astype('int32'), borrow=True)
        self.train_lengths = theano.shared(value=train_lengths.astype('int32'), borrow=True)
        self.train_targets = theano.shared(value=train_targets.astype('int32'), borrow=True)
        self.train_char_samples = theano.shared(value=train_char_samples.astype('int32'), borrow=True)
        self.train_pos = theano.shared(value=train_pos.astype('int32'), borrow=True)
                
        self.srng = RandomStreams(seed=2357)
        
        print 'N labels = ', self.n_labels
        if self.n_labels == 9:
#             self.crf = CRFPP.CRF_Focus_PP_Fx(self.n_labels)
#             self.crf = CRFPP.CRF_LOP_TwoType(self.n_labels)
#             self.crf = CRFPP.CRF_Focus_Only_PP(self.n_labels)
#             self.crf_u = CRFPP.CRF_Only_U_Logistic_Regression(self.n_labels)
            if self.multiview == True:
                if self.entity_segmentation == True:
                    self.crf = CRFPP.CRF_Only_PP_Boundary(self.n_labels)
                else:
                    self.crf = CRFPP.CRF_Focus_Only_PP_Full(self.n_labels)
                
                if self.without_transition_matrix == True:
                    self.crf_u = CRFPP.CRF_Only_U_Logistic_Regression(self.n_labels)
                else:
                    self.crf_u = CRFPP.CRF_Only_U(self.n_labels)
            elif self.unary_view == True:
                self.crf = CRFPP.CRF_Fx(self.n_labels)
            elif self.double_unary == True and self.three_unary == False:
                self.crf = CRFPP.CRF_U(self.n_labels)
                self.crf_u = CRFPP.CRF_Only_U(self.n_labels)
            elif self.three_unary == True:
                self.crf = CRFPP.CRF_U(self.n_labels)
                self.crf_u = CRFPP.CRF_Only_U(self.n_labels)
                self.crf_uu = CRFPP.CRF_U(self.n_labels) 
            else:
                if self.crf_dropout == True:
                    self.crf = CRFPP.CRF_U_PP_Dropout(self.n_labels)
                else:
                    self.crf = CRFPP.CRF_U_PP_Fx(self.n_labels)
        elif self.n_labels == 5:
#             self.crf = CRFPP.CRF_LOP_OneType(self.n_labels)
#             self.crf = CRFPP.CRF_Focus_PP_OneType(self.n_labels)
            if self.multiview == True:
                self.crf_u = CRFPP.CRF_Only_U(self.n_labels)
    #             self.crf_u = CRFPP.CRF_Only_U_Logistic_Regression(self.n_labels)
#                 self.crf = CRFPP.CRF_Focus_Only_PP_Full(self.n_labels)
                if self.entity_segmentation == True:
                    self.crf = CRFPP.CRF_Only_PP_Boundary(self.n_labels)
                else:
                    self.crf = CRFPP.CRF_Focus_Only_PP_Full(self.n_labels)
            elif self.unary_view == True:
                self.crf = CRFPP.CRF_Fx(self.n_labels)
            elif self.double_unary == True and self.three_unary == False:
                self.crf = CRFPP.CRF_U(self.n_labels)                
                self.crf_u = CRFPP.CRF_Only_U(self.n_labels)
            elif self.three_unary == True:
                self.crf = CRFPP.CRF_U(self.n_labels)
                self.crf_u = CRFPP.CRF_Only_U(self.n_labels)
                self.crf_uu = CRFPP.CRF_U(self.n_labels) 
            else:
                if self.crf_dropout == True:
                    self.crf = CRFPP.CRF_U_PP_Dropout(self.n_labels)
                else:
                    self.crf = CRFPP.CRF_U_PP_Fx(self.n_labels)
        elif self.n_labels == 21:
#             self.crf = CRFPP.CRF_LOP_OneType(self.n_labels)
#             self.crf = CRFPP.CRF_Focus_PP_OneType(self.n_labels)
            if self.multiview == True:
                self.crf_u = CRFPP.CRF_Only_U(self.n_labels)
    #             self.crf_u = CRFPP.CRF_Only_U_Logistic_Regression(self.n_labels)
                self.crf = CRFPP.CRF_Focus_Only_PP_Full(self.n_labels)
            elif self.unary_view == True:
                self.crf = CRFPP.CRF_Fx(self.n_labels)
            elif self.three_unary == True:
                self.crf = CRFPP.CRF_U(self.n_labels)
                self.crf_u = CRFPP.CRF_Only_U(self.n_labels)
                self.crf_uu = CRFPP.CRF_U(self.n_labels) 
            else:
                if self.crf_dropout == True:
                    self.crf = CRFPP.CRF_U_PP_Dropout(self.n_labels)
                else:
                    self.crf = CRFPP.CRF_U_PP_Fx(self.n_labels)
        
        self.build()
        
        print 'Train Graph Generated ...'
        
        self.test_build()
        print 'Test Graph Generated ...'
    
    def getName(self):
        if self.unary_view == True:
            return '_unary_view_'
        if self.multiview == True:
            return '_multi_view_'
        return '_unified_view_'
        
    def clip_norms(self, gs, c):
        norm = T.sqrt(T.sum([T.sum(g**2) for g in gs]))
        return [T.switch(T.ge(norm, c), g*c/norm, g) for g in gs]
            
    def glorot_weight_initialization(self, n, row_ndim, col_ndim):
        if n > 0:
            W = np.random.uniform(-np.sqrt(6./np.sum([row_ndim, col_ndim])), np.sqrt(6./np.sum([row_ndim, col_ndim])), (n, row_ndim, col_ndim))
        else:
            W = np.random.uniform(-np.sqrt(6./np.sum([row_ndim, col_ndim])), np.sqrt(6./np.sum([row_ndim, col_ndim])), (row_ndim, col_ndim))
        return W
    
    def unary_character_cnn_step(self, x_t):
        char_embeds = self.P[x_t]
        char_embeds2d = T.shape_padleft(char_embeds.T, 2)
        
        pooled_out = []
        for i, _ in enumerate(self.filter_width):
            conv_out = T.tanh(T.nnet.conv2d(input=char_embeds2d, filters=self.F[i], filter_shape=self.filter_shapes[i], input_shape=(1,1,self.char_embedding_dim, self.max_token_length)))
            pooled_out.append(T.signal.pool.pool_2d(conv_out, ws=self.pool_shapes[i], ignore_border=False))
        
        o_pooled = T.concatenate(pooled_out, axis=1)
                        
        _e = T.cast(T.flatten(o_pooled), theano.config.floatX)
        if self.highway_network_enable == True:
            highway_gate = T.nnet.sigmoid(T.dot(_e, self.mlp_highway_gate_h) + self.mlp_highway_gate_b)
            wo = (1.-highway_gate)*_e + highway_gate*T.nnet.relu(T.dot(_e, self.mlp_highway_h) + self.mlp_highway_b)
        else:
            wo = _e
            
        return wo

    def pp_character_cnn_step(self, x_t):
        char_embeds = self.P[x_t]
        char_embeds2d = T.shape_padleft(char_embeds.T, 2)
        
        pooled_out = []
        for i, _ in enumerate(self.filter_width):
            conv_out = T.tanh(T.nnet.conv2d(input=char_embeds2d, filters=self.F_pp[i], filter_shape=self.filter_shapes[i], input_shape=(1,1,self.char_embedding_dim, self.max_token_length)))
            pooled_out.append(T.signal.pool.pool_2d(conv_out, ws=self.pool_shapes[i], ignore_border=False))
        
        o_pooled = T.concatenate(pooled_out, axis=1)
                        
        _e = T.cast(T.flatten(o_pooled), theano.config.floatX)
        if self.highway_network_enable == True:
            highway_gate = T.nnet.sigmoid(T.dot(_e, self.mlp_highway_gate_h_pp) + self.mlp_highway_gate_b_pp)
            wo = (1.-highway_gate)*_e + highway_gate*T.nnet.relu(T.dot(_e, self.mlp_highway_h_pp) + self.mlp_highway_b_pp)
        else:
            wo = _e
            
        return wo

    def third_unary_character_cnn_step(self, x_t):
        char_embeds = self.P[x_t]
        char_embeds2d = T.shape_padleft(char_embeds.T, 2)
        
        pooled_out = []
        for i, _ in enumerate(self.filter_width):
            conv_out = T.tanh(T.nnet.conv2d(input=char_embeds2d, filters=self.F_u[i], filter_shape=self.filter_shapes[i], input_shape=(1,1,self.char_embedding_dim, self.max_token_length)))
            pooled_out.append(T.signal.pool.pool_2d(conv_out, ws=self.pool_shapes[i], ignore_border=False))
        
        o_pooled = T.concatenate(pooled_out, axis=1)
                        
        _e = T.cast(T.flatten(o_pooled), theano.config.floatX)
        if self.highway_network_enable == True:
            highway_gate = T.nnet.sigmoid(T.dot(_e, self.mlp_highway_gate_h_u) + self.mlp_highway_gate_b_u)
            wo = (1.-highway_gate)*_e + highway_gate*T.nnet.relu(T.dot(_e, self.mlp_highway_h_u) + self.mlp_highway_b_u)
        else:
            wo = _e
            
        return wo

    def LSTM(self, x_t, s_t, c_t, U_f, U_i, U_o, U_c, W_f, W_i, W_o, W_c, b_f, b_i, b_o, b_c):
        x_t = T.cast(x_t, theano.config.floatX)
        s_t = T.cast(s_t, theano.config.floatX)
        f = T.nnet.sigmoid(T.dot(x_t, U_f) + T.dot(s_t, W_f) + b_f)
        i = T.nnet.sigmoid(T.dot(x_t, U_i) + T.dot(s_t, W_i) + b_i)
        o = T.nnet.sigmoid(T.dot(x_t, U_o) + T.dot(s_t, W_o) + b_o)
        _c = T.tanh(T.dot(x_t, U_c) + T.dot(s_t, W_c) + b_c)
        cell = i*_c + f*c_t
        s = o*T.tanh(cell)
        return s, cell

    def LSTM_LN(self, x_t, s_t, c_t, U, W, b_f, b_i, b_o, b_c, g_x, g_s, g_cell, b_x, b_s, b_cell, n_dim):
        _eps = np.float32(1e-5)
        x_t = T.cast(x_t, theano.config.floatX)
        s_t = T.cast(s_t, theano.config.floatX)
        x_preact = T.dot(x_t, U)
        x_in = (x_preact - x_preact.mean())/T.sqrt(x_preact.var()+_eps)
        x_ln = g_x*x_in + b_x
        s_preact = T.dot(s_t, W)
        s_in = (s_preact - s_preact.mean())/T.sqrt(s_preact.var()+_eps)
        s_ln = g_s*s_in + b_s
        f = T.nnet.sigmoid(x_ln[:n_dim] + s_ln[:n_dim] + b_f)
        i = T.nnet.sigmoid(x_ln[n_dim:2*n_dim] + s_ln[n_dim:2*n_dim] + b_i)
        o = T.nnet.sigmoid(x_ln[2*n_dim:3*n_dim] + s_ln[2*n_dim:3*n_dim] + b_o)
        g = T.tanh(x_ln[3*n_dim:] + s_ln[3*n_dim:] + b_c)
        cell = i*g + f*c_t
        cell_in = (cell - cell.mean())/T.sqrt(cell.var()+_eps)
        cell_ln = g_cell*cell_in + b_cell 
        s = o*T.tanh(cell_ln)

        return s, cell

    def build(self):
        indexes = T.ivector('indexes')
        word_sentences = T.imatrix('word_sentences')
        pos_sentences = T.imatrix('pos_sentences')
        lengths = T.ivector('lengths')
        labels = T.imatrix('labels')
        if self.cnn_on == True:
            char_sentences = T.itensor3('char_sentences')

        params_u = [self.U, self.W, self.V]
        params_u += [self.b, self.c]
#         if self.cnn_pos_embedding == True:
#             params_u += [self.P, self.POS]
        if self.output_mlp == True:
            params_u += [self.mlp_V, self.mlp_V_b]
            
        if self.unary_view == False:
            if self.separate_deep == True:
                params_pp = [self.U_pp, self.W_pp]
                params_pp += [self.b_pp]
            if self.output_mlp == True:
                params_pp += [self.mlp_V_pp, self.mlp_V_pp_b]
            params_pp += [self.V_P]
            params_pp += [self.c_P]
        if self.three_unary == True:
            params_uu = [self.U_u, self.W_u, self.b_u]
            params_uu += [self.V_u, self.c_u]
        
        if self.cnn_on == True:
            params_u += self.F
            if self.highway_network_enable == True:
                params_u += [self.mlp_highway_gate_h, self.mlp_highway_h]
                params_u += [self.mlp_highway_gate_b, self.mlp_highway_b]
            if self.unary_view == False and self.separate_deep == True:
                params_pp += self.F_pp
                if self.highway_network_enable == True:
                    params_pp += [self.mlp_highway_gate_h_pp, self.mlp_highway_h_pp]
                    params_pp += [self.mlp_highway_gate_b_pp, self.mlp_highway_b_pp]
            if self.three_unary == True and self.highway_network_enable == True: 
                params_uu += [self.mlp_highway_gate_h_u, self.mlp_highway_h_u]
                params_uu += [self.mlp_highway_gate_b_u, self.mlp_highway_b_u]
                    
        if self.LN == True:        
            params_u += [self.LN_g_LSTM, self.LN_b_LSTM]
            params_u += [self.LN_g_LSTM_c, self.LN_b_LSTM_c]
            if self.unary_view == False and self.separate_deep == True:
                params_pp += [self.LN_g_LSTM_pp, self.LN_b_LSTM_pp]
                params_pp += [self.LN_g_LSTM_c_pp, self.LN_b_LSTM_c_pp]
                if self.three_unary == True:
                    params_uu += [self.LN_g_LSTM_u, self.LN_b_LSTM_u]
                    params_uu += [self.LN_g_LSTM_c_u, self.LN_b_LSTM_c_u]
            
        if self.initial == True:
            if self.LN == True:
                params_u += [self.LN_g_LSTM_init, self.LN_b_LSTM_init]
                if self.unary_view == False and self.separate_deep == True:
                    params_pp += [self.LN_g_LSTM_init_pp, self.LN_b_LSTM_init_pp]
                if self.three_unary == True:
                    params_uu += [self.LN_g_LSTM_init_u, self.LN_b_LSTM_init_u]
                    
            params_u += [self.init_hidden_mlp_h, self.init_cell_mlp_h]
            params_u += [self.init_hidden_mlp_b, self.init_cell_mlp_b]
            if self.unary_view == False and self.separate_deep == True:
                params_pp += [self.init_hidden_mlp_h_pp, self.init_cell_mlp_h_pp]
                params_pp += [self.init_hidden_mlp_b_pp, self.init_cell_mlp_b_pp]
            if self.three_unary == True:
                params_uu += [self.init_hidden_mlp_h_u, self.init_cell_mlp_h_u]
                params_uu += [self.init_hidden_mlp_b_u, self.init_cell_mlp_b_u]

#         params_pp += self.crf.params
        if self.unary_view == False:
            if self.without_transition_matrix == False:
                params_u += self.crf_u.params if self.multiview == True else self.crf.params
                if self.double_unary == True and self.three_unary == False:
                    params_pp += self.crf_u.params
                if self.three_unary == True:
                    params_pp += self.crf_u.params
                    params_uu += self.crf_uu.params
        else:
            params_u += self.crf.params
        
        if self.unary_view == True:
            params = params_u
        elif self.three_unary == True:
            params = params_u + params_pp + params_uu
        else:
            params = params_u + params_pp
        
        def SentenceStep(word_sentence, length, pos_sentence, label_sequence, char_sentence=None):
            word_sentence = word_sentence[:length]
            pos_sentence = pos_sentence[:length]
            label_sequence = label_sequence[:length]
            if self.cnn_on == True:
                token_sentence = char_sentence[:length]

            if self.cnn_on == True and self.pos_on == True:
                token_cnn_sequence, _ = theano.scan(fn=self.unary_character_cnn_step, sequences=token_sentence)
                input = T.cast(T.concatenate([token_cnn_sequence, self.E[word_sentence], self.POS[pos_sentence]], axis=1), theano.config.floatX)
            elif self.cnn_on == True:
                token_cnn_sequence, _ = theano.scan(fn=self.unary_character_cnn_step, sequences=token_sentence)
                input = T.cast(T.concatenate([token_cnn_sequence, self.E[word_sentence]], axis=1), theano.config.floatX)
            elif self.pos_on == True:
                input = T.cast(T.concatenate([self.E[word_sentence], self.POS[pos_sentence]], axis=1), theano.config.floatX)
            else:
                input = T.cast(self.E[word_sentence], theano.config.floatX)

            forward_lstm_input = input
            backward_lstm_input = input[::-1]
            
            if self.initial == True:
                if self.pos_on == True:
                    x_lstm_init_mlp = T.concatenate([self.E[word_sentence].mean(axis=0).astype(theano.config.floatX), self.POS[pos_sentence].mean(axis=0).astype(theano.config.floatX)], axis=0)
                else:
                    x_lstm_init_mlp = T.concatenate([self.E[word_sentence].mean(axis=0).astype(theano.config.floatX), self.POS[pos_sentence].mean(axis=0).astype(theano.config.floatX)*0.], axis=0)
                if self.LN == True:
                    pre_x_lstm_mlp_init_state = T.dot(x_lstm_init_mlp, self.init_hidden_mlp_h)
                    STD_x_lstm_mlp_init_state = (pre_x_lstm_mlp_init_state - pre_x_lstm_mlp_init_state.mean())/T.sqrt(pre_x_lstm_mlp_init_state.var()+self._eps)
                    LN_x_lstm_mlp_init_state = self.LN_g_LSTM_init[0]*STD_x_lstm_mlp_init_state + self.LN_b_LSTM_init[0]
                     
                    pre_x_lstm_mlp_init_cell = T.dot(x_lstm_init_mlp, self.init_cell_mlp_h)
                    STD_x_lstm_mlp_init_cell = (pre_x_lstm_mlp_init_cell - pre_x_lstm_mlp_init_cell.mean())/T.sqrt(pre_x_lstm_mlp_init_cell.var()+self._eps)
                    LN_x_lstm_mlp_init_cell = self.LN_g_LSTM_init[1]*STD_x_lstm_mlp_init_cell + self.LN_b_LSTM_init[1]
                     
                    init_state = T.tanh(LN_x_lstm_mlp_init_state + self.init_hidden_mlp_b)
                    init_cell = T.tanh(LN_x_lstm_mlp_init_cell + self.init_cell_mlp_b)
                else:
                    init_state = T.dot(x_lstm_init_mlp, self.init_hidden_mlp_h) + self.init_hidden_mlp_b
                    init_cell = T.dot(x_lstm_init_mlp, self.init_cell_mlp_h) + self.init_cell_mlp_b
            else:
                init_state = np.zeros((self.lstm_hidden_dim,), dtype=theano.config.floatX)
                init_cell = np.zeros((self.lstm_hidden_dim,), dtype=theano.config.floatX)
            
            if self.LN == True:
                [f_s, f_c], _ = theano.scan(fn=self.LSTM_LN, 
                                            sequences=forward_lstm_input,
                                            outputs_info=[init_state, 
                                                          init_cell],
                                            non_sequences=[T.concatenate([self.U[0],self.U[1],self.U[2],self.U[3]], axis=1),
                                                           T.concatenate([self.W[0],self.W[1],self.W[2],self.W[3]], axis=1),
                                                           self.b[0],self.b[1],self.b[2],self.b[3],
                                                           self.LN_g_LSTM[0], self.LN_g_LSTM[1], self.LN_g_LSTM_c[0],
                                                           self.LN_b_LSTM[0], self.LN_b_LSTM[1], self.LN_b_LSTM_c[0],
                                                           np.int32(self.lstm_hidden_dim)
                                                           ])
     
                [b_s, b_c], _ = theano.scan(fn=self.LSTM_LN, 
                                            sequences=backward_lstm_input,
                                            outputs_info=[init_state, 
                                                          init_cell],                                                       
                                            non_sequences=[T.concatenate([self.U[4],self.U[5],self.U[6],self.U[7]], axis=1),
                                                           T.concatenate([self.W[4],self.W[5],self.W[6],self.W[7]], axis=1),
                                                           self.b[4],self.b[5],self.b[6],self.b[7],
                                                           self.LN_g_LSTM[2], self.LN_g_LSTM[3], self.LN_g_LSTM_c[1],
                                                           self.LN_b_LSTM[2], self.LN_b_LSTM[3], self.LN_b_LSTM_c[1],
                                                           np.int32(self.lstm_hidden_dim)
                                                           ])                
            else:
                [f_s, f_c], _ = theano.scan(fn=self.LSTM, 
                                            sequences=forward_lstm_input,
                                            outputs_info=[init_state, 
                                                          init_cell],
                                            non_sequences=[self.U[0],self.U[1],self.U[2],self.U[3],
                                                           self.W[0],self.W[1],self.W[2],self.W[3],
                                                           self.b[0],self.b[1],self.b[2],self.b[3]
                                                           ])
     
                [b_s, b_c], _ = theano.scan(fn=self.LSTM, 
                                            sequences=backward_lstm_input,
                                            outputs_info=[init_state, 
                                                          init_cell],                                                       
                                            non_sequences=[self.U[4],self.U[5],self.U[6],self.U[7],
                                                           self.W[4],self.W[5],self.W[6],self.W[7],
                                                           self.b[4],self.b[5],self.b[6],self.b[7]
                                                           ])                
                
            s1 = T.cast(T.concatenate([f_s, b_s[::-1]], axis=1), theano.config.floatX)
            if self.output_mlp == True:
                unary = T.dot(T.tanh(T.dot(s1, self.mlp_V) + self.mlp_V_b), self.V) + self.c 
            else:
                unary = T.dot(s1, self.V) + self.c 

            if self.unary_view == False:
                if self.separate_deep == True:
                    if self.cnn_on == True and self.pos_on == True:
                        pp_token_cnn_sequence, _ = theano.scan(fn=self.pp_character_cnn_step, sequences=token_sentence)
                        pp_input = T.cast(T.concatenate([pp_token_cnn_sequence, self.E[word_sentence], self.POS[pos_sentence]], axis=1), theano.config.floatX)                    
                    elif self.cnn_on == True:
                        pp_token_cnn_sequence, _ = theano.scan(fn=self.pp_character_cnn_step, sequences=token_sentence)
                        pp_input = T.cast(T.concatenate([pp_token_cnn_sequence, self.E[word_sentence]], axis=1), theano.config.floatX)                    
                    elif self.pos_on == True:
                        pp_input = T.cast(T.concatenate([self.E[word_sentence], self.POS[pos_sentence]], axis=1), theano.config.floatX)
                    else:
                        pp_input = T.cast(self.E[word_sentence], theano.config.floatX)
         
                    pp_forward_lstm_input = pp_input
                    pp_backward_lstm_input = pp_input[::-1]
          
                    if self.initial == True:
                        if self.pos_on == True:
                            x_lstm_init_mlp_pp = T.concatenate([self.E[word_sentence].mean(axis=0).astype(theano.config.floatX), self.POS[pos_sentence].mean(axis=0).astype(theano.config.floatX)], axis=0)
                        else:
                            x_lstm_init_mlp_pp = T.concatenate([self.E[word_sentence].mean(axis=0).astype(theano.config.floatX), self.POS[pos_sentence].mean(axis=0).astype(theano.config.floatX)*0.], axis=0)
                        if self.LN == True:
                            pre_x_lstm_mlp_init_state_pp = T.dot(x_lstm_init_mlp_pp, self.init_hidden_mlp_h_pp)
                            STD_x_lstm_mlp_init_state_pp = (pre_x_lstm_mlp_init_state_pp - pre_x_lstm_mlp_init_state_pp.mean())/T.sqrt(pre_x_lstm_mlp_init_state_pp.var()+self._eps)
                            LN_x_lstm_mlp_init_state_pp = self.LN_g_LSTM_init_pp[0]*STD_x_lstm_mlp_init_state_pp + self.LN_b_LSTM_init_pp[0]
                             
                            pre_x_lstm_mlp_init_cell_pp = T.dot(x_lstm_init_mlp_pp, self.init_cell_mlp_h_pp)
                            STD_x_lstm_mlp_init_cell_pp = (pre_x_lstm_mlp_init_cell_pp - pre_x_lstm_mlp_init_cell_pp.mean())/T.sqrt(pre_x_lstm_mlp_init_cell_pp.var()+self._eps)
                            LN_x_lstm_mlp_init_cell_pp = self.LN_g_LSTM_init_pp[1]*STD_x_lstm_mlp_init_cell_pp + self.LN_b_LSTM_init_pp[1]
                 
                            init_state_pp = T.tanh(LN_x_lstm_mlp_init_state_pp + self.init_hidden_mlp_b_pp)
                            init_cell_pp = T.tanh(LN_x_lstm_mlp_init_cell_pp + self.init_cell_mlp_b_pp)
                        else:
                            init_state_pp = T.dot(x_lstm_init_mlp_pp, self.init_hidden_mlp_h_pp) + self.init_hidden_mlp_b_pp
                            init_cell_pp = T.dot(x_lstm_init_mlp_pp, self.init_cell_mlp_h_pp) + self.init_cell_mlp_b_pp  
                    else:
                        init_state_pp = np.zeros((self.lstm_hidden_dim,), dtype=theano.config.floatX)
                        init_cell_pp = np.zeros((self.lstm_hidden_dim,), dtype=theano.config.floatX)
                        
                    if self.LN == True:
                        [f_s_pp, f_c_pp], _ = theano.scan(fn=self.LSTM_LN, 
                                                    sequences=pp_forward_lstm_input,
                                                    outputs_info=[init_state_pp, 
                                                                  init_cell_pp],
                                                    non_sequences=[T.concatenate([self.U_pp[0],self.U_pp[1],self.U_pp[2],self.U_pp[3]], axis=1),
                                                                   T.concatenate([self.W_pp[0],self.W_pp[1],self.W_pp[2],self.W_pp[3]], axis=1),
                                                                   self.b_pp[0],self.b_pp[1],self.b_pp[2],self.b_pp[3],
                                                                   self.LN_g_LSTM_pp[0], self.LN_g_LSTM_pp[1], self.LN_g_LSTM_c_pp[0],
                                                                   self.LN_b_LSTM_pp[0], self.LN_b_LSTM_pp[1], self.LN_b_LSTM_c_pp[0],
                                                                   np.int32(self.lstm_hidden_dim)
                                                                   ])
              
                        [b_s_pp, b_c_pp], _ = theano.scan(fn=self.LSTM_LN, 
                                                    sequences=pp_backward_lstm_input,
                                                    outputs_info=[init_state_pp, 
                                                                  init_cell_pp],                                                       
                                                    non_sequences=[T.concatenate([self.U_pp[4],self.U_pp[5],self.U_pp[6],self.U_pp[7]], axis=1),
                                                                   T.concatenate([self.W_pp[4],self.W_pp[5],self.W_pp[6],self.W_pp[7]], axis=1),
                                                                   self.b_pp[4],self.b_pp[5],self.b_pp[6],self.b_pp[7],
                                                                   self.LN_g_LSTM_pp[2], self.LN_g_LSTM_pp[3], self.LN_g_LSTM_c_pp[1],
                                                                   self.LN_b_LSTM_pp[2], self.LN_b_LSTM_pp[3], self.LN_b_LSTM_c_pp[1],
                                                                   np.int32(self.lstm_hidden_dim)
                                                                   ])                
                    else:
                        [f_s_pp, f_c_pp], _ = theano.scan(fn=self.LSTM, 
                                                    sequences=pp_forward_lstm_input,
                                                    outputs_info=[init_state_pp, 
                                                                  init_cell_pp],
                                                    non_sequences=[self.U_pp[0],self.U_pp[1],self.U_pp[2],self.U_pp[3],
                                                                   self.W_pp[0],self.W_pp[1],self.W_pp[2],self.W_pp[3],
                                                                   self.b_pp[0],self.b_pp[1],self.b_pp[2],self.b_pp[3]
                                                                   ])
              
                        [b_s_pp, b_c_pp], _ = theano.scan(fn=self.LSTM, 
                                                    sequences=pp_backward_lstm_input,
                                                    outputs_info=[init_state_pp, 
                                                                  init_cell_pp],                                                       
                                                    non_sequences=[self.U_pp[4],self.U_pp[5],self.U_pp[6],self.U_pp[7],
                                                                   self.W_pp[4],self.W_pp[5],self.W_pp[6],self.W_pp[7],
                                                                   self.b_pp[4],self.b_pp[5],self.b_pp[6],self.b_pp[7]
                                                                   ])                
                        
                    s_pp = T.cast(T.concatenate([f_s_pp, b_s_pp[::-1]], axis=1), theano.config.floatX)
                else:
                    s_pp = s1
                if self.double_unary == False:
                    s_pp_head = s_pp[1:]
                    s_pp_back = s_pp[:-1]
                    s_pp = T.concatenate([s_pp_head, s_pp_back], axis=1)
                    if self.pp_addition == True:
    #                     s_pp_delta = T.abs_(s_pp_head-s_pp_back)
    #                     s_pp_delta = s_pp_head-s_pp_back
                        s_pp_dot = s_pp_head*s_pp_back
    #                     s_pp = T.concatenate([s_pp_delta, s_pp_dot, s_pp_head, s_pp_back], axis=1)
                        s_pp = T.concatenate([s_pp_dot, s_pp_head, s_pp_back], axis=1)
                if self.output_mlp == True:
                    pp = T.dot(T.tanh(T.dot(s_pp, self.mlp_V_pp) + self.mlp_V_pp_b), self.V_P) + self.c_P
                else:
                    pp = T.dot(s_pp, self.V_P) + self.c_P
#             all_paths_scores_u, all_paths_scores_pp, all_paths_scores_lop
#             loss, Z_u, Z_pp, Z_lop = self.crf.fprop(unary, pp, label_sequence, mode='train')
#             model = T.nnet.softmax(unary)
#             loss_u = -T.mean(T.log(model)[T.arange(label_sequence.shape[0]), label_sequence])

                if self.three_unary == True:
                    if self.cnn_on == True and self.pos_on == True:
                        u_token_cnn_sequence, _ = theano.scan(fn=self.third_unary_character_cnn_step, sequences=token_sentence)
                        u_input = T.cast(T.concatenate([pp_token_cnn_sequence, self.E[word_sentence], self.POS[pos_sentence]], axis=1), theano.config.floatX)                    
                    elif self.cnn_on == True:
                        u_token_cnn_sequence, _ = theano.scan(fn=self.third_unary_character_cnn_step, sequences=token_sentence)
                        u_input = T.cast(T.concatenate([pp_token_cnn_sequence, self.E[word_sentence]], axis=1), theano.config.floatX)                    
                    elif self.pos_on == True:
                        u_input = T.cast(T.concatenate([self.E[word_sentence], self.POS[pos_sentence]], axis=1), theano.config.floatX)
                    else:
                        u_input = T.cast(self.E[word_sentence], theano.config.floatX)
         
                    u_forward_lstm_input = u_input
                    u_backward_lstm_input = u_input[::-1]
          
                    if self.initial == True:
                        if self.pos_on == True:
                            x_lstm_init_mlp_u = T.concatenate([self.E[word_sentence].mean(axis=0).astype(theano.config.floatX), self.POS[pos_sentence].mean(axis=0).astype(theano.config.floatX)], axis=0)
                        else:
                            x_lstm_init_mlp_u = T.concatenate([self.E[word_sentence].mean(axis=0).astype(theano.config.floatX), self.POS[pos_sentence].mean(axis=0).astype(theano.config.floatX)*0.], axis=0)
                        if self.LN == True:
                            pre_x_lstm_mlp_init_state_u = T.dot(x_lstm_init_mlp_u, self.init_hidden_mlp_h_u)
                            STD_x_lstm_mlp_init_state_u = (pre_x_lstm_mlp_init_state_u - pre_x_lstm_mlp_init_state_u.mean())/T.sqrt(pre_x_lstm_mlp_init_state_u.var()+self._eps)
                            LN_x_lstm_mlp_init_state_u = self.LN_g_LSTM_init_u[0]*STD_x_lstm_mlp_init_state_u + self.LN_b_LSTM_init_u[0]
                             
                            pre_x_lstm_mlp_init_cell_u = T.dot(x_lstm_init_mlp_u, self.init_cell_mlp_h_u)
                            STD_x_lstm_mlp_init_cell_u = (pre_x_lstm_mlp_init_cell_u - pre_x_lstm_mlp_init_cell_u.mean())/T.sqrt(pre_x_lstm_mlp_init_cell_u.var()+self._eps)
                            LN_x_lstm_mlp_init_cell_u = self.LN_g_LSTM_init_u[1]*STD_x_lstm_mlp_init_cell_u + self.LN_b_LSTM_init_u[1]
                 
                            init_state_u = T.tanh(LN_x_lstm_mlp_init_state_u + self.init_hidden_mlp_b_u)
                            init_cell_u = T.tanh(LN_x_lstm_mlp_init_cell_u + self.init_cell_mlp_b_u)
                        else:
                            init_state_u = T.dot(x_lstm_init_mlp_u, self.init_hidden_mlp_h_u) + self.init_hidden_mlp_b_u
                            init_cell_u = T.dot(x_lstm_init_mlp_u, self.init_cell_mlp_h_u) + self.init_cell_mlp_b_u  
                    else:
                        init_state_u = np.zeros((self.lstm_hidden_dim,), dtype=theano.config.floatX)
                        init_cell_u = np.zeros((self.lstm_hidden_dim,), dtype=theano.config.floatX)
                    
                    if self.LN == True:
                        [f_s_u, f_c_u], _ = theano.scan(fn=self.LSTM_LN, 
                                                    sequences=u_forward_lstm_input,
                                                    outputs_info=[init_state_u, 
                                                                  init_cell_u],
                                                    non_sequences=[T.concatenate([self.U_u[0],self.U_u[1],self.U_u[2],self.U_u[3]], axis=1),
                                                                   T.concatenate([self.W_u[0],self.W_u[1],self.W_u[2],self.W_u[3]], axis=1),
                                                                   self.b_u[0],self.b_u[1],self.b_u[2],self.b_u[3],
                                                                   self.LN_g_LSTM_u[0], self.LN_g_LSTM_u[1], self.LN_g_LSTM_c_u[0],
                                                                   self.LN_b_LSTM_u[0], self.LN_b_LSTM_u[1], self.LN_b_LSTM_c_u[0],
                                                                   np.int32(self.lstm_hidden_dim)
                                                                   ])
              
                        [b_s_u, b_c_u], _ = theano.scan(fn=self.LSTM_LN, 
                                                    sequences=u_backward_lstm_input,
                                                    outputs_info=[init_state_u, 
                                                                  init_cell_u],                                                       
                                                    non_sequences=[T.concatenate([self.U_u[4],self.U_u[5],self.U_u[6],self.U_u[7]], axis=1),
                                                                   T.concatenate([self.W_u[4],self.W_u[5],self.W_u[6],self.W_u[7]], axis=1),
                                                                   self.b_u[4],self.b_u[5],self.b_u[6],self.b_u[7],
                                                                   self.LN_g_LSTM_u[2], self.LN_g_LSTM_u[3], self.LN_g_LSTM_c_u[1],
                                                                   self.LN_b_LSTM_u[2], self.LN_b_LSTM_u[3], self.LN_b_LSTM_c_u[1],
                                                                   np.int32(self.lstm_hidden_dim)
                                                                   ])                
                    else:
                        [f_s_u, f_c_u], _ = theano.scan(fn=self.LSTM, 
                                                    sequences=u_forward_lstm_input,
                                                    outputs_info=[init_state_u, 
                                                                  init_cell_u],
                                                    non_sequences=[self.U_u[0],self.U_u[1],self.U_u[2],self.U_u[3],
                                                                   self.W_u[0],self.W_u[1],self.W_u[2],self.W_u[3],
                                                                   self.b_u[0],self.b_u[1],self.b_u[2],self.b_u[3]
                                                                   ])
              
                        [b_s_u, b_c_u], _ = theano.scan(fn=self.LSTM, 
                                                    sequences=u_backward_lstm_input,
                                                    outputs_info=[init_state_u, 
                                                                  init_cell_u],                                                       
                                                    non_sequences=[self.U_u[4],self.U_u[5],self.U_u[6],self.U_u[7],
                                                                   self.W_u[4],self.W_u[5],self.W_u[6],self.W_u[7],
                                                                   self.b_u[4],self.b_u[5],self.b_u[6],self.b_u[7]
                                                                   ])                
                        
                    s_u = T.cast(T.concatenate([f_s_u, b_s_u[::-1]], axis=1), theano.config.floatX)
                    unary2 = T.dot(s_u, self.V_u) + self.c_u
            
            if self.multiview == True:
                loss_u = self.crf_u.fprop(unary, label_sequence, mode='train')
                loss_pp = self.crf.fprop(pp, label_sequence, mode='train')
                loss = loss_u + loss_pp
            elif self.unary_view == True:
                loss = self.crf.fprop(unary, label_sequence, mode='train')
                loss_u = loss
                loss_pp = loss
            elif self.double_unary == True and self.three_unary == False:
                loss_u = self.crf_u.fprop(unary, label_sequence, mode='train')
                loss_pp = self.crf.fprop(pp, label_sequence, mode='train')
                loss = loss_u + loss_pp
            elif self.three_unary == True:
                loss_u = self.crf_u.fprop(unary, label_sequence, mode='train')
                loss_pp = self.crf.fprop(pp, label_sequence, mode='train')
                loss_uu = self.crf_uu.fprop(unary2, label_sequence, mode='train') 
                loss = loss_u + loss_pp + loss_uu
            else:
                loss, loss_u, loss_pp  = self.crf.fprop(unary, pp, label_sequence, mode='train')
            
            return loss, loss_u, loss_pp
        
        scan_input = [word_sentences, lengths, pos_sentences, labels]
        if self.cnn_on == True:
            scan_input.append(char_sentences)
        [_loss, _loss_u, _loss_pp], _ = theano.scan(fn=SentenceStep, sequences=scan_input)

        loss = T.mean(_loss)
        loss_u = T.mean(_loss_u)
        loss_pp = T.mean(_loss_pp)  
        
        if self.cnn_on == True:
            dLoss_dP = T.sqrt(T.sum(T.grad(loss, self.P)**2))
        else:
            dLoss_dP = loss
#         dLoss_dP += T.sqrt(T.sum(T.grad(loss_pp, self.P)**2))
        updates, norm, pp_norm = self.adam(loss, params, learning_rate=0.001, b1=0.9, b2=0.999, e=1e-6, gamma=1-1e-8, gradient_threshold=self.gradient_threshold)
#         updates_pp, norm, pp_norm = self.adam(loss_pp, params_pp, learning_rate=0.001, b1=0.9, b2=0.999, e=1e-6, gamma=1-1e-8, gradient_threshold=self.gradient_threshold)
#         updates.append(updates_pp)
#         updates_u, norm_u, V_norm_u = self.adam_u(loss_u, params_u, learning_rate=0.001, b1=0.9, b2=0.999, e=1e-6, gamma=1-1e-8, gradient_threshold=self.gradient_threshold)
#         updates_pp, norm_pp, V_norm_pp = self.adam_pp(loss_pp, params_pp, learning_rate=0.001, b1=0.9, b2=0.999, e=1e-6, gamma=1-1e-8, gradient_threshold=self.gradient_threshold)

#         updates = []
#         for entry in updates_u:
#             updates.append(entry)
#         for entry in updates_pp:
#             updates.append(entry)
        if self.cnn_on == True:
            self.trainer = theano.function([indexes],
                                           [loss, norm, pp_norm, loss_u, loss_pp, dLoss_dP], 
                                           updates=updates, 
                                           allow_input_downcast=True,
                                           givens={
                                               word_sentences : self.train_words[indexes], 
                                               lengths : self.train_lengths[indexes], 
                                               char_sentences : self.train_char_samples[indexes],
                                               pos_sentences : self.train_pos[indexes],
                                               labels : self.train_targets[indexes]
                                           })
        else:
            self.trainer = theano.function([indexes],
                                           [loss, norm, pp_norm, loss_u, loss_pp, dLoss_dP], 
                                           updates=updates, 
                                           allow_input_downcast=True,
                                           givens={
                                               word_sentences : self.train_words[indexes], 
                                               lengths : self.train_lengths[indexes], 
                                               pos_sentences : self.train_pos[indexes],
                                               labels : self.train_targets[indexes]
                                           })

    def test_build(self):
        index = T.iscalar('index')
        word_sentences = T.ivector('word_sentences')
        length = T.iscalar('length')
        pos_sentences = T.ivector('pos_sentences')
        if self.cnn_on == True:
            char_sentences = T.imatrix('char_sentences')
            token_sentence = char_sentences[:length]

        word_sentence = word_sentences[:length]
        pos_sentence = pos_sentences[:length]

        if self.cnn_on == True and self.pos_on == True:        
            token_cnn_sequence, _ = theano.scan(fn=self.unary_character_cnn_step, sequences=token_sentence)
            input = T.cast(T.concatenate([token_cnn_sequence, self.E[word_sentence], self.POS[pos_sentence]], axis=1), theano.config.floatX)
        elif self.cnn_on == True:                    
            token_cnn_sequence, _ = theano.scan(fn=self.unary_character_cnn_step, sequences=token_sentence)
            input = T.cast(T.concatenate([token_cnn_sequence, self.E[word_sentence]], axis=1), theano.config.floatX)
        elif self.pos_on == True:
            input = T.cast(T.concatenate([self.E[word_sentence], self.POS[pos_sentence]], axis=1), theano.config.floatX)
        else:
            input = T.cast(self.E[word_sentence], theano.config.floatX)

        forward_lstm_input = input
        backward_lstm_input = input[::-1]
 
        if self.initial == True:
            if self.pos_on == True:
                x_lstm_init_mlp = T.concatenate([self.E[word_sentence].mean(axis=0).astype(theano.config.floatX), self.POS[pos_sentence].mean(axis=0).astype(theano.config.floatX)], axis=0)
            else:
                x_lstm_init_mlp = T.concatenate([self.E[word_sentence].mean(axis=0).astype(theano.config.floatX), self.POS[pos_sentence].mean(axis=0).astype(theano.config.floatX)*0.], axis=0)
            if self.LN == True:
                pre_x_lstm_mlp_init_state = T.dot(x_lstm_init_mlp, self.init_hidden_mlp_h)
                STD_x_lstm_mlp_init_state = (pre_x_lstm_mlp_init_state - pre_x_lstm_mlp_init_state.mean())/T.sqrt(pre_x_lstm_mlp_init_state.var()+self._eps)
                LN_x_lstm_mlp_init_state = self.LN_g_LSTM_init[0]*STD_x_lstm_mlp_init_state + self.LN_b_LSTM_init[0]
                 
                pre_x_lstm_mlp_init_cell = T.dot(x_lstm_init_mlp, self.init_cell_mlp_h)
                STD_x_lstm_mlp_init_cell = (pre_x_lstm_mlp_init_cell - pre_x_lstm_mlp_init_cell.mean())/T.sqrt(pre_x_lstm_mlp_init_cell.var()+self._eps)
                LN_x_lstm_mlp_init_cell = self.LN_g_LSTM_init[1]*STD_x_lstm_mlp_init_cell + self.LN_b_LSTM_init[1]
                 
                init_state = T.tanh(LN_x_lstm_mlp_init_state + self.init_hidden_mlp_b)
                init_cell = T.tanh(LN_x_lstm_mlp_init_cell + self.init_cell_mlp_b)
            else:
                init_state = T.dot(x_lstm_init_mlp, self.init_hidden_mlp_h) + self.init_hidden_mlp_b
                init_cell = T.dot(x_lstm_init_mlp, self.init_cell_mlp_h) + self.init_cell_mlp_b
        else:
            init_state = np.zeros((self.lstm_hidden_dim,), dtype=theano.config.floatX)
            init_cell = np.zeros((self.lstm_hidden_dim,), dtype=theano.config.floatX)
        
        if self.LN == True:
            [f_s, f_c], _ = theano.scan(fn=self.LSTM_LN, 
                                        sequences=forward_lstm_input,
                                        outputs_info=[init_state, 
                                                      init_cell],
                                        non_sequences=[T.concatenate([self.U[0],self.U[1],self.U[2],self.U[3]], axis=1),
                                                       T.concatenate([self.W[0],self.W[1],self.W[2],self.W[3]], axis=1),
                                                       self.b[0],self.b[1],self.b[2],self.b[3],
                                                       self.LN_g_LSTM[0], self.LN_g_LSTM[1], self.LN_g_LSTM_c[0],
                                                       self.LN_b_LSTM[0], self.LN_b_LSTM[1], self.LN_b_LSTM_c[0],
                                                       np.int32(self.lstm_hidden_dim)
                                                       ])
     
            [b_s, b_c], _ = theano.scan(fn=self.LSTM_LN, 
                                        sequences=backward_lstm_input,
                                        outputs_info=[init_state, 
                                                      init_cell],                                                       
                                        non_sequences=[T.concatenate([self.U[4],self.U[5],self.U[6],self.U[7]], axis=1),
                                                       T.concatenate([self.W[4],self.W[5],self.W[6],self.W[7]], axis=1),
                                                       self.b[4],self.b[5],self.b[6],self.b[7],
                                                       self.LN_g_LSTM[2], self.LN_g_LSTM[3], self.LN_g_LSTM_c[1],
                                                       self.LN_b_LSTM[2], self.LN_b_LSTM[3], self.LN_b_LSTM_c[1],
                                                       np.int32(self.lstm_hidden_dim)
                                                       ])                
        else:
            [f_s, f_c], _ = theano.scan(fn=self.LSTM, 
                                        sequences=forward_lstm_input,
                                        outputs_info=[init_state, 
                                                      init_cell],
                                        non_sequences=[self.U[0],self.U[1],self.U[2],self.U[3],
                                                       self.W[0],self.W[1],self.W[2],self.W[3], 
                                                       self.b[0],self.b[1],self.b[2],self.b[3]
                                                       ])
     
            [b_s, b_c], _ = theano.scan(fn=self.LSTM, 
                                        sequences=backward_lstm_input,
                                        outputs_info=[init_state, 
                                                      init_cell],                                                       
                                        non_sequences=[self.U[4],self.U[5],self.U[6],self.U[7],
                                                       self.W[4],self.W[5],self.W[6],self.W[7],
                                                       self.b[4],self.b[5],self.b[6],self.b[7]
                                                       ])                
            
        s1 = T.cast(T.concatenate([f_s, b_s[::-1]], axis=1), theano.config.floatX)
        if self.output_mlp == True:
            unary = T.dot(T.tanh(T.dot(s1, self.mlp_V) + self.mlp_V_b), self.V) + self.c 
        else:
            unary = T.dot(s1, self.V) + self.c 

        if self.unary_view == False:
            if self.separate_deep == True:
                if self.cnn_on == True and self.pos_on == True:
                    pp_token_cnn_sequence, _ = theano.scan(fn=self.pp_character_cnn_step, sequences=token_sentence)
                    pp_input = T.cast(T.concatenate([pp_token_cnn_sequence, self.E[word_sentence], self.POS[pos_sentence]], axis=1), theano.config.floatX)                    
                elif self.cnn_on == True:
                    pp_token_cnn_sequence, _ = theano.scan(fn=self.pp_character_cnn_step, sequences=token_sentence)
                    pp_input = T.cast(T.concatenate([pp_token_cnn_sequence, self.E[word_sentence]], axis=1), theano.config.floatX)                    
                elif self.pos_on == True:
                    pp_input = T.cast(T.concatenate([self.E[word_sentence], self.POS[pos_sentence]], axis=1), theano.config.floatX)       
                else:
                    pp_input = T.cast(self.E[word_sentence], theano.config.floatX)
         
                pp_forward_lstm_input = pp_input
                pp_backward_lstm_input = pp_input[::-1]
          
                if self.initial == True:
                    if self.pos_on == True:
                        x_lstm_init_mlp_pp = T.concatenate([self.E[word_sentence].mean(axis=0).astype(theano.config.floatX), self.POS[pos_sentence].mean(axis=0).astype(theano.config.floatX)], axis=0)
                    else:
                        x_lstm_init_mlp_pp = T.concatenate([self.E[word_sentence].mean(axis=0).astype(theano.config.floatX), self.POS[pos_sentence].mean(axis=0).astype(theano.config.floatX)*0.], axis=0)
                    if self.LN == True:
                        pre_x_lstm_mlp_init_state_pp = T.dot(x_lstm_init_mlp_pp, self.init_hidden_mlp_h_pp)
                        STD_x_lstm_mlp_init_state_pp = (pre_x_lstm_mlp_init_state_pp - pre_x_lstm_mlp_init_state_pp.mean())/T.sqrt(pre_x_lstm_mlp_init_state_pp.var()+self._eps)
                        LN_x_lstm_mlp_init_state_pp = self.LN_g_LSTM_init_pp[0]*STD_x_lstm_mlp_init_state_pp + self.LN_b_LSTM_init_pp[0]
                         
                        pre_x_lstm_mlp_init_cell_pp = T.dot(x_lstm_init_mlp_pp, self.init_cell_mlp_h_pp)
                        STD_x_lstm_mlp_init_cell_pp = (pre_x_lstm_mlp_init_cell_pp - pre_x_lstm_mlp_init_cell_pp.mean())/T.sqrt(pre_x_lstm_mlp_init_cell_pp.var()+self._eps)
                        LN_x_lstm_mlp_init_cell_pp = self.LN_g_LSTM_init_pp[1]*STD_x_lstm_mlp_init_cell_pp + self.LN_b_LSTM_init_pp[1]
                 
                        init_state_pp = T.tanh(LN_x_lstm_mlp_init_state_pp + self.init_hidden_mlp_b_pp)
                        init_cell_pp = T.tanh(LN_x_lstm_mlp_init_cell_pp + self.init_cell_mlp_b_pp)
                    else:
                        init_state_pp = T.dot(x_lstm_init_mlp_pp, self.init_hidden_mlp_h_pp) + self.init_hidden_mlp_b_pp
                        init_cell_pp = T.dot(x_lstm_init_mlp_pp, self.init_cell_mlp_h_pp) + self.init_cell_mlp_b_pp 
                else:
                    init_state_pp = np.zeros((self.lstm_hidden_dim,), dtype=theano.config.floatX)
                    init_cell_pp = np.zeros((self.lstm_hidden_dim,), dtype=theano.config.floatX)
                     
                if self.LN == True:
                    [f_s_pp, f_c_pp], _ = theano.scan(fn=self.LSTM_LN, 
                                                sequences=pp_forward_lstm_input,
                                                outputs_info=[init_state_pp, 
                                                              init_cell_pp],
                                                non_sequences=[T.concatenate([self.U_pp[0],self.U_pp[1],self.U_pp[2],self.U_pp[3]], axis=1),
                                                               T.concatenate([self.W_pp[0],self.W_pp[1],self.W_pp[2],self.W_pp[3]], axis=1),
                                                               self.b_pp[0],self.b_pp[1],self.b_pp[2],self.b_pp[3],
                                                               self.LN_g_LSTM_pp[0], self.LN_g_LSTM_pp[1], self.LN_g_LSTM_c_pp[0],
                                                               self.LN_b_LSTM_pp[0], self.LN_b_LSTM_pp[1], self.LN_b_LSTM_c_pp[0],
                                                               np.int32(self.lstm_hidden_dim)
                                                               ])
              
                    [b_s_pp, b_c_pp], _ = theano.scan(fn=self.LSTM_LN, 
                                                sequences=pp_backward_lstm_input,
                                                outputs_info=[init_state_pp, 
                                                              init_cell_pp],                                                       
                                                non_sequences=[T.concatenate([self.U_pp[4],self.U_pp[5],self.U_pp[6],self.U_pp[7]], axis=1),
                                                               T.concatenate([self.W_pp[4],self.W_pp[5],self.W_pp[6],self.W_pp[7]], axis=1),
                                                               self.b_pp[4],self.b_pp[5],self.b_pp[6],self.b_pp[7],
                                                               self.LN_g_LSTM_pp[2], self.LN_g_LSTM_pp[3], self.LN_g_LSTM_c_pp[1],
                                                               self.LN_b_LSTM_pp[2], self.LN_b_LSTM_pp[3], self.LN_b_LSTM_c_pp[1],
                                                               np.int32(self.lstm_hidden_dim)
                                                               ])                
                else:
                    [f_s_pp, f_c_pp], _ = theano.scan(fn=self.LSTM, 
                                                sequences=pp_forward_lstm_input,
                                                outputs_info=[init_state_pp, 
                                                              init_cell_pp],
                                                non_sequences=[self.U_pp[0],self.U_pp[1],self.U_pp[2],self.U_pp[3],
                                                               self.W_pp[0],self.W_pp[1],self.W_pp[2],self.W_pp[3],
                                                               self.b_pp[0],self.b_pp[1],self.b_pp[2],self.b_pp[3]
                                                               ])
              
                    [b_s_pp, b_c_pp], _ = theano.scan(fn=self.LSTM, 
                                                sequences=pp_backward_lstm_input,
                                                outputs_info=[init_state_pp, 
                                                              init_cell_pp],                                                       
                                                non_sequences=[self.U_pp[4],self.U_pp[5],self.U_pp[6],self.U_pp[7],
                                                               self.W_pp[4],self.W_pp[5],self.W_pp[6],self.W_pp[7],
                                                               self.b_pp[4],self.b_pp[5],self.b_pp[6],self.b_pp[7]
                                                               ])                
                    
                s_pp = T.cast(T.concatenate([f_s_pp, b_s_pp[::-1]], axis=1), theano.config.floatX)
            else:
                s_pp = s1
            if self.double_unary == False:
                s_pp_head = s_pp[1:]
                s_pp_back = s_pp[:-1]
                s_pp = T.concatenate([s_pp_head, s_pp_back], axis=1)
                if self.pp_addition == True:
    #                 s_pp_delta = T.abs_(s_pp_head-s_pp_back)
    #                 s_pp_delta = s_pp_head-s_pp_back
                    s_pp_dot = s_pp_head*s_pp_back
    #                 s_pp = T.concatenate([s_pp_delta, s_pp_dot, s_pp_head, s_pp_back], axis=1)
                    s_pp = T.concatenate([s_pp_dot, s_pp_head, s_pp_back], axis=1)
            if self.output_mlp == True:
                pp = T.dot(T.tanh(T.dot(s_pp, self.mlp_V_pp) + self.mlp_V_pp_b), self.V_P) + self.c_P
            else:
                pp = T.dot(s_pp, self.V_P) + self.c_P

            if self.three_unary == True:
                if self.cnn_on == True and self.pos_on == True:
                    u_token_cnn_sequence, _ = theano.scan(fn=self.third_unary_character_cnn_step, sequences=token_sentence)
                    u_input = T.cast(T.concatenate([pp_token_cnn_sequence, self.E[word_sentence], self.POS[pos_sentence]], axis=1), theano.config.floatX)                    
                elif self.cnn_on == True:
                    u_token_cnn_sequence, _ = theano.scan(fn=self.third_unary_character_cnn_step, sequences=token_sentence)
                    u_input = T.cast(T.concatenate([pp_token_cnn_sequence, self.E[word_sentence]], axis=1), theano.config.floatX)                    
                elif self.pos_on == True:
                    u_input = T.cast(T.concatenate([self.E[word_sentence], self.POS[pos_sentence]], axis=1), theano.config.floatX)
                else:
                    u_input = T.cast(self.E[word_sentence], theano.config.floatX)
     
                u_forward_lstm_input = u_input
                u_backward_lstm_input = u_input[::-1]
      
                if self.initial == True:
                    if self.pos_on == True:
                        x_lstm_init_mlp_u = T.concatenate([self.E[word_sentence].mean(axis=0).astype(theano.config.floatX), self.POS[pos_sentence].mean(axis=0).astype(theano.config.floatX)], axis=0)
                    else:
                        x_lstm_init_mlp_u = T.concatenate([self.E[word_sentence].mean(axis=0).astype(theano.config.floatX), self.POS[pos_sentence].mean(axis=0).astype(theano.config.floatX)*0.], axis=0)
                    if self.LN == True:
                        pre_x_lstm_mlp_init_state_u = T.dot(x_lstm_init_mlp_u, self.init_hidden_mlp_h_u)
                        STD_x_lstm_mlp_init_state_u = (pre_x_lstm_mlp_init_state_u - pre_x_lstm_mlp_init_state_u.mean())/T.sqrt(pre_x_lstm_mlp_init_state_u.var()+self._eps)
                        LN_x_lstm_mlp_init_state_u = self.LN_g_LSTM_init_u[0]*STD_x_lstm_mlp_init_state_u + self.LN_b_LSTM_init_u[0]
                         
                        pre_x_lstm_mlp_init_cell_u = T.dot(x_lstm_init_mlp_u, self.init_cell_mlp_h_u)
                        STD_x_lstm_mlp_init_cell_u = (pre_x_lstm_mlp_init_cell_u - pre_x_lstm_mlp_init_cell_u.mean())/T.sqrt(pre_x_lstm_mlp_init_cell_u.var()+self._eps)
                        LN_x_lstm_mlp_init_cell_u = self.LN_g_LSTM_init_u[1]*STD_x_lstm_mlp_init_cell_u + self.LN_b_LSTM_init_u[1]
             
                        init_state_u = T.tanh(LN_x_lstm_mlp_init_state_u + self.init_hidden_mlp_b_u)
                        init_cell_u = T.tanh(LN_x_lstm_mlp_init_cell_u + self.init_cell_mlp_b_u)
                    else:
                        init_state_u = T.dot(x_lstm_init_mlp_u, self.init_hidden_mlp_h_u) + self.init_hidden_mlp_b_u
                        init_cell_u = T.dot(x_lstm_init_mlp_u, self.init_cell_mlp_h_u) + self.init_cell_mlp_b_u  
                else:
                    init_state_u = np.zeros((self.lstm_hidden_dim,), dtype=theano.config.floatX)
                    init_cell_u = np.zeros((self.lstm_hidden_dim,), dtype=theano.config.floatX)

                if self.LN == True:
                    [f_s_u, f_c_u], _ = theano.scan(fn=self.LSTM_LN, 
                                                sequences=u_forward_lstm_input,
                                                outputs_info=[init_state_u, 
                                                              init_cell_u],
                                                non_sequences=[T.concatenate([self.U_u[0],self.U_u[1],self.U_u[2],self.U_u[3]], axis=1),
                                                               T.concatenate([self.W_u[0],self.W_u[1],self.W_u[2],self.W_u[3]], axis=1),
                                                               self.b_u[0],self.b_u[1],self.b_u[2],self.b_u[3],
                                                               self.LN_g_LSTM_u[0], self.LN_g_LSTM_u[1], self.LN_g_LSTM_c_u[0],
                                                               self.LN_b_LSTM_u[0], self.LN_b_LSTM_u[1], self.LN_b_LSTM_c_u[0],
                                                               np.int32(self.lstm_hidden_dim)
                                                               ])
          
                    [b_s_u, b_c_u], _ = theano.scan(fn=self.LSTM_LN, 
                                                sequences=u_backward_lstm_input,
                                                outputs_info=[init_state_u, 
                                                              init_cell_u],                                                       
                                                non_sequences=[T.concatenate([self.U_u[4],self.U_u[5],self.U_u[6],self.U_u[7]], axis=1),
                                                               T.concatenate([self.W_u[4],self.W_u[5],self.W_u[6],self.W_u[7]], axis=1),
                                                               self.b_u[4],self.b_u[5],self.b_u[6],self.b_u[7],
                                                               self.LN_g_LSTM_u[2], self.LN_g_LSTM_u[3], self.LN_g_LSTM_c_u[1],
                                                               self.LN_b_LSTM_u[2], self.LN_b_LSTM_u[3], self.LN_b_LSTM_c_u[1],
                                                               np.int32(self.lstm_hidden_dim)
                                                               ])                
                else:
                    [f_s_u, f_c_u], _ = theano.scan(fn=self.LSTM, 
                                                sequences=u_forward_lstm_input,
                                                outputs_info=[init_state_u, 
                                                              init_cell_u],
                                                non_sequences=[self.U_u[0],self.U_u[1],self.U_u[2],self.U_u[3],
                                                               self.W_u[0],self.W_u[1],self.W_u[2],self.W_u[3],
                                                               self.b_u[0],self.b_u[1],self.b_u[2],self.b_u[3]
                                                               ])
          
                    [b_s_u, b_c_u], _ = theano.scan(fn=self.LSTM, 
                                                sequences=u_backward_lstm_input,
                                                outputs_info=[init_state_u, 
                                                              init_cell_u],                                                       
                                                non_sequences=[self.U_u[4],self.U_u[5],self.U_u[6],self.U_u[7],
                                                               self.W_u[4],self.W_u[5],self.W_u[6],self.W_u[7],
                                                               self.b_u[4],self.b_u[5],self.b_u[6],self.b_u[7]
                                                               ])                

                s_u = T.cast(T.concatenate([f_s_u, b_s_u[::-1]], axis=1), theano.config.floatX)
                unary2 = T.dot(s_u, self.V_u) + self.c_u

        if self.multiview == True:
            prediction = self.crf.sequence_predict(unary, pp, self.crf_u.params[0])
        elif self.unary_view == True:
            prediction = self.crf.fprop(unary, ground_truth=None, viterbi=True, return_best_sequence=True, mode='eval')
            pp = T.zeros((self.n_labels*self.n_labels,), dtype=theano.config.floatX)
        elif self.double_unary == True and self.three_unary == False:
            prediction = self.crf.sequence_predict(unary, pp, self.crf_u.params[0])
        elif self.three_unary == True:
            prediction = self.crf.sequence_predict_three_unary(unary, pp, unary2, self.crf_u.params[0], self.crf_uu.params[0])
        else:
            prediction = self.crf.fprop(unary, pp, ground_truth=None, viterbi=True, return_best_sequence=True, mode='eval')

        if self.cnn_on == True:
            self.testing = theano.function([index],
                                           [prediction, unary, pp], 
                                           allow_input_downcast=True,
                                           givens={
                                               word_sentences : self.test_words[index], 
                                               length : self.test_lengths[index], 
                                               char_sentences : self.test_char_samples[index],
                                               pos_sentences : self.test_pos[index]
                                           })
        else:
            self.testing = theano.function([index],
                                       [prediction, unary, pp], 
                                       allow_input_downcast=True,
                                       givens={
                                           word_sentences : self.test_words[index], 
                                           length : self.test_lengths[index], 
                                           pos_sentences : self.test_pos[index]
                                       })


    def train(self, indexes):
        loss, norm, pp_norm, loss_u, loss_pp, P_g = self.trainer(indexes)
        return loss, norm, pp_norm, loss_u, loss_pp, P_g

    def testing(self, index):
        prediction, unary_score, pp_score = self.testing(index)
        return prediction, unary_score, pp_score
                              
    def get_parameters(self):
        if self.without_transition_matrix == False:
            if self.multiview == True:
                return self.crf_u.transition_matrix.get_value()
            if self.unary_view == True:
                return self.crf.transition_matrix.get_value()
            if self.double_unary == True:
                return self.crf_u.transition_matrix.get_value()
            if self.three_unary == True:
                return self.crf_u.transition_matrix.get_value()
        return None

    def gradient_norm(self, gs):
        norm = T.sum([T.sqrt(T.sum(g**2)) for g in gs])
        return norm

    def clip_norms(self, gs, norm, c):
        return [theano.ifelse.ifelse(T.ge(norm, c), g*c/norm, g) for g in gs]
    
    def momentum(self, loss, all_params, learning_rate=0.01, momentum=0.9):
        updates = []
        all_grads = T.grad(loss, all_params)
        all_grads = clip_norms(all_grads, 50.)
        for p, g in zip(all_params, all_grads):
            m = theano.shared(np.zeros(p.get_value().shape, dtype=theano.config.floatX).astype(theano.config.floatX))
            v = (momentum*m) - (learning_rate*g)

            updates.append((m, v))
            updates.append((p, p + v))

        return updates

    def adam(self, loss, all_params, learning_rate=0.001, b1=0.9, b2=0.999, e=1e-6, gamma=1-1e-8, gradient_threshold=10.0):
        updates = []
        all_grads = T.grad(loss, all_params)
        norm = self.gradient_norm(all_grads)
        if self.unary_view == False:
            pp_norm = T.sqrt(T.sum(T.grad(loss, self.V_P)**2))
        else:
            pp_norm = T.sqrt(T.sum(T.grad(loss, self.V)**2))
        print 'gradient norm threshold = ', gradient_threshold
        all_grads = self.clip_norms(all_grads, norm, gradient_threshold)
        alpha = learning_rate
        t = theano.shared(np.float32(1.).astype(theano.config.floatX))
        b1_t = b1*gamma**(t-1.)   #(Decay the first moment running average coefficient)
    
        for theta_previous, g in zip(all_params, all_grads):
            m_previous = theano.shared(np.zeros(theta_previous.get_value().shape, dtype=theano.config.floatX).astype(theano.config.floatX))
            v_previous = theano.shared(np.zeros(theta_previous.get_value().shape, dtype=theano.config.floatX).astype(theano.config.floatX))
            
            m = b1_t*m_previous + (1. - b1_t)*g                             # (Update biased first moment estimate)
            v = b2*v_previous + (1. - b2)*g**2.                              # (Update biased second raw moment estimate)
            m_hat = m / (1.-b1**t)                                          # (Compute bias-corrected first moment estimate)
            v_hat = v / (1.-b2**t)                                          # (Compute bias-corrected second raw moment estimate)
            theta = theta_previous - (alpha * m_hat) / (T.sqrt(v_hat) + e) #(Update parameters)
    
            updates.append((m_previous, m))
            updates.append((v_previous, v))
            updates.append((theta_previous, theta))
        updates.append((t, t + 1.))
        
        return updates, norm, pp_norm
    
    def adam_u(self, loss, all_params, learning_rate=0.001, b1=0.9, b2=0.999, e=1e-6, gamma=1-1e-8, gradient_threshold=10.0):
        updates = []
        all_grads = T.grad(loss, all_params)
        norm = self.gradient_norm(all_grads)
        pp_norm = T.sqrt(T.sum(T.grad(loss, self.V)**2))
        print 'gradient norm threshold = ', gradient_threshold
        all_grads = self.clip_norms(all_grads, norm, gradient_threshold)
        alpha = learning_rate
        t = theano.shared(np.float32(1.).astype(theano.config.floatX))
        b1_t = b1*gamma**(t-1.)   #(Decay the first moment running average coefficient)
    
        for theta_previous, g in zip(all_params, all_grads):
            m_previous = theano.shared(np.zeros(theta_previous.get_value().shape, dtype=theano.config.floatX).astype(theano.config.floatX))
            v_previous = theano.shared(np.zeros(theta_previous.get_value().shape, dtype=theano.config.floatX).astype(theano.config.floatX))
            
            m = b1_t*m_previous + (1. - b1_t)*g                             # (Update biased first moment estimate)
            v = b2*v_previous + (1. - b2)*g**2.                              # (Update biased second raw moment estimate)
            m_hat = m / (1.-b1**t)                                          # (Compute bias-corrected first moment estimate)
            v_hat = v / (1.-b2**t)                                          # (Compute bias-corrected second raw moment estimate)
            theta = theta_previous - (alpha * m_hat) / (T.sqrt(v_hat) + e) #(Update parameters)
    
            updates.append((m_previous, m))
            updates.append((v_previous, v))
            updates.append((theta_previous, theta))
        updates.append((t, t + 1.))
        
        return updates, norm, pp_norm

    def adam_pp(self, loss, all_params, learning_rate=0.001, b1=0.9, b2=0.999, e=1e-6, gamma=1-1e-8, gradient_threshold=10.0):
        updates = []
        all_grads = T.grad(loss, all_params)
        norm = self.gradient_norm(all_grads)
        pp_norm = T.sqrt(T.sum(T.grad(loss, self.V_P)**2))
        print 'gradient norm threshold = ', gradient_threshold
        all_grads = self.clip_norms(all_grads, norm, gradient_threshold)
        alpha = learning_rate
        t = theano.shared(np.float32(1.).astype(theano.config.floatX))
        b1_t = b1*gamma**(t-1.)   #(Decay the first moment running average coefficient)
    
        for theta_previous, g in zip(all_params, all_grads):
            m_previous = theano.shared(np.zeros(theta_previous.get_value().shape, dtype=theano.config.floatX).astype(theano.config.floatX))
            v_previous = theano.shared(np.zeros(theta_previous.get_value().shape, dtype=theano.config.floatX).astype(theano.config.floatX))
            
            m = b1_t*m_previous + (1. - b1_t)*g                             # (Update biased first moment estimate)
            v = b2*v_previous + (1. - b2)*g**2.                              # (Update biased second raw moment estimate)
            m_hat = m / (1.-b1**t)                                          # (Compute bias-corrected first moment estimate)
            v_hat = v / (1.-b2**t)                                          # (Compute bias-corrected second raw moment estimate)
            theta = theta_previous - (alpha * m_hat) / (T.sqrt(v_hat) + e) #(Update parameters)
    
            updates.append((m_previous, m))
            updates.append((v_previous, v))
            updates.append((theta_previous, theta))
        updates.append((t, t + 1.))
        
        return updates, norm, pp_norm

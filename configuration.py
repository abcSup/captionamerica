class Config(object):

    def __init__(self):
        self.vocab_size = 13000

        longest = 50

        self.seq_len = longest + 1

        self.cap_len = longest + 2
        
        self.batch_size = 32

        self.num_batch = 128

        self.emb_size = 512

        self.num_lstm = 512

        self.dropout = False

        self.beam_size = 5

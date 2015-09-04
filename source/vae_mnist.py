from chainer import FunctionSet,Variable
import chainer.functions as F
import numpy
from vae_model import VAE_bernoulli,VAE_gaussian

class VAE_MNIST(VAE_gaussian):
    def __init__(self):
        super(VAE_MNIST,self).__init__(
            enc_l1 = F.Linear(784,500),
            enc_l_mu = F.Linear(500,30),
            enc_l_log_sig_2 = F.Linear(500,30),
            dec_l1 = F.Linear(30,500),
            dec_l_mu = F.Linear(500,784),
            dec_l_log_sig_2 = F.Linear(500,784))

    def encode(self,x):
        h = x
        h = self.enc_l1(h)
        h = F.relu(h)
        return self.enc_l_mu(h),self.enc_l_log_sig_2(h)

    def decode(self,z):
        h = z
        h = self.dec_l1(h)
        h = F.relu(h)
        return self.dec_l_mu(h),self.dec_l_log_sig_2(h)

class VAE_MNIST_b(VAE_bernoulli):
    def __init__(self):
        super(VAE_MNIST_b,self).__init__(
            enc_l1 = F.Linear(784,500),
            enc_l2 = F.Linear(500,500),
            enc_l_mu = F.Linear(500,30),
            enc_l_log_sig_2 = F.Linear(500,30),
            dec_l1 = F.Linear(30,500),
            dec_l2 = F.Linear(500,500),
            dec_l_mu = F.Linear(500,784))

    def encode(self,x):
        h = x
        h = self.enc_l1(h)
        h = F.relu(h)
        h = self.enc_l2(h)
        h = F.relu(h)
        return self.enc_l_mu(h),self.enc_l_log_sig_2(h)

    def decode(self,z):
        h = z
        h = self.dec_l1(h)
        h = F.relu(h)
        h = self.dec_l2(h)
        h = F.relu(h)
        return self.dec_l_mu(h)




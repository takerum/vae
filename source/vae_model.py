from chainer import FunctionSet,Variable
import chainer.functions as F
import numpy


class VAE_gaussian(FunctionSet):

    # You must add attr named with 'enc_l1' and 'dec_l1'
    # for specification of the input and latent dimensions, in __init__ of FunctionSet.

    def encode(self,x):
        raise NotImplementedError()

    def decode(self,z):
        raise NotImplementedError()

    def free_energy(self,x):
        #return -(free energy)
        enc_mu, enc_log_sigma_2 = self.encode(x)
        kl = F.gaussian_kl_divergence(enc_mu,enc_log_sigma_2)
        z = F.gaussian(enc_mu,enc_log_sigma_2)
        dec_mu, dec_log_sigma_2 = self.decode(z)
        nll = F.gaussian_nll(x,dec_mu,dec_log_sigma_2)
        return nll+kl

    def generate(self,N,sampling_x=False):
        z_dim = self['dec_l1'].W.shape[1]
        if(isinstance(self['dec_l1'].W,numpy.ndarray)):
            zero_mat = Variable(numpy.zeros((N,z_dim),'float32'))
            z = F.gaussian(zero_mat,zero_mat)
        else:
            raise NotImplementedError()
        dec_mu, dec_log_sigma_2 = self.decode(z)
        if(sampling_x):
            x = F.gaussian(dec_mu,dec_log_sigma_2)
        else:
            x = dec_mu
        return x

    def reconstruct(self,x):
        enc_mu, enc_log_sigma_2 = self.encode(x)
        dec_mu, dec_log_sigma_2 = self.decode(enc_mu)
        return dec_mu

class VAE_bernoulli(VAE_gaussian):

    def free_energy(self,x):
        #return -(free energy)
        enc_mu, enc_log_sigma_2 = self.encode(x)
        kl = F.gaussian_kl_divergence(enc_mu,enc_log_sigma_2)
        z = F.gaussian(enc_mu,enc_log_sigma_2)
        dec_mu = self.decode(z)
        nll = F.bernoulli_nll(x,dec_mu)
        return nll+kl

    def generate(self,N,sampling_x=False):
        z_dim = self['dec_l1'].W.shape[1]
        if(isinstance(self['dec_l1'].W,numpy.ndarray)):
            zero_mat = Variable(numpy.zeros((N,z_dim),'float32'))
            z = F.gaussian(zero_mat,zero_mat)
        else:
            raise NotImplementedError()

        dec_mu = F.sigmoid(self.decode(z))
        if(sampling_x):
            raise NotImplementedError()
        else:
            x = dec_mu
        return x


    def reconstruct(self,x):
        enc_mu, enc_log_sigma_2 = self.encode(x)
        dec_mu = F.sigmoid(self.decode(enc_mu))
        return dec_mu



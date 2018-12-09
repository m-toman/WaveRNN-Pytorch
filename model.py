import torch
from torch import nn
import torch.nn.functional as F
from hparams import hparams as hp
from torch.utils.data import DataLoader, Dataset
from distributions import *
from utils import num_params

from tqdm import tqdm
import numpy as np

class ResBlock(nn.Module) :
    def __init__(self, dims) :
        super().__init__()
        self.conv1 = nn.Conv1d(dims, dims, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(dims, dims, kernel_size=1, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(dims)
        self.batch_norm2 = nn.BatchNorm1d(dims)
        
    def forward(self, x) :
        residual = x
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        return x + residual

class MelResNet(nn.Module) :
    def __init__(self, res_blocks, in_dims, compute_dims, res_out_dims) :
        super().__init__()
        assert hp.resnet_pad == (hp.resnet_kernel-1)//2 #padding has to match kernel overhang
        self.conv_in = nn.Conv1d(in_dims, compute_dims, kernel_size=hp.resnet_kernel, padding=hp.resnet_pad, bias=False)
        self.batch_norm = nn.BatchNorm1d(compute_dims)
        self.layers = nn.ModuleList()
        for i in range(res_blocks) :
            self.layers.append(ResBlock(compute_dims))
        self.conv_out = nn.Conv1d(compute_dims, res_out_dims, kernel_size=1)
        
    def forward(self, x) :
        x = self.conv_in(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        for f in self.layers:
            x = f(x)
        x = self.conv_out(x)
        return x


class UpsampleNetwork(nn.Module) :
    def __init__(self, feat_dims, upsample_scales, compute_dims, 
                 res_blocks, res_out_dims):
        super().__init__()
        total_scale = hp.hop_size

        self.resnet = MelResNet(res_blocks, feat_dims, compute_dims, res_out_dims)
        self.resnet_stretch = nn.Upsample(scale_factor=total_scale, mode='linear', align_corners=False)

        self.upsample = nn.Upsample(scale_factor=total_scale, mode='linear', align_corners=False)

    def forward(self, m):
        aux = self.resnet(m)
        aux = self.resnet_stretch(aux)

        m = self.upsample(m)
        #m = m.squeeze(1)[:, :, self.indent:-self.indent]
        return m.transpose(1, 2), aux.transpose(1, 2)


class Model(nn.Module) :
    def __init__(self, rnn_dims, fc_dims, bits, upsample_factors,
                 feat_dims, compute_dims, res_out_dims, res_blocks):
        super().__init__()
        self.n_classes = 2**bits
        self.rnn_dims = rnn_dims
        self.aux_dims = res_out_dims // 4
        self.upsample = UpsampleNetwork(feat_dims, upsample_factors, compute_dims, 
                                        res_blocks, res_out_dims)
        self.I = nn.Linear(feat_dims + self.aux_dims + 1, rnn_dims)
        self.rnn1 = nn.GRU(rnn_dims, rnn_dims, batch_first=True)
        self.rnn2 = nn.GRU(rnn_dims + self.aux_dims, rnn_dims, batch_first=True)
        self.fc1 = nn.Linear(rnn_dims + self.aux_dims, fc_dims)
        self.fc2 = nn.Linear(fc_dims + self.aux_dims, fc_dims)
        self.fc3 = nn.Linear(fc_dims, self.n_classes)
        num_params(self)
    
    def forward(self, x, mels) :
        bsize = x.size(0)
        h1 = torch.zeros(1, bsize, self.rnn_dims).cuda()
        h2 = torch.zeros(1, bsize, self.rnn_dims).cuda()
        mels, aux = self.upsample(mels)
        
        aux_idx = [self.aux_dims * i for i in range(5)]
        a1 = aux[:, :, aux_idx[0]:aux_idx[1]]
        a2 = aux[:, :, aux_idx[1]:aux_idx[2]]
        a3 = aux[:, :, aux_idx[2]:aux_idx[3]]
        a4 = aux[:, :, aux_idx[3]:aux_idx[4]]
        
        x = torch.cat([x.unsqueeze(-1), mels, a1], dim=2)
        x = self.I(x)
        res = x
        x, _ = self.rnn1(x, h1)
        
        x = x + res
        res = x
        x = torch.cat([x, a2], dim=2)
        x, _ = self.rnn2(x, h2)
        
        x = x + res
        x = torch.cat([x, a3], dim=2)
        x = F.relu(self.fc1(x))
        
        x = torch.cat([x, a4], dim=2)
        x = F.relu(self.fc2(x))

        x = self.fc3(x)

        return F.log_softmax(x, dim=-1)


    def preview_upsampling(self, mels) :
        mels, aux = self.upsample(mels)
        return mels, aux
    
    # def generate(self, mels) :
    #     self.eval()
    #     output = []
    #     rnn1 = self.get_gru_cell(self.rnn1)
    #     rnn2 = self.get_gru_cell(self.rnn2)
    #
    #     with torch.no_grad() :
    #         x = torch.zeros(1, 1).cuda()
    #         h1 = torch.zeros(1, self.rnn_dims).cuda()
    #         h2 = torch.zeros(1, self.rnn_dims).cuda()
    #
    #         mels = torch.FloatTensor(mels).cuda().unsqueeze(0)
    #         mels, aux = self.upsample(mels)
    #
    #         aux_idx = [self.aux_dims * i for i in range(5)]
    #         a1 = aux[:, :, aux_idx[0]:aux_idx[1]]
    #         a2 = aux[:, :, aux_idx[1]:aux_idx[2]]
    #         a3 = aux[:, :, aux_idx[2]:aux_idx[3]]
    #         a4 = aux[:, :, aux_idx[3]:aux_idx[4]]
    #
    #         seq_len = mels.size(1)
    #
    #         for i in tqdm(range(seq_len)) :
    #
    #             m_t = mels[:, i, :]
    #             a1_t = a1[:, i, :]
    #             a2_t = a2[:, i, :]
    #             a3_t = a3[:, i, :]
    #             a4_t = a4[:, i, :]
    #
    #             x = torch.cat([x, m_t, a1_t], dim=1)
    #             x = self.I(x)
    #             h1 = rnn1(x, h1)
    #
    #             x = x + h1
    #             inp = torch.cat([x, a2_t], dim=1)
    #             h2 = rnn2(inp, h2)
    #
    #             x = x + h2
    #             x = torch.cat([x, a3_t], dim=1)
    #             x = F.relu(self.fc1(x))
    #
    #             x = torch.cat([x, a4_t], dim=1)
    #             x = F.relu(self.fc2(x))
    #             x = self.fc3(x)

                # posterior = F.softmax(x, dim=1).view(-1)
                # distrib = torch.distributions.Categorical(posterior)
    #             output.append(sample.view(-1))
    #             x = torch.FloatTensor([[sample]]).cuda()
    #     output = torch.stack(output).cpu().numpy()
    #     self.train()
    #     return output

    def pad_tensor(self, x, pad, side='both') :
        # NB - this is just a quick method i need right now
        # i.e., it won't generalise to other shapes/dims
        b, t, c = x.size()
        total = t + 2 * pad if side == 'both' else t + pad
        padded = torch.zeros(b, total, c).cuda()
        if side == 'before' or side == 'both' :
            padded[:, pad:pad+t, :] = x
        elif side == 'after':
            padded[:, :t, :] = x
        return padded


    def fold_with_overlap(self, x, target, overlap) :

        ''' Fold the tensor with overlap for quick batched inference.
            Overlap will be used for crossfading in xfade_and_unfold()

        Args:
            x (tensor)    : Upsampled conditioning features.
                            shape=(1, timesteps, features)
            target (int)  : Target timesteps for each index of batch
            overlap (int) : Timesteps for both xfade and rnn warmup

        Return:
            (tensor) : shape=(num_folds, target + 2 * overlap, features)

        Details:
            x = [[h1, h2, ... hn]]

            Where each h is a vector of conditioning features

            Eg: target=2, overlap=1 with x.size(1)=10

            folded = [[h1, h2, h3, h4],
                      [h4, h5, h6, h7],
                      [h7, h8, h9, h10]]
        '''

        _, total_len, features = x.size()

        # Calculate variables needed
        num_folds = (total_len - overlap) // (target + overlap)
        extended_len = num_folds * (overlap + target) + overlap
        remaining = total_len - extended_len

        # Pad if some time steps poking out
        if remaining != 0 :
            num_folds += 1
            padding = target + 2 * overlap - remaining
            x = self.pad_tensor(x, padding, side='after')

        folded = torch.zeros(num_folds, target + 2 * overlap, features).cuda()

        # Get the values for the folded tensor
        for i in range(num_folds) :
            start = i * (target + overlap)
            end = start + target + 2 * overlap
            folded[i] = x[:, start:end, :]

        return folded


    def xfade_and_unfold(self, y, target, overlap) :

        ''' Applies a crossfade and unfolds into a 1d array.

        Args:
            y (ndarry)    : Batched sequences of audio samples
                            shape=(num_folds, target + 2 * overlap)
                            dtype=np.float64
            overlap (int) : Timesteps for both xfade and rnn warmup

        Return:
            (ndarry) : audio samples in a 1d array
                       shape=(total_len)
                       dtype=np.float64

        Details:
            y = [[seq1],
                 [seq2],
                 [seq3]]

            Apply a gain envelope at both ends of the sequences

            y = [[seq1_in, seq1_target, seq1_out],
                 [seq2_in, seq2_target, seq2_out],
                 [seq3_in, seq3_target, seq3_out]]

            Stagger and add up the groups of samples:

            [seq1_in, seq1_target, (seq1_out + seq2_in), seq2_target, ...]

        '''

        num_folds, length = y.shape
        target = length - 2 * overlap
        total_len = num_folds * (target + overlap) + overlap

        # Need some silence for the rnn warmup
        silence_len = overlap // 2
        fade_len = overlap - silence_len
        silence = np.zeros((silence_len), dtype=np.float64)

        # Equal power crossfade
        t = np.linspace(-1, 1, fade_len, dtype=np.float64)
        fade_in = np.sqrt(0.5 * (1 + t))
        fade_out = np.sqrt(0.5 * (1 - t))

        # Concat the silence to the fades
        fade_in = np.concatenate([silence, fade_in])
        fade_out = np.concatenate([fade_out, silence])

        # Apply the gain to the overlap samples
        y[:, :overlap] *= fade_in
        y[:, -overlap:] *= fade_out

        unfolded = np.zeros((total_len), dtype=np.float64)

        # Loop to add up all the samples
        for i in range(num_folds ) :
            start = i * (target + overlap)
            end = start + target + 2 * overlap
            unfolded[start:end] += y[i]

        return unfolded

    def generate(self, mels, target=11000, overlap=550, batched=True):

        self.eval()
        output = []

        rnn1 = self.get_gru_cell(self.rnn1)
        rnn2 = self.get_gru_cell(self.rnn2)

        with torch.no_grad():

            mels = torch.FloatTensor(mels).cuda().unsqueeze(0)
            mels = self.pad_tensor(mels.transpose(1, 2), pad=hp.pad_gen, side='both')
            mels, aux = self.upsample(mels.transpose(1, 2))

            if batched:
                target = mels.shape[1] // hp.batch_size_gen
                mels = self.fold_with_overlap(mels, target, overlap)
                aux = self.fold_with_overlap(aux, target, overlap)

            b_size, seq_len, _ = mels.size()

            h1 = torch.zeros(b_size, self.rnn_dims).cuda()
            h2 = torch.zeros(b_size, self.rnn_dims).cuda()
            x = torch.zeros(b_size, 1).cuda()

            d = self.aux_dims
            aux_split = [aux[:, :, d * i:d * (i + 1)] for i in range(4)]

            for i in range(seq_len):

                m_t = mels[:, i, :]

                a1_t, a2_t, a3_t, a4_t = \
                    (a[:, i, :] for a in aux_split)

                x = torch.cat([x, m_t, a1_t], dim=1)
                x = self.I(x)
                h1 = rnn1(x, h1)

                x = x + h1
                inp = torch.cat([x, a2_t], dim=1)
                h2 = rnn2(inp, h2)

                x = x + h2
                x = torch.cat([x, a3_t], dim=1)
                x = F.relu(self.fc1(x))

                x = torch.cat([x, a4_t], dim=1)
                x = F.relu(self.fc2(x))

                logits = self.fc3(x)
                posterior = F.softmax(logits, dim=1)
                distrib = torch.distributions.Categorical(posterior)
                sample = 2 * distrib.sample().float() / (self.n_classes - 1.) - 1.
                # sample = ((2.* torch.argmax(posterior,dim=1).float()) / (self.n_classes - 1.) - 1.)
                output.append(sample)
                x = sample.unsqueeze(-1)

        output = torch.stack(output).transpose(0, 1)
        output = output.cpu().numpy()
        output = output.astype(np.float64)

        if batched:
            output = self.xfade_and_unfold(output, target, overlap)
        else:
            output = output[0]

        self.train()
        return output

    @staticmethod
    def _round_up(num, divisor, bsize):
        return num + divisor - (num%divisor)



    def _batch_mels(self, mel_in):

        n_frames_full = mel_in.shape[1]
        n_frames_batch = n_frames_full // hp.batch_size_gen
        self.n_overlap = hp.hop_size*hp.pad_gen

        mel_in = torch.nn.functional.pad(mel_in, (0, 0, self.n_overlap, self.n_overlap, 0, 0))
        idxs=[range(k*n_frames_batch, (k+1)*n_frames_batch+self.n_overlap) for k in range(0, hp.batch_size_gen)]

        mel_batched = mel_in.squeeze(0)[idxs,:] #.permute(0, 2, 1)
        return mel_batched, self.n_overlap


    def glueback(self, wav):
        from itertools import product
        srch_frames = hp.hop_size #how far to search for matching phase

        wav = wav[(self.n_overlap-srch_frames):, :]

        left_wav = wav[:, 0]
        for i in range(wav.shape[1] - 1):

            right_wav = wav[:, i+1]

            dif=[]
            for k in range(srch_frames):
                dif.append(np.sum((left_wav[(-k-10-1):-k-1]-right_wav[k:(k+10)])**2))
            kmin = np.argmin(dif)

            left_wav = np.hstack([left_wav[:-kmin-1], right_wav[kmin+10:]])
        return left_wav

    def _unbatch_sound(self, x, pad_length):

        y = self.glueback(x)
        #y = x.transpose().flatten()
        return y, x

    def batch_generate(self, mels) :
        """mel should be of shape [batch_size x 80 x mel_length]
        """
        self.eval()
        output = []
        rnn1 = self.get_gru_cell(self.rnn1)
        rnn2 = self.get_gru_cell(self.rnn2)

        with torch.no_grad():

            mels = torch.FloatTensor(mels).cuda()
            mels = mels.unsqueeze(0)
            mels, aux = self.upsample(mels)

            mels, pad_length = self._batch_mels(mels)
            aux, _ = self._batch_mels(aux)
            b_size = mels.shape[0]

            x = torch.zeros(b_size, 1).cuda()
            h1 = torch.zeros(b_size, self.rnn_dims).cuda()
            h2 = torch.zeros(b_size, self.rnn_dims).cuda()

            aux_idx = [self.aux_dims * i for i in range(5)]
            a1 = aux[:, :, aux_idx[0]:aux_idx[1]]
            a2 = aux[:, :, aux_idx[1]:aux_idx[2]]
            a3 = aux[:, :, aux_idx[2]:aux_idx[3]]
            a4 = aux[:, :, aux_idx[3]:aux_idx[4]]
            
            seq_len = mels.size(1)
            
            for i in tqdm(range(seq_len)) :

                m_t = mels[:, i, :]
                a1_t = a1[:, i, :]
                a2_t = a2[:, i, :]
                a3_t = a3[:, i, :]
                a4_t = a4[:, i, :]
                
                x = torch.cat([x, m_t, a1_t], dim=1)
                x = self.I(x)
                h1 = rnn1(x, h1)
                
                x = x + h1
                inp = torch.cat([x, a2_t], dim=1)
                h2 = rnn2(inp, h2)
                
                x = x + h2
                x = torch.cat([x, a3_t], dim=1)
                x = F.relu(self.fc1(x))
                
                x = torch.cat([x, a4_t], dim=1)
                x = F.relu(self.fc2(x))
                x = self.fc3(x)

                posterior = F.softmax(x, dim=1).view(b_size, -1)
                distrib = torch.distributions.Categorical(posterior)
                sample = 2 * distrib.sample().float() / (self.n_classes - 1.) - 1.
                output.append(sample.view(-1))
                x = sample.view(b_size,1)
        output = torch.stack(output).cpu().numpy()
        output, output1 = self._unbatch_sound(output, pad_length)
        self.train()
        return output, output1
    
    def get_gru_cell(self, gru) :
        gru_cell = nn.GRUCell(gru.input_size, gru.hidden_size)
        gru_cell.weight_hh.data = gru.weight_hh_l0.data
        gru_cell.weight_ih.data = gru.weight_ih_l0.data
        gru_cell.bias_hh.data = gru.bias_hh_l0.data
        gru_cell.bias_ih.data = gru.bias_ih_l0.data
        return gru_cell


def build_model():
    """build model with hparams settings

    """
    model = Model(hp.rnn_dims, hp.fc_dims, hp.bits,
        hp.upsample_factors, hp.num_mels,
        hp.compute_dims, hp.res_out_dims, hp.res_blocks)

    return model 

def no_test_build_model():
    model = Model(hp.rnn_dims, hp.fc_dims, hp.bits,
        hp.pad, hp.upsample_factors, hp.num_mels,
        hp.compute_dims, hp.res_out_dims, hp.res_blocks).cuda()
    print(vars(model))


def test_batch_generate():
    model = Model(hp.rnn_dims, hp.fc_dims, hp.bits,
        hp.pad, hp.upsample_factors, hp.num_mels,
        hp.compute_dims, hp.res_out_dims, hp.res_blocks).cuda()
    print(vars(model))
    batch_mel = torch.rand(3, 80, 100)
    output = model.batch_generate(batch_mel)
    print(output.shape)
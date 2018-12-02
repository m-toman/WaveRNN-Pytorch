"""Synthesis script for WaveRNN vocoder

usage: synthesize.py [options] <mel_input.npy>

options:
    --checkpoint-dir=<dir>       Directory where model checkpoint is saved [default: checkpoints].
    --output-dir=<dir>           Output Directory [default: model_outputs]
    --hparams=<params>           Hyper parameters [default: ].
    --preset=<json>              Path of preset parameters (json).
    --checkpoint=<path>          Restore model from checkpoint path if given.
    --no-cuda                    Don't run on GPU
    -h, --help                   Show this help message and exit
"""
import os
import librosa
import glob

from docopt import docopt
from model import *
from hparams import hparams
from utils import num_params_count

if __name__ == "__main__":
    args = docopt(__doc__)
    print("Command line args:\n", args)
    checkpoint_dir = args["--checkpoint-dir"]
    output_path = args["--output-dir"]
    checkpoint_path = args["--checkpoint"]
    preset = args["--preset"]
    no_cuda = args["--no-cuda"]

    device = torch.device("cpu" if no_cuda else "cuda")
    print("using device:{}".format(device))

    # Load preset if specified
    if preset is not None:
        with open(preset) as f:
            hparams.parse_json(f.read())
    # Override hyper parameters
    hparams.parse(args["--hparams"])

    mel_file_name = args['<mel_input.npy>']
    mel = np.load(mel_file_name)

    flist = glob.glob(f'{checkpoint_dir}/checkpoint_*.pth')
    latest_checkpoint = max(flist, key=os.path.getctime)
    print('Loading: %s'%latest_checkpoint)
    # build model, create optimizer
    model = build_model().to(device)
    checkpoint = torch.load(latest_checkpoint)
    model.load_state_dict(checkpoint["state_dict"])

    print("I: %.3f million"%(num_params_count(model.I)))
    print("Upsample: %.3f million"%(num_params_count(model.upsample)))
    print("rnn1: %.3f million"%(num_params_count(model.rnn1)))
    print("rnn2: %.3f million"%(num_params_count(model.rnn2)))
    print("fc1: %.3f million"%(num_params_count(model.fc1)))
    print("fc2: %.3f million"%(num_params_count(model.fc2)))
    print("fc3: %.3f million"%(num_params_count(model.fc3)))

    #mel = np.pad(mel,(24000,0),'constant')
    n_mels = mel.shape[1]
    n_mels = hparams.batch_size_gen * (n_mels // hparams.batch_size_gen)
    mel = mel[:, 0:n_mels]
    mel0 = mel.copy()
    output0 = model.generate(mel0)
    librosa.output.write_wav(os.path.join(output_path, os.path.basename(mel_file_name)+'_orig.wav'), output0, hparams.sample_rate)

    mel = mel.reshape([mel.shape[0], hparams.batch_size_gen, -1]).swapaxes(0,1)
    output = model.batch_generate(mel)
    bootstrap_len = hp.hop_size * hp.resnet_pad
    output=output[:,bootstrap_len:].reshape(-1)
    librosa.output.write_wav(os.path.join(output_path, os.path.basename(mel_file_name)+'.wav'), output, hparams.sample_rate)
    print('done')

import pylab as pl

n = output.shape[1]
i=26
pl.clf()
pl.plot(output0[(i*n):(n*(i+2))])
pl.plot(np.hstack([output[i,:],output[i+1,:]]))

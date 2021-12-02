import sys, os, logging, torch, time, configargparse, socket

import pandas as pd
import numpy as np

from pathlib import Path

#appends current directory to sys path allowing data imports.
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

#from data import load_n2c2_2006_train_dev_split, load_n2c2_2006
from bdc.document_bert import BertForDocumentClassification

log = logging.getLogger()

def _initialize_arguments(p: configargparse.ArgParser):
    p.add('--model_storage_directory', help='The directory caching all model runs')
    p.add('--bert_model_path', help='Model path to BERT')
    p.add('--labels', help='Numbers of labels to predict over', type=str)
    p.add('--architecture', help='Training architecture', type=str)
    p.add('--freeze_bert', help='Whether to freeze bert', type=bool)

    p.add('--batch_size', help='Batch size for training multi-label document classifier', type=int)
    p.add('--bert_batch_size', help='Batch size for feeding 510 token subsets of documents through BERT', type=int)
    p.add('--epochs', help='Epochs to train', type=int)
    #Optimizer arguments
    p.add('--learning_rate', help='Optimizer step size', type=float)
    p.add('--weight_decay', help='Adam regularization', type=float)

    p.add('--evaluation_interval', help='Evaluate model on test set every evaluation_interval epochs', type=int)
    p.add('--checkpoint_interval', help='Save a model checkpoint to disk every checkpoint_interval epochs', type=int)

    #Non-config arguments
    p.add('--cuda', action='store_true', help='Utilize GPU for training or prediction')
    p.add('--device')
    p.add('--timestamp', help='Run specific signature')
    p.add('--model_directory', help='The directory storing this model run, a sub-directory of model_storage_directory')
    p.add('--use_tensorboard', help='Use tensorboard logging', type=bool)
    args = p.parse_args()

    args.labels = [x for x in args.labels.split(', ')]

    #Set run specific envirorment configurations
    args.timestamp = time.strftime("run_%Y_%m_%d_%H_%M_%S") + "_{machine}".format(machine=socket.gethostname())
    args.model_directory = os.path.join(args.model_storage_directory, args.timestamp) #directory
    os.makedirs(args.model_directory, exist_ok=True)

    #Handle logging configurations
    log.handlers.clear()
    formatter = logging.Formatter('%(message)s')
    fh = logging.FileHandler(os.path.join(args.model_directory, "log.txt"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    log.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    log.setLevel(logging.INFO)
    log.addHandler(ch)
    log.info(p.format_values())


    #Set global GPU state
    if torch.cuda.is_available() and args.cuda:
        if torch.cuda.device_count() > 1:
            log.info("Using %i CUDA devices" % torch.cuda.device_count() )
        else:
            log.info("Using CUDA device:{0}".format(torch.cuda.current_device()))
        args.device = 'cuda'
    else:
        log.info("Not using CUDA :(")
        args.dev = 'cpu'

    return args


if __name__ == "__main__":
    n_folds = 9
    fold = 0
    variant = 'multi-%df-k%d' % (n_folds, fold)
    p = configargparse.ArgParser(default_config_files=["./config-%s.ini" % variant])
    args = _initialize_arguments(p)

    torch.cuda.empty_cache()

    train = pd.read_pickle('../input/%dfold/train-f%d-8.26.21-0.9-10.20.21.pickle' % (n_folds, fold))
    valid = pd.read_pickle('../input/%dfold/valid-f%d-8.26.21-0.9-10.20.21.pickle' % (n_folds, fold))

    # 10 labels for [ONE, ..., TEN] classification
    train_documents = train.Abstract.tolist()
    train_labels = []
    for i in train.Category.tolist():
        tmp = [0] * 9
        if i == 1:
            tmp[0] = 1
        else:
            tmp[i-2] = 1
        train_labels.append(tmp)

    dev_documents = valid.Abstract.tolist()
    dev_labels = []
    for i in valid.Category.tolist():
        tmp = [0] * 9
        if i == 1:
            tmp[0] = 1
        else:
            tmp[i-2] = 1
        dev_labels.append(tmp)

    model = BertForDocumentClassification(args=args)
    epoch_probs = model.fit((train_documents, train_labels), (dev_documents, dev_labels))

    Path('../output/%s/model' % variant).mkdir(parents=True, exist_ok=True)
    Path('../output/%s/tokenizer' % variant).mkdir(parents=True, exist_ok=True)

    if os.path.isfile('../output/%s/epoch_probs.npy' % variant):
        os.remove('../output/%s/epoch_probs.npy' % variant)

    with open('../output/%s/epoch_probs.npy' % variant, 'wb') as f:
        np.save(f, np.array(epoch_probs))

    model.bert_doc_classification.save_pretrained('../output/%s/model/' % variant)
    model.bert_tokenizer.save_pretrained('../output/%s/tokenizer/' % variant)


from main import main
from BertOrigin import args

if __name__ == "__main__":

    data_dir = "/search/hadoop02/suanfa/songyingxin/data/SST-2"
    output_dir = ".sst_output"
    cache_dir = ".sst_cache"
    log_dir = ".sst_log"

    bert_vocab_file = "/search/hadoop02/suanfa/songyingxin/pytorch_Bert/bert-base-uncased-vocab.txt"
    bert_model_dir = "/search/hadoop02/suanfa/songyingxin/pytorch_Bert/bert-base-uncased"
    
    from Processors.SST2Processor import Sst2Processor
    main(args.get_args(data_dir, output_dir, cache_dir, bert_vocab_file, bert_model_dir, log_dir), Sst2Processor)

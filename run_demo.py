# coding=utf-8

from main import main


if __name__ == "__main__":

    model_name = "BertRCNN"
    data_dir = "/search/hadoop02/suanfa/songyingxin/Github/chiqianqian"
    output_dir = ".demo_output/"
    cache_dir = ".demo_cache"
    log_dir = ".demo_log/"

    model_times = "model_1/"   # 第几次保存的模型，主要是用来获取最佳结果

    bert_vocab_file = "/search/hadoop02/suanfa/songyingxin/pytorch_Bert/bert-base-uncased-vocab.txt"
    bert_model_dir = "/search/hadoop02/suanfa/songyingxin/pytorch_Bert/bert-base-uncased"

    from Processors.DemoProcessor import DemoProcessor

    if model_name == "BertOrigin":
        from BertOrigin import args
        main(
            args.get_args(data_dir, output_dir, cache_dir,
                          bert_vocab_file, bert_model_dir, log_dir), 
                          model_times, DemoProcessor)

    elif model_name == "BertCNN":
        from BertCNN import args
        main(args.get_args(data_dir, output_dir, cache_dir,
                           bert_vocab_file, bert_model_dir, log_dir),
             model_times, DemoProcessor)
    elif model_name == "BertATT":
        from BertATT import args
        main(args.get_args(data_dir, output_dir, cache_dir,
                           bert_vocab_file, bert_model_dir, log_dir),
             model_times, DemoProcessor)
    elif model_name == "BertRCNN":
        from BertRCNN import args
        main(args.get_args(data_dir, output_dir, cache_dir,
                           bert_vocab_file, bert_model_dir, log_dir),
             model_times, DemoProcessor)

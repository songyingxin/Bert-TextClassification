# coding=utf-8
from main_competition import main


if __name__ == "__main__":

    """ Kaggel 比赛 """
    model_name = "BertOrigin"
    data_dir = "/search/hadoop02/suanfa/songyingxin/Github/Competition"
    output_dir = ".competition_output/"
    cache_dir = ".competition_cache"
    log_dir = ".competition_log/"

    model_times = "model_1/"   # 第几次保存的模型，主要是用来获取最佳结果

    bert_vocab_file = "/search/hadoop02/suanfa/songyingxin/pytorch_Bert/bert-base-uncased-vocab.txt"
    bert_model_dir = "/search/hadoop02/suanfa/songyingxin/pytorch_Bert/bert-base-uncased"

    from Processors.CompetitionProcessor import CompetitionProcessor

    if model_name == "BertOrigin":
        from BertOrigin import args

    elif model_name == "BertCNN":
        from BertCNN import args


    elif model_name == "BertATT":
        from BertATT import args

    elif model_name == "BertRCNN":
        from BertRCNN import args
        
    main(args.get_args(data_dir, output_dir, cache_dir,
                           bert_vocab_file, bert_model_dir, log_dir),
             model_times, CompetitionProcessor)

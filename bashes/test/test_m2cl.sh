python3 -m domainbed.scripts.test\
       --data_dir=./domainbed/data/\
       --algorithm M2CL\
       --dataset PACS\
       --batch_size 8\
       --test_env 3\
       --steps 15000\
       --output_dir M2CL\
       --pretrain ./M2CL/model_best_env3_out_acc.pkl\
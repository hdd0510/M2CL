python3 -m domainbed.scripts.test\
       --data_dir=./domainbed/data/\
       --algorithm MMD\
       --dataset PACS\
       --batch_size 8\
       --test_env 3\
       --steps 10000\
       --output_dir MMD\
       --pretrain /mnt/disk1/nmduong/hust/m2cl/MMD/model_best_env3_out_acc.pkl\
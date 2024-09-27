python3 -m domainbed.scripts.test\
       --data_dir=./domainbed/data/\
       --algorithm ERM\
       --dataset PACS\
       --batch_size 3\
       --test_env 3\
       --steps 10000\
       --output_dir ERM\
       --pretrain /mnt/disk1/nmduong/hust/m2cl/ERM/model_best_env3_out_acc.pkl\
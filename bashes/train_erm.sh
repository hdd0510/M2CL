python3 -m domainbed.scripts.train\
       --data_dir=./domainbed/data/\
       --algorithm ERM\
       --dataset PACS\
       --batch_size 8\
       --test_env 3 \
       --steps 10000\
       --output_dir ERM\
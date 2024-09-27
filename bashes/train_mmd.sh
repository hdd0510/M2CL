python3 -m domainbed.scripts.train\
       --data_dir=./domainbed/data/\
       --algorithm MMD\
       --dataset PACS\
       --batch_size 8\
       --test_env 2 \
       --temp 1.0\
       --steps 10000\
       --output_dir MMD\
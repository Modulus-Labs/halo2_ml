#!/bin/bash
for i in lenet_deep_0 lenet_deep_1 lenet_deep_2 lenet_long_0 lenet_long_1 lenet_long_2 lenet_lookalike_0 lenet_lookalike_1 lenet_lookalike_2 lenet_lookalike_3 lenet_lookalike_4 lenet_tall_0 lenet_tall_1 lenet_tall_2 lenet_deep_0_heap lenet_deep_1_heap lenet_deep_2_heap lenet_long_0_heap lenet_long_1_heap lenet_long_2_heap lenet_lookalike_0_heap lenet_lookalike_1_heap lenet_lookalike_2_heap lenet_lookalike_3_heap lenet_lookalike_4_heap lenet_tall_0_heap lenet_tall_1_heap lenet_tall_2_heap
do
    echo "Running tests for network $i" >> /home/ubuntu/halo2_benches/benchmarking_networks.log
    echo "---" >> /home/ubuntu/halo2_benches/benchmarking_networks.log
    # sed -i "s/const SIZE: usize = $prev_i;/const SIZE: usize = $i;/" /home/ubuntu/halo2deeplearningfork/examples/mlp_4d.rs
    # sed -i "s/$prev_i | $prev_i | $prev_i/$i | $i | $i/" /home/ubuntu/halo2deeplearningfork/examples/mlp_4d.rs
    # cargo build --example mlp_4d --release
    /home/ubuntu/halo2_benches/$i >> /home/ubuntu/halo2_benches/benchmarking_networks.log
    echo "---" >> /home/ubuntu/halo2_benches/benchmarking_networks.log
done
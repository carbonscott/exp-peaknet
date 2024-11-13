#!/bin/bash

# Source node information
source node_info.txt

# Get IP addresses
HEAD_IP=$(ssh $HEAD_NODE "hostname -i" | awk '{print $1}')
echo "Head node IP: $HEAD_IP"

set -x

###################
# Environment setup
###################
# Setup
SETUP_CMD=$(cat << EOF
echo 'Loading ana-py3...' &&
source /sdf/group/lcls/ds/ana/sw/conda1/manage/bin/psconda.sh &&
echo 'Loading ml.cong...' &&
conda activate /sdf/data/lcls/ds/prj/prjcwang31/results/conda/ana-4.0.58-py3-ml &&
cd /sdf/data/lcls/ds/prj/prjcwang31/results/proj-peaknet
EOF
)

# Define Ray start command
RAY_HEAD_CMD=$(cat << EOF
ray start --head \
    --node-ip-address=127.0.0.1 \
    --port=6379 \
    --num-cpus=10 \
    --disable-usage-stats
EOF
)
RAY_WORKER_CMD=$(cat << EOF
ray start \
    --address='${HEAD_IP}:6379' \
    --num-cpus=2
EOF
)

# Define MPI commands
JOB1_CMD=$(cat << 'EOF'
mpirun -n 10 psana-ray-producer \
    --exp mfxl1038923 \
    --run 278 \
    --detector_name epix10k2M \
    --queue_size 1000 \
    --ray_namespace my \
    --queue_name input \
    --uses_bad_pixel_mask True \
    --manual_mask_path manual_mask.npy
EOF
)

JOB2_CMD=$(cat << 'EOF'
mpirun -n 10 peaknet-pipeline-mpi \
    --batch_size 4 \
    --num_workers 1 \
    --config_path configs/peaknet-673m.yaml \
    --weights_path weights/peaknet-673m.bin \
    --dtype bfloat16 \
    --accumulation_steps 10 \
    --output_queue_name peak_queue \
    --output_queue_size 100 \
    --input_queue_name input \
    --ray_namespace my
EOF
)

JOB3_CMD=$(cat << 'EOF'
mpirun -n 10 peaknet-pipeline-write-to-cxi \
    --queue_name peak_queue \
    --ray_namespace my \
    --output_dir cxi_output \
    --basename peaknet_output \
    --save_every 2 \
    --min_num_peak 15 \
    --geom_file geom/epix10ka2M-r0138-stream.geom.stream
EOF
)

###################
# Let's go BRRR!!!
###################
# Start Ray on head node
echo "Starting Ray on $HEAD_NODE..."
ssh $HEAD_NODE "$SETUP_CMD && $RAY_HEAD_CMD" &

# Wait for 20 seconds for Ray head node to initialize
sleep 20

# Connect worker nodes to Ray head node
for node in $WORKER_NODES; do
    echo "Connecting $node to Ray head node..."
    ssh $node "$SETUP_CMD && $RAY_WORKER_CMD" &
done

# Wait for workers to connect
sleep 10

# Launch Job1 on head node
echo "Starting Job1 on $HEAD_NODE..."
ssh $HEAD_NODE "$SETUP_CMD && $JOB1_CMD &> inference_log/psana-ray-producer.nersc-atto-2bifpn512-1.0.log" &
JOB1_PID=$!

# Wait for 30 seconds for Job1 to initialize
sleep 30

# Launch Job2 across both nodes
# Initialize an array for all Job2 PIDs (head and workers)
declare -a JOB2_PIDS=()
echo "Starting Job2 across nodes..."
for node in $HEAD_NODE $WORKER_NODES; do
    ssh $node "$SETUP_CMD && $JOB2_CMD &> 'inference_log/peaknet-pipeline-mpi.nersc-atto-2bifpn512-1.0.${node}.log'" &
    JOB2_PIDS+=($!)
done

# Wait for 60 seconds for Job2 to initialize
sleep 60

# Launch Job3 on head node
echo "Starting Job3 on $HEAD_NODE..."
ssh $HEAD_NODE "$SETUP_CMD && $JOB3_CMD &> inference_log/peaknet-pipeline-write-to-cxi.nersc-atto-2bifpn512-1.0.log" &
JOB3_PID=$!

# Wait for all jobs to complete
wait $JOB1_PID ${JOB2_PIDS[@]} $JOB3_PID

echo "All jobs completed"

#!/bin/bash
#
# Save SLURM and training environment variables for later SSH sessions
#

if [ -z "$SLURM_JOB_ID" ]; then
    echo "Error: Not in a SLURM job environment"
    exit 1
fi

# Create session directory
mkdir -p .slurm_sessions

# Save environment variables
ENV_FILE=".slurm_sessions/job_${SLURM_JOB_ID}_env.sh"
INFO_FILE=".slurm_sessions/job_${SLURM_JOB_ID}_info.txt"

echo "# SLURM Environment Variables for Job $SLURM_JOB_ID" > $ENV_FILE
echo "# Generated: $(date)" >> $ENV_FILE
echo "" >> $ENV_FILE

# Save SLURM variables with proper quoting for special characters
env | grep SLURM | sed 's/^/export /' | sed 's/=\(.*\s.*\)/="\1"/' >> $ENV_FILE

# Save training-related variables
echo "" >> $ENV_FILE
echo "# Training Environment Variables" >> $ENV_FILE
env | grep -E "(MASTER_ADDR|MASTER_PORT|NCCL_|MIOPEN_|ROCR_|CUDA_|HIP_|OMP_|CRAY_)" | sed 's/^/export /' | sed 's/=\(.*\s.*\)/="\1"/' >> $ENV_FILE

# Create info file
cat > $INFO_FILE << EOF
# SLURM Session Environment for Job $SLURM_JOB_ID
# Created: $(date)
# Nodes: $SLURM_JOB_NODELIST
# Project: $(pwd)

# To use in new SSH session:
# source .slurm_sessions/job_${SLURM_JOB_ID}_env.sh

# To SSH to a compute node:
# ssh \$(echo $SLURM_JOB_NODELIST | cut -d'[' -f1)02293  # or any node from the list

# Available nodes:
EOF

# List individual nodes
echo "$SLURM_JOB_NODELIST" | tr ',' '\n' | sed 's/frontier\[//; s/\]//; s/-.*$//' >> $INFO_FILE

echo ""
echo "✅ Environment saved!"
echo "📁 Files: $ENV_FILE, $INFO_FILE"
echo "📋 Variables saved: $(wc -l < $ENV_FILE)"
echo ""
echo "🔧 To use in new SSH session:"
echo "   source .slurm_sessions/job_${SLURM_JOB_ID}_env.sh"
echo ""
echo "🖥️  Available compute nodes:"
echo "   $SLURM_JOB_NODELIST"
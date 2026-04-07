#!/usr/bin/env bash
# submit all fields for kappa analysis, each field is submitted as a separate job.
set -euo pipefail

fields=(
  XMM
  VVDS
  GAMA09H
  GAMA15H
  WIDE12H
  HECTOMAP
)

for field in "${fields[@]}"; do
  # mock kappa analysis
  # qsub -N "kappa_${field}" -v FIELD="$field" run_kappa_mock.sh

  # real kappa analysis
  # qsub -N "kappa_${field}" -v FIELD="$field" run_kappa.sh

  # real shear analysis
  # qsub -N "shear_${field}" -v FIELD="$field" run_shear.sh
  qsub -N "shear_${field}" -v FIELD="$field" run_shear_mock.sh
done

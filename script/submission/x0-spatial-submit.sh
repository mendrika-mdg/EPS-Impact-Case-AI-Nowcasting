#!/bin/bash

# Long lead time (12x12)
Dakar_context_long_lt_lat_min=8.69
Dakar_context_long_lt_lat_max=20.69  # 8.69 + 12

Dakar_context_long_lt_lon_min=-23.45
Dakar_context_long_lt_lon_max=-11.45  # -23.45 + 12


# Submit long lead time job
sbatch /localhome/home/mmmhr/EPS-Impact-Case-AI-Nowcasting/script/submission/x0-spatial.sh \
  $Dakar_context_long_lt_lat_min \
  $Dakar_context_long_lt_lat_max \
  $Dakar_context_long_lt_lon_min \
  $Dakar_context_long_lt_lon_max \

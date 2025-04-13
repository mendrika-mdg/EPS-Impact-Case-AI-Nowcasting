#!/bin/bash

# Context domain for shorter lead times (12x12)
Senegal_context_short_lt_lat_min=9.0
Senegal_context_short_lt_lat_max=21.0  # 9.0 + 12

Senegal_context_short_lt_lon_min=-21.0
Senegal_context_short_lt_lon_max=-9.0  # -21.0 + 12

# Context domain for longer lead times (18x18)
Senegal_context_long_lt_lat_min=6.0
Senegal_context_long_lt_lat_max=24.0  # 6.0 + 18

Senegal_context_long_lt_lon_min=-24.0
Senegal_context_long_lt_lon_max=-6.0  # -24.0 + 18

# Submit short lead time job
sbatch /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/script/submission/nb-x0.sh \
  $Senegal_context_short_lt_lat_min \
  $Senegal_context_short_lt_lat_max \
  $Senegal_context_short_lt_lon_min \
  $Senegal_context_short_lt_lon_max \
  senegal \
  short

# Submit long lead time job
sbatch /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/script/submission/nb-x0.sh \
  $Senegal_context_long_lt_lat_min \
  $Senegal_context_long_lt_lat_max \
  $Senegal_context_long_lt_lon_min \
  $Senegal_context_long_lt_lon_max \
  senegal \
  long

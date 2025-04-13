#!/bin/bash

# Short lead time (6x6)
Dakar_context_short_lt_lat_min=11.69
Dakar_context_short_lt_lat_max=17.69  # 11.69 + 6

Dakar_context_short_lt_lon_min=-20.45
Dakar_context_short_lt_lon_max=-14.45  # -20.45 + 6

# Long lead time (12x12)
Dakar_context_long_lt_lat_min=8.69
Dakar_context_long_lt_lat_max=20.69  # 8.69 + 12

Dakar_context_long_lt_lon_min=-23.45
Dakar_context_long_lt_lon_max=-11.45  # -23.45 + 12

# Submit short lead time job
sbatch /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/script/submission/nb-x0.sh \
  $Dakar_context_short_lt_lat_min \
  $Dakar_context_short_lt_lat_max \
  $Dakar_context_short_lt_lon_min \
  $Dakar_context_short_lt_lon_max \
  dakar \
  short

# Submit long lead time job
sbatch /home/users/mendrika/EPS-Impact-Case-AI-Nowcasting/script/submission/nb-x0.sh \
  $Dakar_context_long_lt_lat_min \
  $Dakar_context_long_lt_lat_max \
  $Dakar_context_long_lt_lon_min \
  $Dakar_context_long_lt_lon_max \
  dakar \
  long

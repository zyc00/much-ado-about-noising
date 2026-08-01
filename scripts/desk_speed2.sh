#!/bin/bash
U="http://137.110.112.12:8931/6a411ba785dccf6955258d2d198e963f/robomimic/square/mh/low_dim_abs.hdf5"
S=$(timeout 60 env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    curl -s -o /dev/null -w "%{speed_download} %{size_download}" -r 0-1048575 -m 55 "$U" 2>/dev/null)
echo "DESK_1MB_direct $S"
S=$(timeout 90 env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    curl -s -o /dev/null -w "%{speed_download} %{size_download}" -m 85 "$U" 2>/dev/null)
echo "DESK_full_direct $S"

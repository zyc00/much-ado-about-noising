#!/bin/bash
U="http://137.110.112.12:8931/6a411ba785dccf6955258d2d198e963f/robomimic/square/mh/low_dim_abs.hdf5"
S=$(timeout 45 env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    curl -s -o /dev/null -w "%{speed_download}" -r 0-20971519 "$U" 2>/dev/null)
echo "DESK_direct ${S%.*} B/s"
S=$(timeout 45 curl -s -x http://172.17.0.12:2080 -o /dev/null -w "%{speed_download}" -r 0-20971519 "$U" 2>/dev/null)
echo "DESK_proxy ${S%.*} B/s"

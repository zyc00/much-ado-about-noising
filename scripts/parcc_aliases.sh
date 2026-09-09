# Source on a PARCC login node: source ~/.config/parcc/aliases.sh
# Compatible with bash and zsh. All commands are read-only.

parcc_gpu() {
    local parcc_nodes
    parcc_nodes=$(scontrol show nodes -o) || return
    printf '%s\n' "$parcc_nodes" | awk '
    function gpu(t, a,n,i) {
        n=split(t,a,",")
        for(i=1;i<=n;i++) if(a[i] ~ /^gres\/gpu=/) {
            sub(/^gres\/gpu=/,"",a[i]); return a[i]+0
        }
        return 0
    }
    {
        part=state=cfg=alloc=""
        for(i=1;i<=NF;i++) {
            v=$i; sub(/^[^=]*=/,"",v)
            if($i ~ /^Partitions=/) part=v
            if($i ~ /^State=/) state=v
            if($i ~ /^CfgTRES=/) cfg=v
            if($i ~ /^AllocTRES=/) alloc=v
        }
        total=gpu(cfg); used=gpu(alloc)
        if(!total || !part) next
        free=total-used; if(free<0) free=0
        limited=(state ~ /DRAIN|DOWN|FAIL|MAINT|RESERVED|REBOOT|POWER|NO_RESPOND|UNKNOWN|INVALID/)
        n=split(part,parts,",")
        for(j=1;j<=n;j++) {
            p=parts[j]; totals[p]+=total; useds[p]+=used
            if(limited) blocked[p]+=free; else candidates[p]+=free
        }
    }
    END {
        printf "%-18s %7s %7s %11s %13s\n", "PARTITION","TOTAL","USED","FREE_CAND","FREE_LIMITED"
        for(p in totals)
            printf "%-18s %7d %7d %11d %13d\n",p,totals[p],useds[p],candidates[p],blocked[p]
        print "FREE_CAND: unallocated on nodes without the restrictions checked above; scheduling is still required."
        print "FREE_LIMITED: unallocated, but reserved/draining/down/rebooting/etc. Check gnodes and gresv."
        print "MIG partitions count GPU slices, not whole cards. Shared partitions must not be added together."
    }'
}

parcc_jobs() {
    local parcc_queue
    parcc_queue=$(squeue -h "$@" -O 'JobID:0|,UserName:0|,Partition:0|,StateCompact:0|,TimeUsed:0|,TimeLimit:0|,tres-alloc:0|,ReasonList:0|,Name:0') || return
    printf '%s\n' "$parcc_queue" | awk -F '|' '
    BEGIN {
        printf "%-20s %-12s %-16s %-3s %8s %9s %-12s %-12s %-25s %s\n", "JOBID","USER","PARTITION","ST","GPU_USED","GPU_SPEC","ELAPSED","LIMIT","NODE/REASON","NAME"
    }
    NF>=9 {
        gpu=0; n=split($7,tres,",")
        for(i=1;i<=n;i++) if(tres[i] ~ /^gres\/gpu=/) {
            sub(/^gres\/gpu=/,"",tres[i]); gpu=tres[i]+0
        }
        allocated=($4 ~ /^(R|CG|S|ST|RS|SI|SO|CF)$/) ? gpu : 0
        total+=allocated; jobs++
        printf "%-20s %-12s %-16s %-3s %8d %9d %-12s %-12s %-25s %s\n",$1,$2,$3,$4,allocated,gpu,$5,$6,$8,$9
    }
    END {
        if(!jobs) print "No matching jobs."
        printf "Allocated GPUs in listed jobs: %d (MIG slices count as GPUs).\n",total
        print "GPU_SPEC: allocated GPUs for started jobs; requested GPUs per task for pending jobs/arrays."
    }'
}

alias gfree='parcc_gpu'
alias gnodes='sinfo -N -p dgx-b200,b200-mig45,b200-mig90 -O NodeList:16,Partition:16,StateLong:35,Gres:30,GresUsed:60'
alias gresv='scontrol show reservation'
alias mj='parcc_jobs -u "$USER"'
alias mrun='parcc_jobs -u "$USER" -t RUNNING,COMPLETING'
alias mpend='parcc_jobs -u "$USER" -t PENDING'
alias mstart='squeue --start -u "$USER" -o "%.18i %.16P %.2t %.19S %.12l %R"'
alias lj='parcc_jobs -A "${PARCC_ACCOUNT:-jiayuanm-mao-lab}"'
alias jinfo='scontrol show job'
alias mhist='sacct -X -u "$USER" --starttime=now-1days --format=JobID,JobName%30,Partition,State,Elapsed,AllocTRES%65'

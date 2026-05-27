#!/bin/bash
set -e
S=$1
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin:$PATH
cd ~/v5_pathD; source ~/v5_venv/bin/activate
LOG=A_nograph_S${S}.log
echo "=== [A no-graph S$S] start $(date) ===" > $LOG

echo "=== Stage 1 (no-graph: no edges + no gate) seed $S ===" >> $LOG
python -u train_stage1_ecoli.py --mode full --device cuda --seed $S --no-srna-edges --no-srna-gate --embed-dim 128 --n-layers 4 --epochs 50 --batch-size 8 \
  --ckpt-dir checkpoints_nograph_stage1_S${S} >> $LOG 2>&1

echo "=== Stage 2 v2.1 (inherits empty edges) seed $S ===" >> $LOG
python -u train_stage2_v2.py --stage1-ckpt checkpoints_nograph_stage1_S${S}/best.pt \
  --seed $S --ckpt-dir checkpoints_nograph_stage2_S${S} >> $LOG 2>&1

echo "=== [fix] patch stage2 ckpt mrna_names/srna_names ===" >> $LOG
python -c "
import torch
p='checkpoints_nograph_stage2_S${S}/best.pt'
c=torch.load(p,map_location='cpu',weights_only=False)
vn=list(c['vocab_names']); nm=c['n_mrna']; nse=c['n_srna']
assert len(vn)==nm+nse,(len(vn),nm,nse)
c['mrna_names']=vn[:nm]; c['srna_names']=vn[nm:nm+nse]
torch.save(c,p); print('patched',len(c['mrna_names']),len(c['srna_names']))
" >> $LOG 2>&1

echo "=== Stage 3 conditioned seed $S (epochs 200) ===" >> $LOG
python -u train_stage3_v2_1_conditioned.py --stage2-ckpt checkpoints_nograph_stage2_S${S}/best.pt \
  --seed $S --epochs 200 --device cuda --ckpt-dir checkpoints_nograph_stage3_S${S} >> $LOG 2>&1

echo "=== PATHWAY + log2FC eval seed $S ===" >> $LOG
python eval_pathway_no_context_and_log2fc.py --ckpt checkpoints_nograph_stage3_S${S}/best.pt --device cuda >> $LOG 2>&1
python eval_yb1_mrna_n7.py --ckpt checkpoints_nograph_stage3_S${S}/best.pt >> $LOG 2>&1
python eval_yb1_holdout_n7.py --ckpt checkpoints_nograph_stage3_S${S}/best.pt >> $LOG 2>&1
echo "DONE_NOGRAPH_S${S} $(date)" >> $LOG

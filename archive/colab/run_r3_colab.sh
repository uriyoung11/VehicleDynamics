#!/bin/bash
# R3(신경망 타이어) 1초 손실 학습 15개를 코랩 GPU에서 돌린다. CPU에서는 에폭당 약 15분이라 노트북으로는 하루 넘게 걸린다.
#
# 코랩 사용법 (런타임 → 런타임 유형 변경 → GPU)
#   1) 저장소 받기:  !git clone -b shinyoung https://github.com/uriyoung11/VehicleDynamics.git
#      (비공개면 노트북의 저장소 폴더를 zip으로 올려 풀어도 된다)
#   2) colab_inputs.zip 업로드 (노트북에서 만든 것: D1 실험 파일 + Thunderhill West 트랙 파일) 후
#      !cd VehicleDynamics && unzip -o ../colab_inputs.zip
#   3) !cd VehicleDynamics && bash colab/run_r3_colab.sh
#   4) 끝나면 r3_checkpoints.zip 을 내려받아 노트북 저장소 폴더에서 풀기 → python models/ladder/run_ablation.py eval
set -e
cd "$(dirname "$0")/.."
python -c "import torch; assert torch.cuda.is_available(), 'GPU 런타임이 아니다'; print('GPU', torch.cuda.get_device_name(0))"
[ -f data/tracks/ThunderhillWest_centerline.csv ] || { echo "트랙 파일 없음: colab_inputs.zip 을 풀었는지 확인"; exit 1; }
ls data/scirob_submission/Model_Learning/data/exp/exp_data/dry/exp_*.npz > /dev/null || { echo "D1 파일 없음: colab_inputs.zip 확인"; exit 1; }
python colab/get_d7.py
while IFS=';' read -r e r s; do
  [ -z "$e" ] && continue
  echo "== $e $r seed $s"
  python -W ignore models/ladder/run_ablation.py train --exp "$e" --rung "$r" --seed "$s" --device cuda --threads 2
done < colab/r3_jobs.txt
zip -qr r3_checkpoints.zip checkpoints/ladder/*/R3_s*.pt checkpoints/ladder/*/R3_s*.json
echo "완료: r3_checkpoints.zip"

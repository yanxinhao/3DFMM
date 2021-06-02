###
 # @Author: yanxinhao
 # @Email: 1914607611xh@i.shu.edu.cn
 # @LastEditTime: 2021-06-02 07:23:16
 # @LastEditors: yanxinhao
 # @Description: 
### 
cd ..
source activate pytorch3d
# val
python experiments/flame_fit.py -i ./Data/dave_dvp/val \
    -o ./Results/dave_dvp/val/
# train
python experiments/flame_fit.py -i ./Data/dave_dvp/train \
    -o ./Results/dave_dvp/train/
# test
python experiments/flame_fit.py -i ./Data/dave_dvp/test \
    -o ./Results/dave_dvp/test/
cd scripts
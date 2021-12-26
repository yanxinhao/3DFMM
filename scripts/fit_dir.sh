###
 # @Author: yanxinhao
 # @Email: 1914607611xh@i.shu.edu.cn
 # @LastEditTime: 2021-07-15 17:27:33
 # @LastEditors: yanxinhao
 # @Description: 
### 
cd ..
source activate pytorch3d
# val
python experiments/flame_fit.py -i ./Data/dave_dvp/val \
    -o ./Results/dave_dvp_principle0/val/
# train
python experiments/flame_fit.py -i ./Data/dave_dvp/train \
    -o ./Results/dave_dvp_principle0/train/
# test
python experiments/flame_fit.py -i ./Data/dave_dvp/test \
    -o ./Results/dave_dvp_principle0/test/
cd scripts
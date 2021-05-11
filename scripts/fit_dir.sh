###
 # @Author: yanxinhao
 # @Email: 1914607611xh@i.shu.edu.cn
 # @LastEditTime: 2021-05-10 14:35:50
 # @LastEditors: yanxinhao
 # @Description: 
### 
conda init
conda activate pytorch3d
# train
python experiments/flame_fit.py -i ./Data/dave_dvp/train -o ./Results/dave_dvp/train/
# test
python experiments/flame_fit.py -i ./Data/dave_dvp/test -o ./Results/dave_dvp/test/
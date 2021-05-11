###
 # @Author: yanxinhao
 # @Email: 1914607611xh@i.shu.edu.cn
 # @LastEditTime: 2021-05-10 14:35:38
 # @LastEditors: yanxinhao
 # @Description: 
### 
conda init
conda activate pytorch3d
bash scripts/camera_calib.sh
bash scripts/identity_fit.sh
bash scripts/fit_dir.sh

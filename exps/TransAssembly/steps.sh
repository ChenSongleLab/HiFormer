#!/usr/bin/env bash



CURDIR=$(cd $(dirname $0); pwd)
cd ${CURDIR}
echo 'The work dir is: ' ${CURDIR}



cd ${CURDIR}/../../
HDFS_ROOT=hdfs://haruna/home/byte_arnold_lq_mlnlc/user/zhangrufeng/
echo 'Start Downloading Data.'
hadoop fs -get ${HDFS_ROOT}/datasets/3DPartAssembly/3DPartAssembly_partnet.zip ./
echo 'Finish Downloading Data.'
unzip ./3DPartAssembly_partnet.zip




cd ${CURDIR}
HDFS_ROOT=hdfs://haruna/home/byte_arnold_lq_mlnlc/user/zhangrufeng/
hadoop fs -get ${HDFS_ROOT}/softwares/Anaconda3-2021.11-Linux-x86_64.sh
export http_proxy=10.20.47.147:3128 https_proxy=10.20.47.147:3128 no_proxy=code.byted.org
bash ./Anaconda3-2021.11-Linux-x86_64.sh -b -p ${CURDIR}/anaconda3
source ${CURDIR}/anaconda3/bin/activate



















export http_proxy=10.20.47.147:3128 https_proxy=10.20.47.147:3128 no_proxy=code.byted.org
echo 'Start building.'
cd ${CURDIR}

conda env create -f TransAssembly.yaml
conda activate TransAssembly
pip3 install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install -r TransAssembly.txt
cd ${CURDIR}/../utils/cd
python3 setup.py build develop
echo 'Finish building.'

:<<! 
[script description]: lanuch the web page

note 1. you can config the knn-model of the web page
by config model_configs.yml file.
!
# this line speed up faiss
export OMP_WAIT_POLICY=PASSIVE

PROJECT_PATH=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../..
APP_PATH=$PROJECT_PATH/knnbox-scripts/vanilla-knn-mt-visual/src/app.py

CUDA_VISIBLE_DEVICES=0 streamlit run $APP_PATH --server.port 8999


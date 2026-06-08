# PROCESS TO MERGE SMPL+H AND MANO PARAMETERS

source scripts/activate_wbt.sh

cd src/motion_convertor/third_party
git clone https://github.com/vchoutas/smplx.git
cd smplx
conda activate interact
conda install -c conda-forge chumpy
pip install --no-deps -e .

python tools/merge_smplh_mano.py \
--smplh-fn ../../../../data/00_raw_datasets/models/smplh/female/model.npz \
--mano-left-fn ../../../../data/00_raw_datasets/models/mano_v1_2/models/MANO_LEFT.pkl \
--mano-right-fn ../../../../data/00_raw_datasets/models/mano_v1_2/models/MANO_RIGHT.pkl \
--output-folder ../../../../data/00_raw_datasets/models/smplh/
mv ../../../../data/00_raw_datasets/models/smplh/model.pkl ../../../../data/00_raw_datasets/models/smplh/SMPLH_FEMALE.pkl

python tools/merge_smplh_mano.py \
--smplh-fn ../../../../data/00_raw_datasets/models/smplh/male/model.npz \
--mano-left-fn ../../../../data/00_raw_datasets/models/mano_v1_2/models/MANO_LEFT.pkl \
--mano-right-fn ../../../../data/00_raw_datasets/models/mano_v1_2/models/MANO_RIGHT.pkl \
--output-folder ../../../../data/00_raw_datasets/models/smplh/
mv ../../../../data/00_raw_datasets/models/smplh/model.pkl ../../../../data/00_raw_datasets/models/smplh/SMPLH_MALE.pkl

python tools/merge_smplh_mano.py \
--smplh-fn ../../../../data/00_raw_datasets/models/smplh/neutral/model.npz \
--mano-left-fn ../../../../data/00_raw_datasets/models/mano_v1_2/models/MANO_LEFT.pkl \
--mano-right-fn ../../../../data/00_raw_datasets/models/mano_v1_2/models/MANO_RIGHT.pkl \
--output-folder ../../../../data/00_raw_datasets/models/smplh/
mv ../../../../data/00_raw_datasets/models/smplh/model.pkl ../../../../data/00_raw_datasets/models/smplh/SMPLH_NEUTRAL.pkl
export covfit_stuff_path="./notebooks/plm_circuits/covfit_stuff"

echo "installying ipykernel (make sure you're already in exploratory_env conda environment!!)"
conda install ipykernel

echo "installing nextstrain"
curl -fsSL --proto '=https' https://nextstrain.org/cli/installer/linux | bash
printf '\n%s\n' 'eval "$("/home/ubuntu/.nextstrain/cli-standalone/nextstrain" init-shell bash)"' >> ~/.bashrc
eval "$("/home/ubuntu/.nextstrain/cli-standalone/nextstrain" init-shell bash)"

echo "Downloading data"
snakemake --cores all data/pathogen/{"sars_cov_2_spike","sars_cov_2_spike_asia","sars_cov_2_spike_africa","sars_cov_2_spike_europe","sars_cov_2_spike_north_america","sars_cov_2_spike_oceania","sars_cov_2_spike_south_america"}/branches.tsv

echo "downloading covfit"
echo "models will be located in ${covfit_stuff_path}"
wget  -O "${covfit_stuff_path}/covfit_cli_20241007.tar.gz" "https://zenodo.org/records/14438178/files/covfit_cli_20241007.tar.gz"
tar -xf ${covfit_stuff_path}/covfit_cli_20241007.tar.gz -C ${covfit_stuff_path}
rm ${covfit_stuff_path}/covfit_cli_20241007.tar.gz
cp -r ${covfit_stuff_path}/CoVFit_CLI/_internal/files/models ${covfit_stuff_path}
rm -rf ${covfit_stuff_path}/CoVFit_CLI



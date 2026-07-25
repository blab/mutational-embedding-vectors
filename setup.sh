export covfit_stuff_path="./notebooks/plm_circuits/covfit_stuff"

# echo "Downloading data"
# snakemake --cores all all_pathogens

echo "downloading covfit"
echo "models will be located in ${covfit_stuff_path}"
# wget  -O "${covfit_stuff_path}/covfit_cli_20241007.tar.gz" "https://zenodo.org/records/14438178/files/covfit_cli_20241007.tar.gz"
tar -xf ${covfit_stuff_path}/covfit_cli_20241007.tar.gz -C ${covfit_stuff_path}
rm ${covfit_stuff_path}/covfit_cli_20241007.tar.gz
cp -r ${covfit_stuff_path}/CoVFit_CLI/_internal/files/models ${covfit_stuff_path}
rm -rf ${covfit_stuff_path}/CoVFit_CLI



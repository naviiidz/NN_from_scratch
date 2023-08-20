# NN_from_scratch
Neural Networks Sample codes from scratch

# How to download YCB dataset:  
Git clone this repo:  
`git clone https://github.com/hsp-iit/fast-ycb.git  
`  
Then:  
`cd YOUR_CLONE_LOCATION/fast-ycb/tools/download  
`  
Now use the .bash to download dataset:  
`bash download_dataset.sh  
`  
No need to download them all. Just a few of the objects are enough. The files are downloaded as splitted zip files with .z0 ... .zn  

Now install these:  
`sudo apt-get install p7zip-full  
`  
Now merge them:  
`cat filename.z* > combined_archive.z
`  
Then extract:  
`7z x combined_archive.z
`  

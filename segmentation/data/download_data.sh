# Background images

curl -L https://byu.box.com/shared/static/dc16feb1nhswm3imtce7f6r5ai7d0i6w --output bg.tar

# Hand images

curl -L https://byu.box.com/shared/static/moy2j92p9j9tv8mw8c1dgafn4r4pod19 --output train1.tar
curl -L https://byu.box.com/shared/static/jdto18tt4q89pdmn2l2wiiics2ltdr54 --output train2.tar
curl -L https://byu.box.com/shared/static/0yj1iqlsmt7aw7odp3ns50e39nmer4vo --output train3.tar
curl -L https://byu.box.com/shared/static/fr3lcjscu5xit6qbyqdooy6pi6uyk1q3 --output train4.tar

# Extract files and delete 

tar -xvf bg.tar -C ./bg_imgs
rm -rf bg.tar

tar -xvf train1.tar -C ./train_imgs
rm -rf train1.tar

tar -xvf train2.tar -C ./train_imgs
rm -rf train2.tar

tar -xvf train3.tar -C ./train_imgs
rm -rf train3.tar

tar -xvf train4.tar -C ./train_imgs
rm -rf train4.tar
function img=resize_img(img_name)
    img=imread(img_name);
    img=imresize(img,[28,28]);
end
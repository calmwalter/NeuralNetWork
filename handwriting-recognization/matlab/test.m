load("w1");
load("w2");
load("b1");
load("b2");
img=im2double(resize_img("8.jpg"));
img=rgb2gray(img);
img=imbinarize(img);
img=imcomplement(img);
imshow(img);
%test
%feed fowward 
aa1=reshape(img,[1,28*28]);
aa2=sigmoid(aa1*transpose(w1)+b1);
aa3=sigmoid(aa2*transpose(w2)+b2);
fprintf("%f ",transpose(aa3));
fprintf("\n");
max=1;
for j=2:10
    if aa3(1,j)>aa3(1,max)
        max=j;
    end
end
fprintf("%d\n",max-1);


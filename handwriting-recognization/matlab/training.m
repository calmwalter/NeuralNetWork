%network model 784*25*10
l1=784;
l2=25;
l3=10;
%input image 60000*784
x=transpose(loadMNISTImages("train-images.idx3-ubyte"));

%input label 60000*10
y=zeros(60000,10);
tmp=loadMNISTLabels("train-labels.idx1-ubyte");
for i=1:60000
    y(i,tmp(i,1)+1)=1;
end 

%learning rate
alpha=2;
%weight for every level
epsilon_init = 0.12;
w1 =rand(l2, l1) * 2 * epsilon_init - epsilon_init;
w2 =rand(l3, l2) * 2 * epsilon_init - epsilon_init;

%iter number
iter_number=6000;
%photo numeber
number=length(x);

%initial bios
b1=rand() * 2 * epsilon_init - epsilon_init;
b2=rand() * 2 * epsilon_init - epsilon_init;

%start to iter
for pred=1:50
    for i=1:iter_number
        %initialize every ativation value
        a1=zeros(1,l1);
        a2=zeros(1,l2);
        a3=zeros(1,l3);
        j=0;
        delta2=zeros(l3,l2);
        delta1=zeros(l2,l1);
        bls1=0;
        bls2=0;
        for k=1+(i-1)*10:i*10
            %feed fowward 
            a1=x(k,:);
            z2=a1*transpose(w1)+b1;
            a2=sigmoid(z2);
            z3=a2*transpose(w2)+b2;
            a3=sigmoid(z3);

            %cost function
            j=j-y(k,:)*transpose(log(a3))-(1-y(k,:))*transpose(log(1-a3));

            %Backpropagation
            los3=a3-y(k,:);
            los2=los3*w2.*grad_sigmoid(z2);

            delta2=delta2+transpose(los3)*a2; 
            delta1=delta1+transpose(los2)*a1;
            bls2=bls2+sum(los3);
            bls1=bls1+sum(los2);
            
        end
        j=j./number;
        D1=delta1./number;
        D2=delta2./number;
        Db1=bls1./number;
        Db2=bls2./number;

        %gradient
        w1=w1-alpha.*D1;
        w2=w2-alpha.*D2;
        b1=b1-alpha.*Db1;
        b2=b2-alpha.*Db2;

    end
    %output Jval
    fprintf("%.6f\n",j);
    test=transpose(loadMNISTImages("t10k-images.idx3-ubyte"));
    testlabel=loadMNISTLabels("t10k-labels.idx1-ubyte");
    cnt=0;
    for i=1:length(test)
        %feed fowward 
        a1=test(i,:);
        z2=a1*transpose(w1)+b1;
        a2=sigmoid(z2);
        z3=a2*transpose(w2)+b2;
        a3=sigmoid(z3);
        %fprintf("%f ",transpose(a3));
        %fprintf("\n");
        max=1;
        for j=2:10
            if a3(1,j)>a3(1,max)
                max=j;
            end
        end
        fprintf("%d %d\n",max,testlabel(i,1)+1);
        if max==testlabel(i,1)+1
            cnt=cnt+1;
        end
    end
    fprintf("%d %d %f\n",cnt,length(test),cnt/length(test));
end



clc
clear all ;
y = audioread('03-01-05-02-02-01-02.wav');
tau=1;
xp=y(1:end-tau);
xm=y(1+tau:end);
SD1=std(xp-xm)/sqrt(2);
SD2=std(xp+xm)/sqrt(2);
vol=pi*SD1*SD2
info=[SD1 SD2  vol SD2/SD1];
plot(xp,xm,'r');
%axis([0 1.2 0 1.2 ]);
xlabel('Xn');
ylabel('Xn+1');
grid on
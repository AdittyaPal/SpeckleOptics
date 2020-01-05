x=randn(10);
avg=zeros(201);
avg(1)=mean2(x);
for n=1:200
  x1=randn(10);
  x=plus(x,x1);
  avg(n+1)=mean2(x)/(n+1);
 end
 plot(avg);

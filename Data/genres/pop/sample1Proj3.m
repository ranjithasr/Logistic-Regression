clc
clear all

for i=1:100
    if i<=10
X=['C:\Users\M4-1012TX\Documents\Machine Learning\LRProj3\genres\jazz\jazz.0000',num2str(i-1),'.au'];
    else     
X=['C:\Users\M4-1012TX\Documents\Machine Learning\LRProj3\genres\jazz\jazz.000',num2str(i-1),'.au'];  
    end
[y1(:,i),Fs]=audioread(X);
Y1(:,i)=abs(fft(y1(:,i)));

end
%%
y2=zeros(661794,100);
Y2=zeros(661794,100);

for i=1:100
    if i<=10
X=['C:\Users\M4-1012TX\Documents\Machine Learning\LRProj3\genres\classical\classical.0000',num2str(i-1),'.au'];
    else     
X=['C:\Users\M4-1012TX\Documents\Machine Learning\LRProj3\genres\classical\classical.000',num2str(i-1),'.au'];  
    end
[y2(:,i),Fs]=audioread(X);
Y2(:,i)=fft(y2(:,i));
end
%%

for i=1:100
    if i<=10
X=['C:\Users\M4-1012TX\Documents\Machine Learning\LRProj3\genres\country\country.0000',num2str(i-1),'.au'];
    else     
X=['C:\Users\M4-1012TX\Documents\Machine Learning\LRProj3\genres\country\country.000',num2str(i-1),'.au'];  
    end
[y3(:,i),Fs]=audioread(X);
Y3(:,i)=fft(y3(:,i));
end

%%


for i=1:100
    if i<=10
X=['C:\Users\M4-1012TX\Documents\Machine Learning\LRProj3\genres\pop\pop.0000',num2str(i-1),'.au'];
    else     
X=['C:\Users\M4-1012TX\Documents\Machine Learning\LRProj3\genres\pop\pop.000',num2str(i-1),'.au'];  
    end
[y4(:,i),Fs]=audioread(X);
Y4(:,i)=fft(y4(:,i));
end

%%
for i=1:100
    if i<=10
X=['C:\Users\M4-1012TX\Documents\Machine Learning\LRProj3\genres\rock\rock.0000',num2str(i-1),'.au'];
    else     
X=['C:\Users\M4-1012TX\Documents\Machine Learning\LRProj3\genres\rock\rock.000',num2str(i-1),'.au'];  
    end
[y5(:,i),Fs]=audioread(X);
Y5(:,i)=fft(y5(:,i));
end
%%

for i=1:100
    if i<=10
X=['C:\Users\M4-1012TX\Documents\Machine Learning\LRProj3\genres\metal\metal.0000',num2str(i-1),'.au'];
    else     
X=['C:\Users\M4-1012TX\Documents\Machine Learning\LRProj3\genres\metal\metal.000',num2str(i-1),'.au'];  
    end
[y6(:,i),Fs]=audioread(X);
Y6(:,i)=fft(y6(:,i));
end
%%




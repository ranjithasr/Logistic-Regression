%% Reading the input file with audioread and collecting all the samples
clc;
clear all;
x=zeros(1000,600);
for i=1:100
    if i<=10
Xr=['C:\Users\M4-1012TX\Documents\Machine Learning\LRProj3\genres\rock\rock.0000',num2str(i-1),'.au'];
    else
Xr=['C:\Users\M4-1012TX\Documents\Machine Learning\LRProj3\genres\rock\rock.000',num2str(i-1),'.au'];
    end


 [temp,Fs]=audioread(Xr);
    fft_coefficient=fft(temp,1000);
    abs_fft=abs(fft_coefficient);
   x(:,i)=abs_fft;
nr=i;
end
for i=1:100
  nr=nr+1;  
  if i<=10
  Xr=['C:\Users\M4-1012TX\Documents\Machine Learning\LRProj3\genres\jazz\jazz.0000',num2str(i-1),'.au'];
  else
  Xr=['C:\Users\M4-1012TX\Documents\Machine Learning\LRProj3\genres\jazz\jazz.000',num2str(i-1),'.au'];
  end
  [temp,Fs]=audioread(Xr);
    fft_coefficient=fft(temp,1000);
    abs_fft=abs(fft_coefficient);
   x(:,i+100)=abs_fft;
    
  
end

for i=1:100
  nr=nr+1;  
  if i<=10
  Xr=['C:\Users\M4-1012TX\Documents\Machine Learning\LRProj3\genres\pop\pop.0000',num2str(i-1),'.au'];
  else
  Xr=['C:\Users\M4-1012TX\Documents\Machine Learning\LRProj3\genres\pop\pop.000',num2str(i-1),'.au'];
  end
  [temp,Fs]=audioread(Xr);
    fft_coefficient=fft(temp,1000);
    abs_fft=abs(fft_coefficient);
   x(:,i+200)=abs_fft;
  
end

for i=1:100
  nr=nr+1;  
  if i<=10
  Xr=['C:\Users\M4-1012TX\Documents\Machine Learning\LRProj3\genres\metal\metal.0000',num2str(i-1),'.au'];
  else
  Xr=['C:\Users\M4-1012TX\Documents\Machine Learning\LRProj3\genres\metal\metal.000',num2str(i-1),'.au'];
  end
  [temp,Fs]=audioread(Xr);
    fft_coefficient=fft(temp,1000);
    abs_fft=abs(fft_coefficient);
   x(:,i+300)=abs_fft;
   
end

for i=1:100
  nr=nr+1;  
  if i<=10
  Xr=['C:\Users\M4-1012TX\Documents\Machine Learning\LRProj3\genres\country\country.0000',num2str(i-1),'.au'];
  else
  Xr=['C:\Users\M4-1012TX\Documents\Machine Learning\LRProj3\genres\country\country.000',num2str(i-1),'.au'];
  end
  [temp,Fs]=audioread(Xr);
  fft_coefficient=fft(temp,1000);
    abs_fft=abs(fft_coefficient);
   x(:,i+400)=abs_fft;
    
end

for i=1:100
  nr=nr+1;  
  if i<=10
  Xr=['C:\Users\M4-1012TX\Documents\Machine Learning\LRProj3\genres\classical\classical.0000',num2str(i-1),'.au'];
  else
  Xr=['C:\Users\M4-1012TX\Documents\Machine Learning\LRProj3\genres\classical\classical.000',num2str(i-1),'.au'];
  end
  [temp,Fs]=audioread(Xr);
 fft_coefficient=fft(temp,1000);
    abs_fft=abs(fft_coefficient);
   x(:,i+500)=abs_fft;
   
end

x=x';
 [m,n]=size(x);
 X=[ones(m,1) x];
 
 %normalizing the initial input data
 max_fft=max(x);
    for i=2:1001
        X(:,i)=x(:,i-1)./max_fft(1,i-1);
       
    end
 
%%
%calculation of indices matrix
%10fold CV
 %taking a random matrix and repeating it 60 times 
Y1=[1;2;3;4;5;6;7;8;9;0];
Y2=repmat(Y1,60,1);
for i=1:10

  %calculating the train and the test indices from the repeated matrix
%for s=1:10
s=i;
   if(s==1)
        indices_test(:,s)=find(Y2==(s-1));
        indices_train(:,s)=find(Y2>(s-1));
   elseif(s==10)
        indices_test(:,s)=find(Y2==(s-1));
        
        indices_train(:,s)=find(Y2<(s-1));
   else
        indices_test(:,s)=find(Y2==(s-1));
        
        indices_train(:,s)=[find(Y2>(s-1));find(Y2<(s-1))];
   end
    %finding out the train and the test data 
     
    training_data=X(indices_train(:,s),:);
     testing_data =X(indices_test(:,s),:);

%%
%inintializing the weights to zero
    Weights=zeros(6,1001);

 initial_training_data=training_data;   
 training_data=training_data';
%%
%calculating the delta matix
a=ones(600,1);a(101:200)=2;
a(201:300)=3;a(301:400)=4;
a(401:500)=5;a(501:600)=6;

%applying the a matrix with the train indices matrix
k1=a(indices_train(:,s));

%initialixing the delta matrix (R) to zero
R=zeros(6,540);  
 

for l=1:540
     
   if k1(l)==1
        R(1,l)=1;
   end 
    if k1(l)==2
        R(2,l)=1;
    end    
     if k1(l)==3
        R(3,l)=1;
     end
         if k1(l)==4
        R(4,l)=1;
         end
         if k1(l)==5
        R(5,l)=1;
         end
         if k1(l)==6
        R(6,l)=1;
        end
end

%setting the epoch value to 100 and eita to 0.01
%calculations for the training data
%initializng the probability
    epochSize=1000;
    eita1=0.01;
    final_matrix=Weights*training_data;
    numerator=exp(final_matrix);
    denominator=1+sum(numerator(1:5,:));
 
   %calculating the probability using the formula for first iteration
    for e=1:540
  
  probability(:,e)= numerator(:,e)./(denominator(e));   
        
        probability(6,e)= 1/(denominator(e));
  
  end
%calculating the probability using the formula for each iteration
  
 for j=1:epochSize
     eita=eita1/(1+(j/epochSize));
     
  %updating the weights with the obtained values of eita and probability
   Weights=Weights + (eita*(((R - probability)*initial_training_data )- 0.001*(Weights)));
    
   final_matrix=Weights*training_data;
   numerator=exp(final_matrix);
   denominator=1+sum(numerator(1:5,:));
 for e=1:540
  
            probability(:,e)= numerator(:,e)./(denominator(e));        
  
        probability(6,e)= 1/(denominator(e));
  
  end
   
 end
 
 %normalizing the probability
 for g1=1:540
       probability(:,g1)= probability(:,g1)./(max( probability(:,g1)));

 end

 %calculations for the testing
 
testing_data=testing_data';

   test_final_matrix=Weights*testing_data;
   numerator_test=exp(test_final_matrix);
   denominator_test=1+sum(numerator_test(1:5,:));
%calculating the test probability
for k=1:60
       
          probability_test(:,k)= numerator_test(:,k)./(denominator_test(k));
        
         probability_test(6,k)= 1/(denominator_test(k));
        
end
 %normalizing the test probability

for f=1:60
         probability_test(:,f)= probability_test(:,f)./(max( probability_test(:,f)));
end

 
 %applying the a matrix with the indices matrix
 for q=1:60    
c_1(1,q)=a(indices_test(q));
 end

 %calculating the maximum of the indices 
[value,argmax]=max(probability_test);

%calculating the confusion matrix
confMatrix1=confusionmat(argmax,c_1);
 
%calculating the accuracy
accuracy1=100*((sum(diag(confMatrix1)))/60)
accuracy(i,1)=accuracy1;

end
%calculating the mean value of all the accuracies
avgAcc=mean(accuracy)






    




    
    




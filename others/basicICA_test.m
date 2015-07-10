
function [AA,W, icasig_tmp,tcorr]=basicICA_test(icadata,numOfPC,repN,extraICAOptions, reference,projection)


%extraICAOptions =  {'verbose','off','lrate',1e-5};
%         tempVar = ones(size(icadata, 1), 1)*mean(icadata);
%         icadata = icadata - tempVar;
%         clear tempVar;
 
if  ~exist('repN','var')
    repN=5;
end


       if exist('reference','var')
             if   exist('projection','var') 
              [V, Lambda] =icatb_pca_reference(icadata,1,numOfPC,reference,0, 'untranspose', 'no',projection);   
             % [V, Lambda] = ica_fuse_pca_reference(icadata, 1,numOfPC, reference,0, 'untranspose', 'no');  % function in fit toolbox   
             else
              [V, Lambda] = ica_fuse_pca_reference(icadata, 1,numOfPC, reference,0, 'untranspose', 'no');  % function in fit toolbox   variance option 
              
             end
             
       else  [V, Lambda] = icatb_v_pca(icadata, 1, numOfPC, 0, 'untranspose', 'no');
       
       end
       
      
           % whiten matrix
           [whitesigSNP, whiteMSNP, dewhiteMSNP] = icatb_v_whiten(icadata, V, Lambda, 'untranspose');
           
       for run=1:repN
           

          [icaAlgo, W, Ar, icasig_tmp] = icatb_icaAlgorithm(1,whitesigSNP,extraICAOptions);%  ica process
        
         %-- for testing sparse ICA
%            sparsethresh=2;
%            [weight,sphere] = runica_sparse(whitesigSNP,'sparse',sparsethresh,'posact','off','bias', 'off','block', round(size(whitesigSNP,2)/1));
%            W = weight*sphere;  icasig_tmpS = W*whitesigSNP;   Ar = pinv(W);
%         %  for i=1:numOfPC; figure;plot(icasig_tmpS(i,:)','.'); end
          
           AA=dewhiteMSNP*Ar;%figure;plot(AA)
           %% conver to z score and save
    %       [icasig_tmp] = icatb_convertToZScores(icasig_tmp);
     %      figure;plot(icasig_tmp','.'), title('Z acore')
           
           % Record the resultd from different runs
           if run == 1
                aveComp = icasig_tmp;
                loadingCoeff = AA;
                aveW=W;
                tcorr=zeros(1,numOfPC);
           else
           % sort the components based on COMPONENT correlation
                temp = corr(aveComp', icasig_tmp');
                rowt = zeros(1, numOfPC);
                colt = rowt; tcorr = rowt;
                for i=1:numOfPC
                    [t,loc]=max(abs(temp(:)));
                    [tempRow, tempCol] = find(abs(temp) == t(1));
                    rowt(i) = tempRow(1);
                    colt(i) = tempCol(1);
                    tcorr(i)=temp(loc(1));
                    temp(rowt(i),:)=0;
                    temp(:,colt(i))=0;
                end
                atemp=icasig_tmp(colt,:);
                wtemp=W(colt,:);
                ltemp = AA(:, colt);
                ind=find(tcorr<0);
                if ~isempty(ind)
                    atemp(ind, :)=atemp(ind, :).*(-1);
                    ltemp(:, ind)=ltemp(:, ind)*-1;
                    wtemp(ind, :)=wtemp(ind, :)*-1;
                end

                aveComp = aveComp(rowt, :) + atemp;
                loadingCoeff = loadingCoeff(:, rowt) + ltemp;
                aveW    = aveW(rowt, :)+wtemp;
           end
           % End for number of ICA runs
       end
       AA=loadingCoeff./repN;
       icasig_tmp=aveComp./repN;
       W=aveW./repN;
      
        
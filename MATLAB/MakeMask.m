imSize = 140;

vecSize = [imSize,1];
p = 2;
pcgte = 0.5;
distType = 2;
radius = 0;
disp = 1;
[pdf,val] = genPDF(vecSize, p, pcgte,distType,radius,disp);

iter = 100;
tol=1;
[maskVec,stat,N] = samplingPattern(pdf,iter,tol);
figure(1);
mask = transpose(repmat(maskVec, [imSize,1]));
sum(mask(:))/numel(mask(:));
imshow(mask)
% save('mask50', 'mask');
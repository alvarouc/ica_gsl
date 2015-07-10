
A = random('normal', 0,1, [1000,100]);
S = random('logistic', 0,1, [100,50000]);
X = A*S + random('normal', 0,1, [1000,50000]);

tic;
basicICA_test(X', 100, 1,{});
toc

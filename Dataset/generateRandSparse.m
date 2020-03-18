function [A,cond] = generateRandSparse(n,a,b)
%n is the dim of our matrix
%a is the lower boundary of eigenvalues and b the upper boundary

p = abs(rand())*.06;
rc = (b-a).*rand(n,1) + a;

B = sprandsym(n,p,rc);

figure();
subplot(1,2,1),spy(B),title('B')

q = symrcm(B);  %cuthill-mckee
A = B(q,q);

subplot(1,2,2),spy(A),title('B(p,p)')

rc = abs(rc);
cond = max(rc)/min(rc);

end
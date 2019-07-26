function result = InfoMutual(pd1, F1, pd2, F2, Entropy)
% I(X, Y) = H(X) + H(X|Y);
hx = Entropy(pd1, F1);
hy = Entropy(pd2, F2);
hxy = unionEntropy(pd1, F1, pd2, F2);

result = hx + hy - hxy;
end
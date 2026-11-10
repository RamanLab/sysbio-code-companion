function u = uniform(seed, n)
%UNIFORM  Random numbers that are identical in MATLAB and Python.
%   u = sbp.uniform(seed, n) returns n numbers in (0, 1), from the
%   "minimal standard" generator of Park and Miller (x <- 48271 x mod
%   2^31 - 1), started at the integer seed (1 to 2^31 - 2). Every product
%   is below 2^53, so the arithmetic is exact in double precision, and
%   sbp.uniform in Python returns the same numbers. Use it where the two
%   languages must make the same random choices; otherwise, use rand.

m = 2147483647;
x = mod(seed, m);
u = zeros(1, n);
for k = 1:n
    x = mod(48271 * x, m);
    u(k) = x / m;
end
end

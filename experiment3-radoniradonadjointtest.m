% Dimensions of the matrices
n = 400; % size of the matrix (e.g. 400x400)
tol = 10^-10; % tolerance for the break condition

% 1. Creating random matrix
A = randn(n, n);
for i=1:n
    for j=1:n
        if (i-n/2)^2+(j-n/2)^2 > (n/4)^2
            A(i,j) = 0;
        end
    end
end 

% 2. Radon transform of the matrix (A)
theta = linspace(0, 180, 41) % angles for the Radon transform
R = radon(A, theta);

mn = size(R);
B = randn(mn(1), mn(2));
for i=1:mn(1)
    for j=1:mn(2)
        if (i-mn(1)/2)^2+(j-mn(2)/2)^2 > (n/4)^2
            B(i,j) = 0;
        end
    end
end 

iR = iradon(B,theta,'linear','None',1,100);

% 3. Determing the normdifference
% Parameters
maxiter = 1000; % maximum number of iterations

% Initialization with random normalized "vectors"
u = randn(mn(1), mn(2));
u = u / norm(u, "fro");

v = randn(n, n);
v = v / norm(v, "fro");

% Calculate the estimator
uAVv = sum(dot(u, radon(v, theta))) - sum(dot(iradon(u,theta,'linear','None',1,n), v));
if uAVv < 0
    u = -u;
end

% list of estimates
nAV = [];

% list of values of a
a_s = [];

nAV(end+1) = uAVv;
nostep = 0;

for i = 1:maxiter
    % Sampling x orthogonal to v
    x = randn(n,n);
    x = x - (sum(dot(x, v)) * v);
    x = x / norm(x, "fro");

    w = randn(mn(1), mn(2));
    w = w - (sum(dot(w, u)) * u);
    w = w / norm(w, "fro");

    vOld = v;

    a = sum(dot(w, radon(v, theta))) - sum(dot(iradon(w,theta,'linear','None',1,n), v)) + sum(dot(u, radon(x, theta))) - sum(dot(iradon(u,theta,'linear','None',1,n), x));
    b = 2 * (sum(dot(w, radon(x, theta)))- sum(dot(iradon(w,theta,'linear','None',1,n), x)) - sum(dot(u, radon(v, theta))) + sum(dot(iradon(u,theta,'linear','None',1,n), v)));

    if a ~= 0
        tau = sign(a) * (b / (2 * abs(a)) + sqrt(b^2 / (4 * a^2) + 1));
        v = v + tau * x;
        v = v / norm(v, "fro");
        u = u + tau * w;
        u = u / norm(u, "fro");
    else
        if b > 0
            u = w;
            v = x;
        end
    end

    % Update estimate
    uAVv = sum(dot(u, radon(v, theta))) - sum(dot(iradon(u,theta,'linear','None',1,n), v));

    % Save values
    nAV(end+1) = uAVv;
    a_s(end+1) = a;

    % Check break condition
    if np.abs(a) > tol
        count = 0
    else 
        count = count + 1
    end
    if count == 100:
        break
    end
end

% 4. Determing the norm using the method described in aeXiv:2410.08297
% Parameters
maxiter = 100000; % maximum number of iterations

% Initialization with random normalized "vectors"
v = randn(n, n);
v = v / norm(v, "fro");

% list of estimates
nA = [];

% list of values of a
a_s = [];

nA(end+1) = abs(uAVv);
nostep = 0;

Av = radon(v, theta);
Ax = radon(x, theta);

for i = 1:maxiter
    % Sampling x orthogonal to v
    x = randn(n,n);
    x = x - (sum(dot(x, v)) * v);
    x = x / norm(x, "fro");

    vOld = v;

    a = sum(dot(Av, Ax));
    b = norm(Ax, "fro")^2 - norm(Av, "fro")^2;

    if a ~= 0
        tau = sign(a) * (b / (2 * abs(a)) + sqrt(b^2 / (4 * a^2) + 1));
        v = v + tau * x;
        v = v / norm(v, "fro");
    else
        if b > 0
            v = x;
        end
    end

    Av = radon(v, theta);
    Ax = radon(x, theta);

    % Update estimate
    estimate = norm(Av, "fro")

    % Save values
    nA(end+1) = estimate;
    a_s(end+1) = a;

    % Check break condition
    if np.abs(a) > tol
        count = 0
    else 
        count = count + 1
    end
    if count == 100:
        break
    end
end

% 5. Print result
disp('Relative measurement of the adjoint mismatch is:')
disp(nAV(-1)/nA(-1))
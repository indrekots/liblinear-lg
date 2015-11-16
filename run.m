addpath './liblinear/matlab';
addpath './specfun/inst'

function a = testIf(x)
     if x == 1
        a = 1;
     else 
        a = -1;
     end
 end

 function cost = lgCost(w, X, y, C)
 	cost = (0.5 * w*w' + C * sum(log(1 + exp(-(y .* (X*w'))))))/size(y, 1);
 end

%data = load('data.csv');
%y = arrayfun(@testIf, data(:, 1));
%X = [ones(size(data, 1), 1), data(:, 2:end)];
%
%y_train = y(1:5000);
%X_train = X(1:5000, :);
%y_test = y(5001:end);
%X_test = X(5001:end, :);
%
%liblinear_options = '-s 0';
%
%model = train(y_train, sparse(X_train), liblinear_options);

%sig = sigmoid(X_test*model.w');
%Jsum = sum((-y_test .* log(sig)) - ((1 - y_test) .* log(1 - sig)));
%J = Jsum / size(y_test, 1);

%p = predict(y_test, sparse(X_test), model);

training_m = 20000;
data = load('data_new.csv');
X = data(:, 2:end);
%X = data(:, 2:end);
y = arrayfun(@testIf, data(:, 1));
x_train = X(1:training_m, :);
y_train = y(1:training_m);
x_test = X(training_m + 1:end, :);
y_test = y(training_m + 1:end);

C = 1;
B = 1;
liblinear_options = cstrcat('-s 0 ', ' -B ', num2str(B), ' -c ', num2str(C), ' -q ');
%liblinear_options = '-s 0 -B 1 -c 1';
l_curve = [];

%for i = 1:training_m/10
%	x_iter = x_train(1:i*10, :);
%	y_iter = y_train(1:i*10);
%	model = train(y_iter, sparse(x_iter), liblinear_options);
%	yt = lgCost(model.w, [x_iter, ones(i*10, 1)], y_iter, C);
%	ycv = lgCost(model.w, [x_test, ones(size(x_test, 1), 1)], y_test, C);
%	
%	pt = mean(double(predict(y_iter, sparse(x_iter), model) == y_iter));
%	pcv = mean(double(predict(y_test, sparse(x_test), model) == y_test));
%	
%	l_curve = [l_curve; i, yt, ycv, pt, pcv];
%end
%
%figure(1);
%plot(1:training_m/10, l_curve(:, 2), 'linewidth', 2, 1:training_m/10, l_curve(:, 3), 'linewidth', 2);
%title('Learning curve for linear regression')
%legend('Train', 'Cross Validation')
%xlabel('Number of training examples')
%ylabel('Error')
%%axis([0 13 0 150])
%
%figure(2);
%plot(1:training_m/10, l_curve(:, 4), 'linewidth', 2, 1:training_m/10, l_curve(:, 5), 'linewidth', 2);
%title('Accuracy for linear regression')
%legend('Train', 'Cross Validation')
%xlabel('Number of training examples')
%ylabel('Accuracy')

model = train(y_train, sparse(x_train), liblinear_options);
[p, a, d] = predict(y_test, sparse(x_test), model);
sum(p == y_test)

%fprintf('\nProgram paused. Press enter to continue.\n');
%pause;

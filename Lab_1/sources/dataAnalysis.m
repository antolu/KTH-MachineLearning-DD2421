load data
close all

fractions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

mean1 = zeros(6, 1);
mse1 = zeros(6, 1);
for i=1:6
    mean1(i) = mean(monk1(:, i));
    mse1(i) = immse(mean1(i)*ones(length(monk1(:, i)), 1), monk1(:, i));
end

mean3 = zeros(6, 1);
mse3 = zeros(6, 1);
for i=1:6
    mean3(i) = mean(monk3(:, i));
    mse3(i) = immse(mean3(i)*ones(length(monk3(:, 1)), 1), monk3(:, 1));
end

figure;
plot(fractions, mean1, 'xr', 'MarkerSize', 8');
title('Mean error of validation data on pruned tree, monk1');
axis([0 1 0.6 1])
grid on
xlabel('Fraction of test data split')
ylabel('Validation data mean error')
set(gca,'Color',[0.95 0.95 0.95], 'FontSize', 12);

figure;
plot(fractions, mse1, 'xr', 'MarkerSize', 8');
title('MSE of validation error on pruned tree, monk1');
axis([0 1 0 0.01])
grid on
xlabel('Fraction of test data split')
ylabel('Validation error MSE')
set(gca,'Color',[0.95 0.95 0.95], 'FontSize', 12);

figure;
plot(fractions, mean3, 'xr', 'MarkerSize', 8');
title('Mean error of validation data on pruned tree, monk3');
axis([0 1 0.6 1])
grid on
xlabel('Fraction of test data split')
ylabel('Validation data mean error')
set(gca,'Color',[0.95 0.95 0.95], 'FontSize', 12);

figure;
plot(fractions, mse3, 'xr', 'MarkerSize', 8');
title('MSE of validation error on pruned tree, monk3');
axis([0 1 0 0.01])
grid on
xlabel('Fraction of test data split')
ylabel('Validation error MSE')
set(gca,'Color',[0.95 0.95 0.95], 'FontSize', 12);
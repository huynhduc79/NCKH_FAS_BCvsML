% Script so sánh thời gian chạy giữa BC và ML
N_test = 20:50:2000; % Thử với N từ 20 đến 2000
time_BC = zeros(length(N_test), 1);
time_ML = zeros(length(N_test), 1);

addpath('Core/')

W = 5; mu2 = 0.97;
load('trained_net_softmax.mat'); % Load mô hình ML

for i = 1:length(N_test)
    N = N_test(i);
    
    % --- ĐO THỜI GIAN BC ---
    tic;
    Sigma_jakes = toeplitz(besselj(0, 2*pi*(0:N-1)*W/(N-1)));
    rho = sort(eig(Sigma_jakes), 'descend');
    Num_eig = sum(rho > N/100);
    L_BC = BlockCorrelation(N, rho, Num_eig, mu2);
    time_BC(i) = toc;
    
    % --- ĐO THỜI GIAN ML ---
    tic;
    input_ML = [W, 2, 0, N]; % [W, U, gamma_dB, N]
    P_predicted = predict(net, input_ML); 
    L_ML = round(P_predicted * N);
    time_ML(i) = toc;
end

% Vẽ đồ thị
figure;
plot(N_test, time_BC, 'b-o', 'LineWidth', 2); hold on;
plot(N_test, time_ML, 'r-*', 'LineWidth', 2);
xlabel('Số lượng cổng (N)');
ylabel('Thời gian tính toán (giây)');
legend('Thuật toán BC (Toán học)', 'Mô hình ML (Dự đoán)');
title('So sánh độ phức tạp tính toán: BC vs ML');
grid on;